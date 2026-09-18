"""Classify a generation-tagged capture, lane by lane.

    python3 classify_flaky.py flaky gen0=hw_g0.bin gen1=hw_g1.bin ...

Reports, per generation and per 64-kernel block, how many lanes were correct,
never written, zero, stale from another generation, or garbage. Staleness is
named with the generation it came from.
"""
import json, os, sys
import numpy as np


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    root = sys.argv[1]
    man = json.load(open(os.path.join(root, "manifest.json")))
    N, M, TAG = man["N"], man["M"], man["tag_stride"]
    POISON, G = man["poison_int16"], man["gens"]
    base = np.arange(N) - N // 2
    goldens = {g: base + TAG * (g + 1) for g in range(G)}

    # health has its OWN N (small, one 64-block per core) so the gate cannot show
    # the block-prefix fault. Old kits without the field fall back to the gen N.
    HN = int(man.get("health_probe", {}).get("N", N))
    base_g = np.arange(HN) - HN // 2       # health golden is untagged

    def health_verdict(v, nb=None):
        nb = HN // 64 if nb is None else nb
        """Per-64-block pattern, not a lane total.

        A kernel with the half-kernel fault can only ever get 32 of every 64
        lanes, so gating on all-N lanes assumes a working kernel and can
        never pass. Gate on the SHAPE of the correct set instead: uniform 64
        per block is fully healthy, uniform 32 per block is healthy given the
        known fault, anything ragged is a degraded device.
        """
        ok = (v == base_g)
        # Count the LIVE half (lanes 0..31) explicitly. Counting a whole 64-block lets a
        # never-written lane in 32..63 match its golden BY CHANCE and inflate the total - seen
        # as [[32],[33],[32]] on a healthy device, which then read as "ragged". The dead half is
        # reported separately: it should be ~0, and a large count there would mean the
        # half-kernel fault did not apply.
        live = [int(ok[b * 64:b * 64 + 32].sum()) for b in range(nb)]
        dead = [int(ok[b * 64 + 32:(b + 1) * 64].sum()) for b in range(nb)]
        per = [f"{l}+{d}" for l, d in zip(live, dead)]
        u = set(live)
        if u == {32} and set(dead) == {32}:
            return "healthy", per, "64/64 per block (no half-kernel fault)"
        if u == {32}:
            return "healthy", per, ("32/64 per block (half-kernel fault, expected); "
                                    f"{sum(dead)} incidental match(es) in the dead half")
        return "degraded", per, f"ragged live halves {sorted(u)}"
    caps, last0, health = {}, None, {}
    for arg in sys.argv[2:]:
        k, _, path = arg.partition("=")
        if k in ("health_pre", "health_post"):
            raw = np.fromfile(path, dtype=np.int16)
            v = raw[:M * HN].reshape(-1, HN)[0].astype(np.int64)
            health[k] = v
            continue
        if k == "gen0_last":
            raw = np.fromfile(path, dtype=np.int16)
            last0 = raw[:M * N].reshape(-1, N)[0].astype(np.int64)
            continue
        g = int(k.replace("gen", ""))
        raw = np.fromfile(path, dtype=np.int16)
        if raw.size < N:
            raise SystemExit(f"{path}: {raw.size} int16, need >= {N}")
        caps[g] = raw[:M * N].reshape(-1, N)[0].astype(np.int64)

    nb = N // 64
    print(f"K={man['K']} N={N} blocks={nb} gens={sorted(caps)}\n")

    # ---- health gate --------------------------------------------------
    # A degraded device and a stale buffer produce the SAME wrong lanes.
    # Only the bracketing health runs tell them apart, so refuse to
    # interpret anything without them.
    if "health_pre" not in health:
        print("!! NO health_pre GIVEN. NPU state crosses process boundaries, "
              "so an ungated reading cannot distinguish contamination from "
              "staleness. Everything below is UNTRUSTED.\n")
    else:
        verdict, per, why = health_verdict(health["health_pre"])
        print(f"health_pre  {verdict}: {why}")
        print(f"            per-64-block: {per}")
        if verdict == "degraded":
            print("  -> VOID: the device was already degraded before this "
                  "measurement. Settle, re-run health until the pattern is "
                  "uniform, then measure. Do not read the lanes below.\n")
            raise SystemExit(2)
    if "health_post" in health:
        verdict, per, why = health_verdict(health["health_post"])
        print(f"health_post {verdict}: {why}")
        if verdict == "degraded":
            print(f"            per-64-block: {per}")
            print("  -> this config DEGRADED the device. This reading may "
                  "stand, but the next one is suspect until health passes.\n")
        else:
            print("  -> clean afterwards\n")
    for g in sorted(caps):
        v = caps[g]
        cat = np.full(N, "garbage  ", dtype="<U9")
        cat[v == goldens[g]] = "correct  "
        cat[v == POISON] = "unwritten"
        cat[v == 0] = "zero     "
        for og in range(G):
            if og == g:
                continue
            m = (v == goldens[og]) & (cat == "garbage  ")
            cat[m] = f"stale<-g{og}"[:9]

        tot = {c: int((cat == c).sum()) for c in np.unique(cat)}
        print(f"gen{g}: " + "  ".join(f"{k.strip()}={v_}" for k, v_ in
                                      sorted(tot.items(), key=lambda x: -x[1])))
        ok = cat == "correct  "
        per = [int(ok[b * 64:(b + 1) * 64].sum()) for b in range(nb)]
        print(f"      per-64-block correct: {per}")

        # Under the half-kernel fault, lanes 0..31 of every 64-block are the
        # ones that should survive and 32..63 are expected dead. A block
        # reading 28 means FOUR lanes that should have survived did not --
        # a third failure mode, distinct from "clean" and "block absent".
        # Where those lanes sit inside the block is the diagnostic.
        lo = np.array([ok[b * 64:b * 64 + 32].sum() for b in range(nb)])
        hi = np.array([ok[b * 64 + 32:(b + 1) * 64].sum() for b in range(nb)])
        print(f"      lanes 0-31  (expected live): {lo.tolist()}")
        if hi.any():
            print(f"      lanes 32-63 (expected dead): {hi.tolist()}  "
                  f"<- unexpected, the half-kernel fault did not apply here")
        partial = [b for b in range(nb) if 0 < lo[b] < 32]
        if partial:
            print(f"      PARTIAL blocks (0<live<32): {partial}")
            pos = np.concatenate([np.where(~ok[b * 64:b * 64 + 32])[0]
                                  for b in partial])
            u, c = np.unique(pos, return_counts=True)
            print(f"      bad lane positions within block: "
                  f"{dict(zip(u.tolist(), c.tolist()))}")
            print(f"      -> clustered positions mean a fixed sub-lane group "
                  f"is dropping; scattered means corruption, not geometry")
        bad = np.where(cat != "correct  ")[0]
        if bad.size:
            b0 = bad[0]
            print(f"      first bad lane {b0} (block {b0 // 64}): "
                  f"{cat[b0].strip()}, got {v[b0]}, want {goldens[g][b0]}")

    if last0 is not None and 0 in caps:
        first_ok = int((caps[0] == goldens[0]).sum())
        last_ok = int((last0 == goldens[0]).sum())
        print(f"\nWARM-UP TEST  gen0 first: {first_ok}/{N} correct   "
              f"gen0 last: {last_ok}/{N} correct")
        if last_ok > first_ok:
            print("  -> WARM-UP CONFIRMED. The first dispatch of a process is "
                  "bad and later ones are fine. Every reading taken as a fresh "
                  "process is contaminated; re-measure with one warm-up "
                  "dispatch discarded.")
        elif first_ok == last_ok == N:
            print("  -> no warm-up effect at this config; both dispatches clean.")
        else:
            print("  -> NOT warm-up: the config fails on a warm process too. "
                  "This failure is real.")

    print("\nIf any lane reads stale<-gN, the buffer is being reused before "
          "the previous dispatch retires -- that is a sequencing bug, and no "
          "amount of HMULT will fix it.")
    print("If lanes read 'unwritten', the dispatch never covered them: that "
          "IS a geometry/HMULT question.")


if __name__ == "__main__":
    main()
