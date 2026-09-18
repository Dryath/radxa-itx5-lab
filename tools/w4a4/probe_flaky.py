"""Generation-tagged probes -- tell staleness apart from silence.

The block fault turned out to be non-deterministic at some configs (K=3072
h6 gives 2/3, 3/3, 2/3 across identical runs). A pass/fail count cannot say
WHY a block failed, and the three explanations need opposite fixes:

    never written   -> the dispatch did not cover that block
    wrote zero      -> it was covered, the data was not fetched
    stale           -> a previous dispatch's result is still in the buffer
    misaddressed    -> real data, wrong lane

This kit makes those distinguishable. Each generation g encodes a different
golden into the SAME shape:

    C[m][n] == (n - N//2) + TAG*(g+1)

Run the generations back to back into the same output buffer, pre-filled with
POISON. Then every lane classifies itself:

    value == poison                  -> hardware never wrote this lane
    value == 0                       -> wrote, but fetched nothing
    value in generation g' != g      -> STALE from run g'  (decisive)
    value == correct for g           -> fine
    otherwise                        -> misaddressed / garbage

Staleness is the one hypothesis a repeat-count experiment can never confirm,
because a stale value from an identical previous run is indistinguishable
from a correct one. Different goldens per run is what breaks that tie.

    python probe_flaky.py --out flaky --k 3072 --n 1152 --gens 4
"""

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import pack_nibbles, b_native
from pack_rk3588 import encode_sum, encode_sum_dense, KTILE

TAG_MAX = 4096      # generation stride cap; keeps every golden inside int16
                    # The stride must also fit the encoder: a K-row int4
                    # column sums to at most 8*K, so small K needs a smaller
                    # stride. Chosen per shape, never assumed.
POISON = -21846     # 0xAAAA as int16: not producible by any int4 dot product


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="flaky")
    ap.add_argument("--k", type=int, default=3072)
    ap.add_argument("--n", type=int, default=1152)
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--encoding", default="dense", choices=["dense", "sparse"],
                    help="dense = pseudo-random values with exact column sums "
                         "(representative). sparse = prefix-of-ones, mostly "
                         "zeros -- atypical, and it exposed a data-dependent "
                         "fault. Run BOTH; density is the variable.")
    ap.add_argument("--health-n", type=int, default=192,
                    help="N for the health probe. MUST be small enough that "
                         "N/3 <= 64, i.e. exactly ONE 64-kernel block per core, "
                         "so the health shape cannot exhibit the block-prefix "
                         "fault it is meant to exclude. At N=1152 the health "
                         "probe showed core2 [32,32,32,32,0,0] on a HEALTHY "
                         "device and could never pass.")
    ap.add_argument("--health-k", type=int, default=512,
                    help="known-good shape used to prove the NPU is healthy "
                         "BEFORE and AFTER the measured config")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
    os.makedirs(root, exist_ok=True)
    K, N, M = a.k, a.n, a.m
    assert K % KTILE == 0

    enc = encode_sum_dense if a.encoding == "dense" else encode_sum
    base = np.arange(N) - N // 2
    bmax = int(np.abs(base).max())
    room = min(8 * K, 32000) - bmax
    TAG = min(TAG_MAX, room // a.gens)
    assert TAG > bmax, (f"K={K} too small for {a.gens} generations: stride "
                        f"{TAG} does not clear the base range {bmax}")
    TAG = 1 << (int(TAG).bit_length() - 1)          # round down to a power of two
    assert TAG > bmax, f"K={K}: stride {TAG} collides with base range {bmax}"
    print(f"tag stride {TAG} (base range +-{bmax}, encoder limit {8 * K})")

    gens = []
    for g in range(a.gens):
        tgt = base + TAG * (g + 1)
        assert np.abs(tgt).max() <= 8 * K, "target outside int4 sum range"
        Bq = enc(tgt, K)
        A = np.ones((M, K), dtype=np.int8)
        C = A.astype(np.int32) @ Bq.astype(np.int32)
        assert (C[0] == tgt).all() and (C == C[0]).all()

        d = os.path.join(root, f"gen{g}")
        os.makedirs(d, exist_ok=True)
        pack_nibbles(b_native(Bq, 64, KTILE)).tofile(f"{d}/B_nat_n64.i4lo.bin")
        pack_nibbles(A).tofile(f"{d}/A_m{M}_ones.i4lo.bin")
        C.astype(np.int16).tofile(f"{d}/C_m{M}_normal.i16.bin")
        gens.append({"gen": g, "tag_offset": TAG * (g + 1),
                     "golden": f"C[m][n] == n - {N // 2} + {TAG * (g + 1)}"})

    np.full((M, N), POISON, dtype=np.int16).tofile(f"{root}/POISON_fill.i16.bin")

    # ---- health probe -------------------------------------------------
    # NPU state crosses process boundaries: a failing shape leaves the device
    # degraded and the NEXT run of a different shape reads wrong, in both
    # directions. Fresh processes do not isolate it and repeat counts cannot
    # detect it, because every rep is equally contaminated. So every reading
    # has to be bracketed by a known-good shape.
    HK, HN = a.health_k, a.health_n
    assert HN % 192 == 0 and HN // 3 <= 64, (
        f"health N={HN}: needs N/3 <= 64 so there is exactly one 64-kernel block "
        f"per core and the gate cannot show the block-prefix fault")
    hbase = np.arange(HN) - HN // 2
    hd = os.path.join(root, "health")
    os.makedirs(hd, exist_ok=True)
    hb = enc(hbase, HK)                      # untagged: plain golden
    ha = np.ones((M, HK), dtype=np.int8)
    hc = ha.astype(np.int32) @ hb.astype(np.int32)
    assert (hc[0] == hbase).all()
    pack_nibbles(b_native(hb, 64, KTILE)).tofile(f"{hd}/B_nat_n64.i4lo.bin")
    pack_nibbles(ha).tofile(f"{hd}/A_m{M}_ones.i4lo.bin")
    hc.astype(np.int16).tofile(f"{hd}/C_m{M}_normal.i16.bin")

    json.dump({
        "purpose": "distinguish never-written / zero / stale / misaddressed",
        "why": "a repeat-count experiment cannot detect staleness, because a "
               "stale value from an identical previous run looks correct. "
               "Different goldens per run breaks that tie.",
        "K": K, "N": N, "M": M, "gens": a.gens, "encoding": a.encoding,
        "encoding_note": "sparse (prefix-of-ones, mostly zeros) provoked a "
                         "data-dependent fault on silicon that dense random "
                         "weights did not, at identical shape and identical "
                         "packed bytes. Run both and diff.",
        "tag_stride": int(TAG), "poison_int16": POISON,
        "health_probe": {
            "K": HK, "N": HN, "run_at": "HMULT=2 (ceil(512/768)=1 -> floor 2)",
            "golden": "C[m][n] == n - %d  (untagged)" % (HN // 2),
            "pass": "uniform blocks across all 3 cores",
            "why_small_N": "N/3 <= 64 means exactly ONE 64-kernel block per core, "
                           "so the gate cannot exhibit the block-prefix fault. At "
                           "N=1152 the health probe showed core2 "
                           "[32,32,32,32,0,0] on a HEALTHY device and so could "
                           "never pass - the gate was measuring the fault it was "
                           "meant to exclude.",
            "why": "NPU state crosses process boundaries. A failing shape "
                   "leaves the device degraded and the next run of a "
                   "different shape reads wrong, in both directions. Repeat "
                   "counts cannot detect this -- every rep is contaminated.",
            "rule": "measure ONLY from a proven-healthy device; if health "
                    "fails, settle and retry until it passes, then measure",
        },
        "procedure": [
            "run health/ until it passes 3/3 cores  <-- GATE, not optional",
            "fill the output buffer with POISON_fill.i16.bin",
            "dispatch gen0, capture C",
            "WITHOUT clearing, dispatch gen1, capture C",
            "repeat for each gen",
            "RE-DISPATCH gen0 LAST and capture it separately",
            "re-run health/ AFTER the measurement",
            "classify with classify_flaky.py, passing health_pre and health_post",
        ],
        "invalidation": "health_pre failing means the device was already "
                        "degraded and the reading is void. health_post "
                        "failing means the measured config degraded the "
                        "device, so this reading may stand but the NEXT one "
                        "is suspect until health passes again.",
        "note_cores": "failing cores observed as always 1 and 2, never 0 -- "
                      "structural, not a race. Record per-core.",
        "warmup_test": {
            "why": "a first dispatch that fails while every later one passes "
                   "reads as x/3 when each rep is a fresh process. That is "
                   "warm-up, not a race and not a capacity ceiling.",
            "how": "compare gen0-first against gen0-last, same config, same "
                   "process. If gen0-first is poison/garbage and gen0-last is "
                   "correct, warm-up is confirmed and is per-process or "
                   "per-buffer, not per-config.",
            "consequence": "re-measure every HMULT point with one warm-up "
                           "dispatch discarded. Entries previously called "
                           "flaky may be clean, and any single-reading "
                           "conclusion elsewhere in the int4 work may carry "
                           "the same artifact.",
            "still_real_if": "the config fails on a WARM process too "
                             "(K=3072 h5 was 0/3 on every rep, so it is real)",
        },
        "reading": {
            "poison": "hardware never wrote this lane",
            "zero": "covered but fetched nothing",
            "other generation": "STALE from that run -- buffer reuse hazard",
            "correct": "fine",
            "else": "misaddressed or garbage",
        },
        "also_worth_recording": "which CORE fails. If x/3 counts cores, a "
                               "failure that is always core N is structural; "
                               "one that moves between cores is a timing race.",
        "generations": gens,
    }, open(f"{root}/manifest.json", "w"), indent=2)
    print(f"{a.gens} generations, K={K} N={N} M={M}, "
          f"{a.gens * K * N // 2 / 2**20:.2f} MiB -> {root}")
    print(f"poison {POISON} (0xAAAA)")


if __name__ == "__main__":
    main()
