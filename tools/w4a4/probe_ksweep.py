"""K-sweep kernel-id probes -- bracket the K-dependent block-prefix fault.

Observed on silicon at N=1152: per-64-block correct counts are [32]*6 at
K=512 and K=1024, but [32,32,32,0,0,0] at K=2048. Half the blocks stop
producing output entirely. Reproduced with random data, so it is a
hardware/config fault, not a property of any particular bank.

NOT the entries rule. entries=16 at K=2048 was checked directly and is
correct -- every other value is worse. The "16 needs a 5th bit" idea is dead.
The K dependence comes from somewhere else, and the current lead is HMULT:
the H-axis multiplier that already needed to be 2 (dispatch M_hw = 2*M, read
rows 0,2,4,...) may need to scale with K.

These banks are HMULT-independent -- HMULT is a dispatch parameter, not a
property of the weight buffer. So one bank per K serves every HMULT, and the
sweep is K x HMULT over a fixed set of files.

What to look for: if the required HMULT doubles at the K where the fault
appears, the H axis is compensating for feature volume and the rule is
mechanical. If the fault moves with K but HMULT does not fix it, the H axis
is not the variable.

Every probe is self-describing: golden C[m][n] == n - N//2, so adding N//2 to
each output lane names the kernel that actually landed there.

    python probe_ksweep.py --out ksweep --n 1152
"""

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import pack_nibbles, b_native, a_native, c_native
from pack_rk3588 import encode_sum, encode_sum_dense, KTILE

# 128-granular around the observed break, plus anchors that separate
# the candidate HMULT rules (K=1536 and K=2560 disagree between them)
DEFAULT_K = [512, 1024, 1536, 1664, 1792, 1920, 2048, 2176, 2304, 2560, 3072, 4096]


def emit(root, K, N, ms=2, enc=None):
    d = os.path.join(root, f"probe.kernelid_K{K}_N{N}")
    os.makedirs(d, exist_ok=True)
    tgt = np.arange(N) - N // 2
    Bq = (enc or encode_sum)(tgt, K)              # (K, N)
    A = np.ones((ms, K), dtype=np.int8)
    C = A.astype(np.int32) @ Bq.astype(np.int32)
    assert (C[0] == tgt).all() and (C[1] == C[0]).all()

    files = {}

    def w(nm, arr):
        arr.tofile(os.path.join(d, nm))
        files[nm] = int(arr.nbytes)

    # verified triple
    w("B_nat_n64.i4lo.bin", pack_nibbles(b_native(Bq, 64, KTILE)))
    w("A_m2_ones.i4lo.bin", pack_nibbles(A))
    w("C_m2_normal.i16.bin", C.astype(np.int16))
    # alternates
    w("B_normal.i4lo.bin", pack_nibbles(Bq))
    if N % 32 == 0:
        w("B_nat_n32.i4lo.bin", pack_nibbles(b_native(Bq, 32, KTILE)))
    w("A_m2_ones_nat_k32.i4lo.bin", pack_nibbles(a_native(A, KTILE)))
    w("C_m2_normal.i32.bin", C)
    w("C_m2_nat_n8.i16.bin", c_native(C, 8).astype(np.int16))

    json.dump({
        "name": f"kernelid_K{K}_N{N}", "K": K, "N": N, "M": ms,
        "invariant": "C[m][n] == n - N//2 for all m",
        "read_back": "kernel_index = C[m][n] + N//2",
        "entries_K_over_128": K / 128.0,
        "n_blocks_of_64": N // 64,
        "b_tiles_n64": [N // 64, K // KTILE],
        "total_B_bytes": K * N // 2,
        "files": files,
    }, open(os.path.join(d, "meta.json"), "w"), indent=2)
    return K, K // 128, K // 128 <= 15, K * N // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ksweep")
    ap.add_argument("--n", type=int, default=1152,
                    help="kernel count; 1152 matches the observed failure")
    ap.add_argument("--encoding", default="dense",
                    choices=["dense", "sparse"],
                    help="sparse is prefix-of-ones and mostly zeros; it exposed\n a data-dependent fault. dense is representative. Run both.")
    ap.add_argument("--k", default="", help="comma list of K; default sweeps "
                    "128-granular around the break plus HMULT-rule anchors")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    root = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
    os.makedirs(root, exist_ok=True)

    ks = [int(x) for x in a.k.split(",") if x] or DEFAULT_K
    rows, total = [], 0
    for K in ks:
        assert K % KTILE == 0, f"K={K} must be a multiple of {KTILE}"
        enc = encode_sum_dense if a.encoding == 'dense' else encode_sum
        k, e, fits, nb = emit(root, K, a.n, enc=enc)
        rows.append({"K": k, "entries": e})
        total += nb
        print(f"K={k:5d}  entries={e:3d}  blocks={a.n // 64:3d}  "
              f"bank={nb / 2**20:5.2f} MiB")

    json.dump({
        "purpose": "bracket the K-dependent block-prefix fault",
        "observed": {"K512": "[32]*6", "K1024": "[32]*6",
                     "K2048": "[32,32,32,0,0,0]"},
        "ruled_out": "entries rule. entries=16 at K=2048 is confirmed "
                     "correct; every other value is worse.",
        "current_lead": "HMULT (H-axis multiplier) may need to scale with K. "
                        "These banks are HMULT-independent, so sweep "
                        "K x HMULT over the same files.",
        "sweep_advice": "for each K, find the smallest HMULT giving [32]*"
                        "(N/64) with no zero blocks; tabulate HMULT vs K",
        "confirmed": {"K2048_HMULT4": "fixes it, reproduced on random data"},
        "candidate_rules": {
            "A": "HMULT = max(2, K/512)",
            "B": "HMULT = 2*ceil(K/1024)",
            "both_fit": [[512, 2], [1024, 2], [2048, 4]],
            "discriminating_K": {"1536": {"A": 3, "B": 4},
                                 "2560": {"A": 5, "B": 6}},
            "note": "if only powers of two are accepted, A is wrong at 1536 "
                    "and the H axis is quantised, which is itself the answer"},
        "encoding": a.encoding,
        "N": a.n, "layout": "B (N/64,K/32,64,32) k-inner; A normal (M,K); "
                            "C int16 (M,N)",
        "shapes": rows,
    }, open(os.path.join(root, "manifest.json"), "w"), indent=2)
    print(f"\n{len(ks)} probes, {total / 2**20:.1f} MiB -> {root}")


if __name__ == "__main__":
    main()
