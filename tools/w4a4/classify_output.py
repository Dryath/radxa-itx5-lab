"""Board-side: read what the NPU actually wrote and name the layout it used.

    python3 classify_output.py w4a4_probe/probe.kernelid_K64_N192 hw_out.bin

The kernel-id probe is built so the correct answer is C[m][n] == n - N//2.
Add N//2 back to every lane and you get the kernel index the hardware really
placed there. The shape of that permutation names the bug:

    0,1,2,3,...            correct
    0..31,0..31,...        N tiled at 32, hardware expects 64  <- known fault
    0,2,4,...              nibble order swapped inside the tile
    0..63 then garbage     only the first 1024-byte tile landed
"""
import json, os, sys
import numpy as np


def load_c(path, N, M=2):
    raw = np.fromfile(path, dtype=np.uint8)
    # exact fits first -- a partial int16 view of an int32 buffer also "fits",
    # and picking it produces a convincing but entirely fictional permutation
    for rows in (M, 1):
        for dt in (np.int16, np.int32):
            if raw.nbytes == rows * N * np.dtype(dt).itemsize:
                return raw.view(dt).reshape(rows, N).astype(np.int64), np.dtype(dt).name
    for dt in (np.int16, np.int32):
        n = raw.nbytes // np.dtype(dt).itemsize
        if n >= N:
            return (raw.view(dt)[:N].reshape(1, N).astype(np.int64),
                    np.dtype(dt).name + " (partial)")
    raise SystemExit(f"{raw.nbytes} bytes does not fit M={M} N={N}")


def undo_c_native(C, ntile):
    """(N/ntile, M, ntile) -> (M, N).  Pass the flat buffer already reshaped."""
    nb, M = C.shape[1] // ntile, C.shape[0]
    return C.reshape(-1)[:].reshape(nb, M, ntile).transpose(1, 0, 2).reshape(M, -1)


def diagnose(idx, N):
    d = np.diff(idx)
    if (idx == np.arange(N)).all():
        return "CORRECT -- lanes map 1:1 to kernels"
    for blk in (64, 32, 16):
        if N % blk == 0:
            r = idx.reshape(-1, blk)
            if (r == r[0]).all() and (r[0] == np.arange(blk)).all():
                return (f"N TILED AT {blk}: every {blk}-lane group repeats kernels "
                        f"0..{blk - 1}. Bank supplies {blk} kernels where the "
                        f"hardware atom wants 64.")
    if len(d) and (d[:len(d) // 2] == 2).all():
        return "STRIDE 2 -- nibble order swapped, or int8 element width assumed"
    valid = (idx >= 0) & (idx < N)
    if valid[:64].all() and not valid[64:].any():
        return "ONLY FIRST TILE -- one 1024-byte weight fetch landed, rest is stale"
    good = int((idx == np.arange(N)).sum())
    return f"UNRECOGNISED -- {good}/{N} lanes correct; permutation dumped below"


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    pdir, hw = sys.argv[1], sys.argv[2]
    meta = json.load(open(os.path.join(pdir, "meta.json")))
    N, K, M = meta["N"], meta["K"], meta["M"]
    print(f"probe {meta['name']}  K={K} N={N} M={M}")
    print(f"expect {meta['invariant']}\n")

    C, dt = load_c(hw, N, M)
    print(f"read {hw} as {dt}  shape {C.shape}")

    best = None
    for label, arr in (("normal (M,N)", C),
                       ("native (N/8,M,8)", undo_c_native(C, 8) if C.size % (8 * M) == 0 else None),
                       ("native (N/4,M,4)", undo_c_native(C, 4) if C.size % (4 * M) == 0 else None)):
        if arr is None or arr.shape[1] != N:
            continue
        idx = arr[0] + N // 2
        score = int((idx == np.arange(N)).sum())
        print(f"  as {label:20s} -> {score}/{N} lanes correct")
        if best is None or score > best[0]:
            best = (score, label, idx)

    score, label, idx = best
    print(f"\nbest interpretation: {label}")
    print(f"diagnosis: {diagnose(idx, N)}")
    if score != N:
        print(f"\nfirst 80 recovered kernel indices:\n{idx[:80].tolist()}")
        if C.shape[0] > 1:
            print(f"row0 == row1: {(C[0] == C[1]).all()}  "
                  f"(must be True; A is all-ones on both rows)")


if __name__ == "__main__":
    main()
