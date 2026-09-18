"""Integrity check on an emitted probe kit.

Reads the .bin files back off disk exactly as the board would, undoes each
layout, and re-derives C. If this passes, any mismatch the board sees is the
hardware's, not the kit's.

    python verify_kit.py w4a4_probe
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import unpack_nibbles

root = sys.argv[1] if len(sys.argv) > 1 else "w4a4_probe"
root = root if os.path.isabs(root) else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), root)
man = json.load(open(os.path.join(root, "manifest.json")))
fails = 0


def rd4(p, hi=False):
    return unpack_nibbles(np.fromfile(p, dtype=np.uint8), hi)


def check(cond, msg):
    global fails
    if not cond:
        fails += 1
        print(f"    FAIL {msg}")
    return cond


for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if not os.path.isdir(p):
        continue
    m = json.load(open(os.path.join(p, "meta.json")))
    K, N = m["K"], m["N"]
    Bn = rd4(f"{p}/B_normal.i4lo.bin").reshape(K, N)
    print(f"{d}  K={K} N={N}")

    # every native layout must be a permutation of the same logical bank
    for nt in (64, 32, 16):
        f = f"{p}/B_nat_n{nt}.i4lo.bin"
        if not os.path.exists(f):
            continue
        nat = rd4(f).reshape(N // nt, K // 32, nt, 32)
        back = nat.transpose(1, 3, 0, 2).reshape(K, N)
        check((back == Bn).all(), f"B_nat_n{nt} does not invert to B_normal")

    f = f"{p}/B_nat_n64.i4hi.bin"
    if os.path.exists(f):
        nat = rd4(f, hi=True).reshape(N // 64, K // 32, 64, 32)
        check((nat.transpose(1, 3, 0, 2).reshape(K, N) == Bn).all(),
              "i4hi does not invert")

    f = f"{p}/B_nat_n64_kmaj.i4lo.bin"
    if os.path.exists(f):
        nat = rd4(f).reshape(N // 64, K // 32, 32, 64)
        check((nat.transpose(1, 2, 0, 3).reshape(K, N) == Bn).all(),
              "kmaj does not invert")

    # goldens must equal the integer product of the on-disk A and B
    for Mv in m.get("M_variants", [m.get("M", 2)]):
        af = (f"{p}/A_m{Mv}.i4lo.bin" if os.path.exists(f"{p}/A_m{Mv}.i4lo.bin")
              else f"{p}/A_m{Mv}_ones.i4lo.bin")
        if not os.path.exists(af):
            continue
        A = rd4(af).reshape(Mv, K)
        C = A.astype(np.int32) @ Bn.astype(np.int32)
        g = np.fromfile(f"{p}/C_m{Mv}_normal.i32.bin", dtype=np.int32).reshape(Mv, N)
        check((C == g).all(), f"C_m{Mv}_normal != A@B")

        an = rd4(f"{p}/A_m{Mv}_nat_k32.i4lo.bin"
                 if os.path.exists(f"{p}/A_m{Mv}_nat_k32.i4lo.bin")
                 else f"{p}/A_m{Mv}_ones_nat_k32.i4lo.bin").reshape(K // 32, Mv, 32)
        check((an.transpose(1, 0, 2).reshape(Mv, K) == A).all(),
              f"A_m{Mv} native does not invert")

        cn = np.fromfile(f"{p}/C_m{Mv}_nat_n8.i16.bin",
                         dtype=np.int16).reshape(N // 8, Mv, 8)
        check((cn.transpose(1, 0, 2).reshape(Mv, N) == C).all(),
              f"C_m{Mv} native n8 mismatch (or int16 overflow)")
        # int4 output is documented as int16. Accumulating K terms of up to
        # 8*8=64 can in principle reach 64*K, which for K=6144 is 393216 --
        # 12x past int16. Report real headroom; it bounds how far the kernel
        # can go before it has to split K.
        mx = int(np.abs(C).max())
        theo = 64 * K
        print(f"    M={Mv} max|C|={mx:6d}  int16 headroom {32767 / max(mx,1):5.1f}x"
              f"  worst-case {theo} ({'OVERFLOWS' if theo > 32767 else 'safe'})")
        check(mx <= 32767, f"C_m{Mv} exceeds int16: {mx}")

    if "invariant" in m:
        Mv = m["M"]
        C = np.fromfile(f"{p}/C_m{Mv}_normal.i32.bin",
                        dtype=np.int32).reshape(Mv, N)
        check((C[0] == np.arange(N) - N // 2).all(), "kernel-id invariant broken")
        check((C == C[0]).all(), "kernel-id rows differ")

print(f"\n{'ALL KIT CHECKS PASSED' if not fails else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
