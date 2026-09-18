"""Self-checks for the layout primitives. Run before trusting any bank."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (pack_nibbles, unpack_nibbles, a_native, b_native, c_native,
                    randomized_hadamard, quant_int4_groupwise,
                    dequant_int4_groupwise)
from pack_rk3588 import encode_sum

rng = np.random.default_rng(7)

# nibble roundtrip, both orders
for hi in (False, True):
    x = rng.integers(-8, 8, 4096).astype(np.int8)
    assert (unpack_nibbles(pack_nibbles(x, hi), hi) == x).all(), f"nibble hi={hi}"
b = pack_nibbles(np.array([1, 2], np.int8))
assert b[0] == 0x21, hex(b[0])          # low nibble holds element 0
b = pack_nibbles(np.array([1, 2], np.int8), hi_first=True)
assert b[0] == 0x12, hex(b[0])
print("nibble pack/unpack           ok")

# B native ordering must match the header text literally:
#   [K1N1, K2N1, ..., K32N1, K1N2, ...]  -> k fastest, then n, then kblock, then nblock
K, N, nt, kt = 64, 128, 64, 32
B = np.arange(K * N, dtype=np.int32).reshape(K, N)
nat = b_native(B, ntile=nt, ktile=kt)
assert nat.shape == (N // nt, K // kt, nt, kt)
flat = nat.reshape(-1)
assert flat[0] == B[0, 0] and flat[1] == B[1, 0] and flat[31] == B[31, 0]
assert flat[32] == B[0, 1]                       # next kernel, k restarts
assert flat[nt * kt] == B[32, 0]                 # next k tile, kernel restarts
assert flat[nt * kt * (K // kt)] == B[0, 64]     # next 64-kernel block
print("b_native (N/64,K/32,64,32)   ok")

for ntile, elem_bytes in ((64, 0.5), (32, 1.0), (16, 2.0)):
    assert ntile * 32 * elem_bytes == 1024
print("1024-byte tile invariant     ok")

A = np.arange(4 * 64, dtype=np.int32).reshape(4, 64)
an = a_native(A, 32)
assert an.shape == (2, 4, 32) and an[0, 0, 1] == A[0, 1] and an[1, 0, 0] == A[0, 32]
C = np.arange(4 * 32, dtype=np.int32).reshape(4, 32)
cn = c_native(C, 8)
assert cn.shape == (4, 4, 8) and cn[0, 0, 1] == C[0, 1] and cn[1, 0, 0] == C[0, 8]
print("a_native / c_native          ok")

# CRITICAL: with a single K tile the N-tiling is unobservable.
# (N/64,K/32,64,32) and (N/32,K/32,32,32) serialise to identical bytes when
# K==32, because there is no second k-block to interleave the n-blocks with.
# Every probe run at K=32 is structurally blind to the 64-vs-32 kernel bug.
pat = lambda K, N: np.tile(np.arange(N, dtype=np.int8).reshape(1, N) % 15 - 7, (K, 1))
b = pat(32, 128)
assert (b_native(b, 64, 32).reshape(-1) == b_native(b, 32, 32).reshape(-1)).all(), \
    "K=32 must alias -- if not, the control is wrong"
print("K=32  n64 == n32 (blind)     ok  <- probes need K>=64")
b = pat(64, 128)
n64, n32 = b_native(b, 64, 32).reshape(-1), b_native(b, 32, 32).reshape(-1)
assert not (n64 == n32).all()
first = (n64[:64 * 32] == n32[:64 * 32]).all()
print(f"K=64  n64 != n32             ok  (first tile aliases: {first})")

# kernel-id probe invariant
for Kp, Np in ((64, 192), (64, 384), (128, 192)):
    Bp = encode_sum(np.arange(Np) - Np // 2, Kp)
    C = np.ones((2, Kp), np.int32) @ Bp.astype(np.int32)
    assert (C[0] == np.arange(Np) - Np // 2).all()
    assert (C[1] == C[0]).all()
print("kernel-id probe invariant    ok")

Q = randomized_hadamard(2048, 1337)
assert np.abs(Q.T @ Q - np.eye(2048)).max() < 1e-10
print("hadamard orthonormal         ok")

W = rng.normal(size=(256, 512))
q, s = quant_int4_groupwise(W, 32, axis=1)
assert q.min() >= -8 and q.max() <= 7 and s.shape == (256, 16)
r = np.abs(dequant_int4_groupwise(q, s, 32, 1) - W).max() / np.abs(W).max()
print(f"int4 g32 rtn max rel err     {r:.4f}")
print("\nall layout checks passed")
