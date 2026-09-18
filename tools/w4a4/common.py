"""Shared primitives for the RK3588 W4A4 build.

Conventions
-----------
HF weights are [out_features, in_features] and y = x @ W.T.
RKNN matmul is A(M,K) @ B(K,N) = C(M,N), so B = W.T, and
    K = in_features, N = out_features = "kernel count".

int4 values are signed, range [-8, 7], stored two-per-byte.
Nibble order is a *hypothesis* (vendor never shipped a working int4 path),
so every packer takes nibble_hi_first and we ship both.
"""

import numpy as np

# ---------------------------------------------------------------- Hadamard


def sylvester(n: int) -> np.ndarray:
    """Orthonormal Sylvester-Hadamard of size n (n must be a power of two)."""
    assert n & (n - 1) == 0, f"{n} is not a power of two"
    H = np.ones((1, 1), dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def randomized_hadamard(n: int, seed: int) -> np.ndarray:
    """Q = H_n @ diag(+-1).  Orthonormal; Q.T @ Q == I."""
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n)
    return sylvester(n) * signs[None, :]


def block_diag_repeat(B: np.ndarray, reps: int) -> np.ndarray:
    """blockdiag(B, B, ... reps times) without materialising zeros twice."""
    b = B.shape[0]
    out = np.zeros((b * reps, b * reps), dtype=B.dtype)
    for i in range(reps):
        out[i * b:(i + 1) * b, i * b:(i + 1) * b] = B
    return out


# ------------------------------------------------------------ quantisation


def quant_int4_groupwise(W: np.ndarray, group: int = 32, axis: int = 1):
    """Symmetric int4 RTN along `axis` in groups of `group`.

    W: float array. Returns (q int8 in [-8,7], scale float32).
    scale has the same shape as W with `axis` collapsed to n_groups.
    """
    assert W.shape[axis] % group == 0, f"axis {axis} len {W.shape[axis]} % {group}"
    Wm = np.moveaxis(W, axis, -1)
    shp = Wm.shape
    Wg = Wm.reshape(*shp[:-1], shp[-1] // group, group)
    amax = np.abs(Wg).max(axis=-1, keepdims=True)
    scale = np.maximum(amax / 7.0, 1e-8)
    q = np.rint(Wg / scale).clip(-8, 7).astype(np.int8)
    q = np.moveaxis(q.reshape(shp), -1, axis)
    scale = np.moveaxis(scale.squeeze(-1), -1, axis).astype(np.float32)
    return q, scale


def dequant_int4_groupwise(q: np.ndarray, scale: np.ndarray, group: int = 32, axis: int = 1):
    qm = np.moveaxis(q.astype(np.float32), axis, -1)
    sm = np.moveaxis(scale, axis, -1)
    shp = qm.shape
    out = qm.reshape(*shp[:-1], shp[-1] // group, group) * sm[..., None]
    return np.moveaxis(out.reshape(shp), -1, axis)


# --------------------------------------------------------------- nibbles


def pack_nibbles(a: np.ndarray, hi_first: bool = False) -> np.ndarray:
    """Flatten `a` (values in [-8,7]) and pack pairs into uint8.

    hi_first=False -> element 2i lands in bits [3:0] (low nibble first).
    """
    flat = np.asarray(a).reshape(-1).astype(np.int16) & 0x0F
    assert flat.size % 2 == 0, "odd element count cannot be nibble-packed"
    lo, hi = flat[0::2], flat[1::2]
    if hi_first:
        lo, hi = hi, lo
    return ((hi << 4) | lo).astype(np.uint8)


def unpack_nibbles(b: np.ndarray, hi_first: bool = False) -> np.ndarray:
    b = np.asarray(b, dtype=np.uint8)
    lo = b & 0x0F
    hi = (b >> 4) & 0x0F
    if hi_first:
        lo, hi = hi, lo
    out = np.empty(b.size * 2, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return np.where(out > 7, out - 16, out).astype(np.int8)


# ------------------------------------------------------- native layouts
#
# Every layout below is a *candidate*. The vendor header documents the
# `b_native_n64` / `a_native_k32` / `c_native_n8` triple for RK3588 int4,
# but librknnrt rejects all int4 matmul types, so nothing here has ever
# been validated against silicon. Emit them all; let the hardware vote.


def a_native(A: np.ndarray, ktile: int = 32) -> np.ndarray:
    """A(M,K) -> (K/ktile, M, ktile).  int4 ktile=32, int8 16, fp16 8."""
    M, K = A.shape
    assert K % ktile == 0
    return A.reshape(M, K // ktile, ktile).transpose(1, 0, 2).copy()


def b_native(B: np.ndarray, ntile: int, ktile: int = 32, k_major: bool = False) -> np.ndarray:
    """B(K,N) -> (N/ntile, K/ktile, ntile, ktile).

    ntile=64 is the header's int4 value, 32 the int8 value, 16 the fp16 value.
    k_major=True emits (..., ktile, ntile) instead -- inner pair swapped.
    """
    K, N = B.shape
    assert K % ktile == 0, f"K={K} % {ktile}"
    assert N % ntile == 0, f"N={N} % {ntile}"
    # (K/kt, kt, N/nt, nt)
    t = B.reshape(K // ktile, ktile, N // ntile, ntile)
    if k_major:
        return t.transpose(2, 0, 1, 3).copy()   # (N/nt, K/kt, kt, nt)
    return t.transpose(2, 0, 3, 1).copy()       # (N/nt, K/kt, nt, kt)


def c_native(C: np.ndarray, ntile: int) -> np.ndarray:
    """C(M,N) -> (N/ntile, M, ntile).  int4->int16 ntile=8, int8->int32 ntile=4."""
    M, N = C.shape
    assert N % ntile == 0
    return C.reshape(M, N // ntile, ntile).transpose(1, 0, 2).copy()


def pad_to(W: np.ndarray, axis: int, mult: int, value=0):
    """Zero-pad `axis` up to a multiple of `mult`. Returns (padded, orig_len)."""
    n = W.shape[axis]
    tgt = ((n + mult - 1) // mult) * mult
    if tgt == n:
        return W, n
    pad = [(0, 0)] * W.ndim
    pad[axis] = (0, tgt - n)
    return np.pad(W, pad, constant_values=value), n


def lcm(a: int, b: int) -> int:
    from math import gcd
    return a * b // gcd(a, b)
