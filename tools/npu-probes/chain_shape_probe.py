"""Does PC chaining move the ridge, and does it make block-diagonal fusion unnecessary?

n_curve_probe.py fixed the single-dispatch optimum at K=2048 M=144 N>=4608 (75.9% of peak). Every
number there was ONE dispatch. PC chaining amortises the ~27us submit across up to 32 tasks, so
before the shape is locked we need to know whether the ridge survives it.

Two questions, one probe:

1. DOES THE RIDGE HOLD? If dispatch is already a small fraction at these sizes, chaining should
   change little and K=2048/M=144 stays the answer. If it changes the ordering, the shape spec is
   wrong for any multi-layer net.

2. DOES CHAINING KILL THE CASE FOR FUSION? The earlier argument for one fused block-diagonal
   matmul was dispatch: three separate blocks cost 3 x 27us, fusion costs 1 x 27us, and fusion won
   ~3x DESPITE doing 2.8x the MACs with 64% of them multiplying by zero. If chaining collapses
   that dispatch gap, separate blocks win on every other axis at once - no wasted MACs, and no
   streaming of zeros at M=1 (which is ~64% of decode traffic in the fused layout).
   That would be an architecture-level result, not a tuning one.

CAVEAT, stated up front: `npu_mm_run_int8_tiles` reads N and K from hs[0], so all tiles share a
shape. This measures HOMOGENEOUS chained layers. A real graft has heterogeneous blocks, and a real
sequential net needs layer i+1 to consume layer i's output - which a PC chain of independent
descriptors does not obviously provide. Both are out of scope here and neither is assumed.

Gated: NPU state crosses process boundaries.
"""
import ctypes as C, numpy as np, os, sys, subprocess, time

lib = C.CDLL(os.environ.get("NPU_MM_SO", "./build/npu_mm.so"))
lib.npu_mm_init.restype = C.c_int
lib.npu_mm_prep_int8.argtypes = [C.POINTER(C.c_int8), C.c_int, C.c_int]
lib.npu_mm_prep_int8.restype = C.c_int
lib.npu_mm_run_int8_tiles.argtypes = [C.POINTER(C.c_int8), C.POINTER(C.c_int), C.c_int,
                                      C.POINTER(C.c_int32), C.c_int, C.c_int]
lib.npu_mm_run_int8_tiles.restype = C.c_int
lib.npu_mm_free_int8.argtypes = [C.c_int]
p8 = lambda x: x.ctypes.data_as(C.POINTER(C.c_int8))
p32 = lambda x: x.ctypes.data_as(C.POINTER(C.c_int32))
assert lib.npu_mm_init() == 0

PEAK = 2.67e12
BUDGET = 327680
N = int(os.environ.get('N', 4608))
ITERS = int(os.environ.get('ITERS', 12))
rng = np.random.default_rng(3)


def gate():
    env = dict(os.environ, REPS="1", HMULT="2", VK="512", VN="1152", BLK="64", CG="32", KIN="1",
               EX="dpu_i8")
    for _ in range(6):
        o = subprocess.run([os.environ.get("PYTHON", "python3"),
                            os.environ.get("NPU_VERIFY", "./int4_verify.py")],
                           env=env, capture_output=True, text=True).stdout
        if o.count("per-64-block correct: [32, 32, 32, 32, 32, 32]") == 3:
            return True
    return False


def bench(K, M, T, chained):
    """Wall-clock per submit-group. hw_elapse_time only covers the last task of a chain, so the
    chain must be timed end to end from the host."""
    Mp = ((M + 3)//4)*4
    N_slice = (((N + 47)//48)*48)//3
    hs = []
    for _ in range(T):
        B = rng.integers(-127, 127, (N, K), dtype=np.int8)
        h = lib.npu_mm_prep_int8(p8(B), N, K)
        if h < 0:
            for x in hs: lib.npu_mm_free_int8(x)
            return None
        hs.append(h)
    ph = (C.c_int * T)(*hs)
    A = rng.integers(-127, 127, (Mp, K), dtype=np.int8)
    out = np.zeros(3 * Mp * N_slice, np.int32)
    if lib.npu_mm_run_int8_tiles(p8(A), ph, T, p32(out), Mp, chained) != 0:
        for x in hs: lib.npu_mm_free_int8(x)
        return None
    best = None
    for _ in range(ITERS):
        t0 = time.perf_counter()
        rc = lib.npu_mm_run_int8_tiles(p8(A), ph, T, p32(out), Mp, chained)
        dt = (time.perf_counter() - t0) * 1e6
        if rc != 0:
            best = None
            break
        if best is None or dt < best:
            best = dt
    for x in hs: lib.npu_mm_free_int8(x)
    return best


if not gate():
    sys.exit("ABORT: device degraded before measuring")
print(f"gate: CLEAN   N={N}\n")

SHAPES = [(2048, 144), (2048, 112), (3072, 96), (2560, 96), (1536, 160), (1024, 144)]
TS = [1, 2, 4, 8]

print(f"  {'K':>5} {'M':>4} {'T':>3} {'sep us':>9} {'chain us':>9} {'speedup':>8} "
      f"{'us/layer':>9} {'%peak':>7}")
rows = []
for K, M in SHAPES:
    if K * M > BUDGET:
        continue
    Mp = ((M + 3)//4)*4
    for T in TS:
        sep = bench(K, M, T, 0)
        ch = bench(K, M, T, 1)
        if sep is None or ch is None:
            print(f"  {K:>5} {M:>4} {T:>3} {'rc!=0':>9}")
            continue
        mac = T * Mp * K * N
        pct = mac / (ch * 1e-6) / PEAK * 100
        print(f"  {K:>5} {M:>4} {T:>3} {sep:>9.0f} {ch:>9.0f} {sep/ch:>7.2f}x "
              f"{ch/T:>9.0f} {pct:>6.1f}%")
        if T == max(TS):
            rows.append((pct, K, M, sep/ch))
    print()

rows.sort(reverse=True)
print(f"  RIDGE AT T={max(TS)} CHAINED:")
for pct, K, M, sp in rows:
    print(f"    K={K:<5} M={M:<4} {pct:5.1f}% of peak   (chain speedup {sp:.2f}x)")
print("\nIf the ordering here matches the single-dispatch table, the ridge survives chaining and")
print("K=2048/M=144 locks. If chain speedup is large, fusion-with-zeros loses its justification.")
