"""THE THIRD AXIS: N, and whether the K=2048 ridge survives a square-ish layer.

m_curve_probe.py found, at N=9216 fixed:
  - the governing law is the CBUF feature budget, K * M <= 327,680 (10 banks x 32KB), which
    predicts the M_max cliff exactly (K=3072 -> 106; 96 passes, 112 fails)
  - the single-dispatch optimum is K=2048 M=144 at 77.2% of peak
  - at an EQUAL budget, M beats K: (2048,144) 77.2% vs (3072,96) 54.9%
  - M=128 is anomalously bad at every K; M=144 recovers fully

But N was pinned at 9216 throughout, and a scaffold layer is [M x K] @ [K x N] with N comparable
to K, not 4x it. The ridge could move. N does not enter the feature budget (that is M*K), but it
sets weight bytes (K*N) and output bytes (M*N*4), so it changes the traffic mix, not the cap.

Phase 1 sweeps N at the current optimum. Phase 2 re-tests K x M at the best N, because a
recommendation taken from one slice through a 3-D space is exactly the error this probe exists to
correct - the K=3072 rule came from such a slice and did not survive contact with the second axis.

Gated: NPU state crosses process boundaries.
"""
import ctypes as C, numpy as np, os, sys, subprocess

lib = C.CDLL(os.environ.get("NPU_MM_SO", "./build/npu_mm.so"))
lib.npu_mm_init.restype = C.c_int
lib.npu_mm_prep_int8.argtypes = [C.POINTER(C.c_int8), C.c_int, C.c_int]
lib.npu_mm_prep_int8.restype = C.c_int
lib.npu_mm_run_int8_raw.argtypes = [C.POINTER(C.c_int8), C.c_int, C.POINTER(C.c_int32), C.c_int]
lib.npu_mm_run_int8_raw.restype = C.c_int
lib.npu_mm_free_int8.argtypes = [C.c_int]
lib.npu_last_hw_ns.restype = C.c_longlong
p8 = lambda x: x.ctypes.data_as(C.POINTER(C.c_int8))
p32 = lambda x: x.ctypes.data_as(C.POINTER(C.c_int32))
assert lib.npu_mm_init() == 0

PEAK = 2.67e12
BUDGET = 327680
REPS = int(os.environ.get('REPS', 5))
rng = np.random.default_rng(2)


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


def run(K, N, M):
    """Returns (hw_us, pct_peak) or None. M is padded to a multiple of 4."""
    if K * M > BUDGET:
        return None
    Mp = ((M + 3)//4)*4
    B = rng.integers(-127, 127, (N, K), dtype=np.int8)
    h = lib.npu_mm_prep_int8(p8(B), N, K)
    if h < 0:
        return None
    out = np.zeros(3 * Mp * (((N + 47)//48)*48//3), np.int32)
    A = rng.integers(-127, 127, (Mp, K), dtype=np.int8)
    t = None
    for _ in range(REPS):
        if lib.npu_mm_run_int8_raw(p8(A), h, p32(out), Mp) != 0:
            t = None
            break
        v = lib.npu_last_hw_ns() / 1e3
        if v > 0 and (t is None or v < t):
            t = v
    lib.npu_mm_free_int8(h)
    if t is None:
        return None
    return t, Mp * K * N / (t * 1e-6) / PEAK * 100


if not gate():
    sys.exit("ABORT: device degraded before measuring")
print("gate: CLEAN\n")

# ---- phase 1: N at the current optimum -------------------------------------------------
K0, M0 = 2048, 144
NS = [int(x) for x in os.environ.get('NS', '384,768,1536,2304,3072,4608,6144,7680,9216').split(',')]
print(f"PHASE 1 — N sweep at K={K0} M={M0}")
print(f"  {'N':>6} {'hw us':>8} {'GMAC':>8} {'%peak':>7} {'wt MB':>7} {'out MB':>7}")
bestN, bestP = None, 0.0
for N in NS:
    r = run(K0, N, M0)
    if r is None:
        print(f"  {N:>6} {'rc!=0':>8}")
        continue
    t, pct = r
    print(f"  {N:>6} {t:>8.0f} {M0*K0*N/1e9:>8.2f} {pct:>6.1f}% {K0*N/1e6:>7.1f} {M0*N*4/1e6:>7.1f}")
    if pct > bestP:
        bestP, bestN = pct, N

print(f"\n  best N = {bestN} at {bestP:.1f}%\n")

# ---- phase 2: K x M at the best N ------------------------------------------------------
print(f"PHASE 2 — K x M at N={bestN}")
print(f"  {'K':>6} {'M':>5} {'hw us':>8} {'%peak':>7} {'K*M':>8}")
grid = []
for K in (1024, 1536, 2048, 2560, 3072):
    for M in (64, 96, 112, 144, 160):
        if K * M > BUDGET:
            continue
        r = run(K, bestN, M)
        if r is None:
            print(f"  {K:>6} {M:>5} {'rc!=0':>8}")
            continue
        t, pct = r
        print(f"  {K:>6} {M:>5} {t:>8.0f} {pct:>6.1f}% {K*M:>8}")
        grid.append((pct, K, M, t))

grid.sort(reverse=True)
print(f"\n  TOP 5 at N={bestN}:")
for pct, K, M, t in grid[:5]:
    print(f"    K={K:<5} M={M:<4} {pct:.1f}% of peak, {t:.0f}us, K*M={K*M} "
          f"({100*K*M/BUDGET:.0f}% of feature budget)")
