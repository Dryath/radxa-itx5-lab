"""THE ARITHMETIC-INTENSITY CURVE: how many rows of work must share a weight bank?

Two measured constants set the whole design:
  compute   2.67 GMAC/ms   (the shape-lock notes (docs/03), K=3072 N=9216 M=160: 4.53 GMAC / 1695us)
  bandwidth 23.8 GB/s      (bench_bandwidth.c, DMC maxed)

int8 is one byte per weight and one MAC per weight per row, so the intensity this chip demands is
    2.67e12 / 23.8e9 = 112 MACs per weight byte  ->  M >= 112
Below that the weight stream cannot be hidden and the MAC array starves; above it we are
compute-bound. M is capped at 160, so the usable window is predicted to be M in [112, 160].

That prediction already has one confirmation: at K=3072 N=9216 M=160, compute predicts 1696us and
the board measured 1695us, with 1189us of weight traffic hidden underneath. This probe measures
the whole curve instead of one point, because the SHAPE of the knee is what tells us how much a
custom operator has to batch to be worth building.

Why it matters more than the K curve: a standard autoregressive step is M=1. If M=1 sits near 1%
of peak, then ANY architecture whose per-step work is a single vector wastes ~99% of this silicon,
and the design job becomes "find 112-160 rows that share a weight bank per step" - concurrent
modality streams, parallel state slots, batched memory probes. That is an architecture question
this number decides.

Submit-only timing via the driver's hw_elapse_time: we want the silicon's number, not our readback.
"""
import ctypes as C, numpy as np, os, sys

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

def gate():
    """NPU state crosses process boundaries. Never measure on an ungated device - the first
    version of this probe skipped it, threw a submit timeout, and produced numbers I could not
    stand behind."""
    import subprocess
    env = dict(os.environ, REPS="1", HMULT="2", VK="512", VN="1152", BLK="64", CG="32", KIN="1",
               EX="dpu_i8")
    for _ in range(6):
        o = subprocess.run([os.environ.get("PYTHON", "python3"),
                            os.environ.get("NPU_VERIFY", "./int4_verify.py")],
                           env=env, capture_output=True, text=True).stdout
        if o.count("per-64-block correct: [32, 32, 32, 32, 32, 32]") == 3: return True
    return False

KS = [int(x) for x in os.environ.get('KS', '512,1024,2048,3072').split(',')]
N = int(os.environ.get('N', 9216))
REPS = int(os.environ.get('REPS', 5))
MS = [int(x) for x in os.environ.get('MS', '4,8,16,32,48,64,80,96,112,128,144,160').split(',')]

BW = 23.8e9          # bytes/s, measured
PEAK = 2.67e12       # MAC/s at the K=3072 87% point -> used as the practical ceiling

rng = np.random.default_rng(1)

if not gate():
    sys.exit("ABORT: device degraded before measuring")
print("gate: CLEAN\n")

print(f"  {'K':>5} {'M':>5} {'hw us':>8} {'GMAC':>8} {'GMAC/s':>9} {'%peak':>7} {'us/row':>8}")
summary = []
for K in KS:
    B = rng.integers(-127, 127, (N, K), dtype=np.int8)
    h = lib.npu_mm_prep_int8(p8(B), N, K)
    if h < 0:
        print(f"  K={K} prep failed {h}"); continue
    best_pct, best_M, mmax = 0.0, 0, 0
    for M in MS:
        Mp = ((M + 3)//4)*4
        A = rng.integers(-127, 127, (Mp, K), dtype=np.int8)
        out = np.zeros(3 * Mp * (((N + 47)//48)*48//3), np.int32)
        t = None
        for _ in range(REPS):
            if lib.npu_mm_run_int8_raw(p8(A), h, p32(out), Mp) != 0:
                t = None; break
            v = lib.npu_last_hw_ns() / 1e3
            if v > 0 and (t is None or v < t): t = v
        if t is None:
            print(f"  {K:>5} {M:>5} {'rc!=0 (feature budget exceeded)':>8}")
            break
        mmax = Mp
        mac = Mp * K * N
        pct = mac / (t * 1e-6) / PEAK * 100
        if pct > best_pct: best_pct, best_M = pct, Mp
        print(f"  {K:>5} {M:>5} {t:>8.0f} {mac/1e9:>8.2f} {mac/(t*1e-6)/1e9:>9.0f} {pct:>6.1f}% {t/Mp:>8.2f}")
    lib.npu_mm_free_int8(h)
    summary.append((K, mmax, best_M, best_pct, K*mmax))
    print()

print(f"  {'K':>6} {'M_max':>6} {'best M':>7} {'best %peak':>11} {'K*M_max':>9}")
for K, mmax, bM, bp, prod in summary:
    print(f"  {K:>6} {mmax:>6} {bM:>7} {bp:>10.1f}% {prod:>9}")
print("\nIf K*M_max is roughly CONSTANT the feature budget is the law and K/M trade on a")
print("hyperbola; the shape spec currently lists them as independent caps, which would be wrong.")
