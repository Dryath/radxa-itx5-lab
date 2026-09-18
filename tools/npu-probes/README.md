# NPU matmul shape probes

The measurement code behind the **shape-lock** in [`docs/03`](../../docs/03-going-fast.md) — how
we found that the usable batch is `M ∈ [112, 160]`, that `K·M ≤ 327,680` is the governing CBUF
budget, and that **M beats K at equal budget**. These probes sweep the NPU int8 matmul across
shapes and report achieved % of peak, so you can read the knee off your own board instead of
trusting ours.

The headline derivation, straight from `m_curve_probe.py`:

```
compute   2.67 GMAC/ms      bandwidth 23.8 GB/s
int8 = 1 byte/weight, 1 MAC/weight/row  →  2.67e12 / 23.8e9 = 112 MACs/weight-byte  →  M ≥ 112
```

## What each one does

| Probe | Sweeps | Answers |
|---|---|---|
| `m_curve_probe.py` | M ∈ {4…160} across several K | the arithmetic-intensity knee — how many rows must share a weight bank before the MAC array stops starving |
| `n_curve_probe.py` | N (output width) | where N stops helping / the dispatch overhead |
| `chain_shape_probe.py` | chained dispatches | whether chaining buys anything over amortised separate dispatches (spoiler: it doesn't) |

## Dependency (not included)

The probes load an **`npu_mm.so`** that exposes a small int8-matmul ABI:

```
npu_mm_init() -> int
npu_mm_prep_int8(int8* B, N, K) -> handle           # prep/reside a weight bank
npu_mm_run_int8_raw(int8* A, handle, int32* out, M)  # run M rows against it
npu_mm_free_int8(handle)
npu_last_hw_ns() -> long long                        # the driver's own hw_elapse_time
```

That bridge is **not shipped here** — it drives the RK3588 NPU over the reverse-engineered
register interface (see [`docs/02`](../../docs/02-the-command-stream.md) and the projects credited
in [`CREDITS.md`](../../CREDITS.md)); build your own against that ABI. The probes are the
*methodology*: point them at any int8 NPU matmul and they'll draw you the same curve.

## Running

```bash
NPU_MM_SO=./build/npu_mm.so python3 m_curve_probe.py
# optional gate (skip if you don't have a verifier): PYTHON=, NPU_VERIFY= point at an int4 verify script
```

Note the built-in **gate**: NPU state crosses process boundaries, so the probe validates a
known-good shape before it trusts a sweep. If you don't wire up a verifier, drop the `gate()`
call — but the lesson stands ([METHOD](../../METHOD.md)): never measure on an ungated device.
