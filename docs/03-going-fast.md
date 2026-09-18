# 03 — Going fast (the levers that worked)

Everything here is measured on-device, bit-exact against a CPU reference unless noted.
Where a number is a projection rather than a measurement, it says **[projected]**.

## Lever 1 — M-batching (the free lunch)

**The headline discovery.** The CNA has a spatial ("image height") dimension that sits
completely idle when you do one vector at a time (M=1). Feed it a batch of rows and the
work is nearly free until you fill it.

Single core, K=2048, N=2048:

| M | wall time | per-vector | speedup |
|---:|---:|---:|---:|
| 1 | 387 µs | 387 µs | 1× |
| 32 | 387 µs | 12.1 µs | **31.84×** |

Wall time is **flat from M=1 to M=32**, bit-exact (65536/65536 elements match). Utilization
goes 0.6% → ~23% (~1.4 TOPS). This is the difference between "the NPU is a toy" and "the NPU
is useful."

**Constraints** (learned the hard way):
- `K % 32 == 0`, `N % 16 == 0`, and `M == 1 || M % 4 == 0`
- CBUF ceiling: `fd_banks = ceil(M·K·2 / 32768) ≤ 11`. Practically, **M=32 is the stable
  ceiling**; **M=64 returns `err=110` (ETIMEDOUT)**.
- The batch ceiling is set by your *largest* K. The FFN down-projection (K=6144) is the
  worst offender and caps model-wide M at ~32 (less on bigger models).

**Update — the ceiling is a budget, and M goes much higher than 32.** The "M=32" above was the
conservative read at large K. The real governing constraint is a single **CBUF feature budget:
`K·M ≤ 327,680`** (10 banks × 32 KB) — and it predicts the exact M-cliff (at K=3072, M=96 passes
and M=112 fails). So at the production shape **K=2048 the usable range is M ∈ [112, 160]**, an
order of magnitude past 32. Two things fall out of measuring it properly:

- **M beats K at equal budget.** (2048, 144) hits **77.2% of peak**; (3072, 96) — same budget —
  only 54.9%. Wider batch, not deeper K.
- **The arithmetic-intensity floor is 112 MACs/weight-byte** (2.67 GMAC/ms ÷ 23.8 GB/s), which
  is *why* usable M starts at ~112 — below that you're bandwidth-starved, not compute-fed.

**The shape locks at K=2048, M=144, N ≥ 4608** (~76% of peak in a single dispatch), int8 with a
Hadamard transform. A cost model this clean — predicted 1696 µs at K=3072/N=9216/M=160, measured
**1695 µs** — means you can *choose* the shape before you run it. (One oddity worth knowing: M=128
is anomalously bad; M=144 recovers. Don't assume monotonic.) The probe code that measures this
whole curve is in [`tools/npu-probes/`](../tools/npu-probes/).

## Lever 2 — 3-core tensor parallelism (TP3)

Cycle `core_mask` `0x1 → 0x2 → 0x4` across three single-core SUBMITs; they run in parallel.
Pad `N` to a multiple of 48 to balance the (slightly uneven) slices. Measured, warm:

- q/o projections: **27.96×** (combined with M=32)
- gate/up: **28.25×**
- down_proj (K=6144, CBUF-bound): **23.26×**
- **TP3 × M=32 combined ≈ 62×** over single-core M=1.

## Lever 3 — pin to the A76 cluster (+53%, free)

The A55 little cores drag everything down through shared sync barriers and bandwidth
contention. Pin the inference threads to the 4× A76 (cores 4–7):

| | decode |
|---|---|
| unpinned | 4.80 tok/s |
| **pinned to A76** | **7.34 tok/s (+53%)** |

Variance also collapses (7.34 / 7.34 / 7.34 across runs). At this point NPU work is ~131
ms/tok = **96.4%** of per-token time — i.e. you've squeezed the CPU side and the memory bus
is now the ceiling. (Do **not** reach for `isolcpus` to push this further without reading
the landmine warning in [05](05-kernel-and-tuning.md).)

## Lever 4 — int8 dispatch (+70–87%)

Dispatch matmuls as int8 instead of fp16: half the DMA bytes per weight, and native int8
MACs. Measured across two model sizes:

| model | fp16 | int8 | gain |
|---|---:|---:|---:|
| 1.7B | 7.34 tok/s | 12.51 tok/s | **+70%** |
| 4B | 3.09 tok/s | 5.77 tok/s | **+87%** |

A per-row int8 sidecar (int8 weights + fp32 scales, NEON quant/dequant on the CPU side)
achieved this at a **mean quantization error of 0.00027** — negligible. One warning from
experience: make sure the int8 data path uses the **int8 kernel** — a routing bug that ran
int8 data through the fp16 kernel silently corrupted every FFN-down output. Bit-exact
testing catches this; eyeballing token output does not.

## Lever 5 — persistent weight residency

Cold-staging weights into NPU GEM buffers every forward pass is death. Allocate the weight
GEMs once, reorder/upload them a single time, and keep them resident (a "persistent" flag
so they live for the daemon's lifetime). Prefill measurements saw persistent-GEM reuse give
an **8.5×** step on its own; on a llama.cpp-backend variant the same idea (populate a slab
store, reuse warm) took prefill from 30.98 → **46.04 t/s (+48.6%)**. Bonus: with weights
already resident, your wake-from-idle latency is just prefill, not a reload.

## What stacking these looks like

As a concrete case study, a Qwen3-1.7B decode walked from a **3.2 tok/s** naive baseline to
**~10.9 tok/s** by combining a compacted prompt, the int8 sidecar (+87%), persistent GEMs,
M-batched prefill, and A76 pinning — while dropping from 2 IOMMU domains to 1 (half the NPU
memory). None of these levers is exotic; they're all just "respect the hardware."

## The honest asterisk (now measured)

We *projected* one more lever — a non-blocking `FENCE_OUT` submit instead of the kernel's
blocking wait — at **~3× decode**. We since measured it, and the projection was wrong in a
useful way: dispatch turns out to be **compute-bound** (~280 µs NPU vs ~9 µs submit), so
there's no ioctl-wait to reclaim and decode stays bandwidth-bound. The real, measured payoff
is **CPU liberation** — run the CPU *while* the NPU computes — not throughput. Full autopsy
in [05](05-kernel-and-tuning.md).

---

Next: [04 — The bandwidth wall](04-the-bandwidth-wall.md) — why all of the above eventually
hits the same ceiling.
