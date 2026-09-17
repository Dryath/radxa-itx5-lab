# 10 — Running LLMs on the NPU: the RKNPU2 llama.cpp backend

Upstream llama.cpp has **no Rockchip NPU backend at all** — the CPU is the only path. That it
runs on the RK3588 NPU is entirely thanks to prior work, which this note builds on. Credit
first, because none of the numbers below would exist without it.

## Standing on shoulders — read this before the benchmarks

- **The RKNPU2 backend is [@invisiofficial](https://github.com/invisiofficial)'s work**
  ([invisiofficial/rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp)): the backend
  itself, quantisation support, the hardware pipelines and hybrid quantisation, IOMMU domain
  management, the caching system, the environment-variable surface. *Without this, nothing here
  exists.* Contributors credited in that project: **@Polarnik** (zero-copy weights, weight
  pre-packing), **@hvalev**, **Gerald Tan** (@woefulwabbit, cross-compilation), **Martino
  Mensio** (@MartinoMensio, build fixes).
- **[Mojo24x7/rk-llama.cpp](https://github.com/Mojo24x7/rk-llama.cpp)** carries that backend
  onto a **current** llama.cpp master (it had been stuck on a May-2026 base). That's the tree
  this lab's work sits on top of. **@danielferr85** independently rebased the same backend and
  identified which backend-interface slots had changed — which saved the search.
- **[ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)** (Georgi Gerganov and
  contributors) is the parent project underneath all of it.

What the lab added on top: MoE experts executed **on** the NPU, cross-board tensor
parallelism, a quantisation-eligibility routing fix, and the measurement study below. The
backend's own README/CONTRIBUTING under `ggml/src/ggml-rknpu2/` are the authors' and ship
unmodified.

## The shape of the thing

The backend implements only **`MUL_MAT` / `MUL_MAT_ID`** (matmul and MoE-routed matmul), plus
optional on-device "glue" ops; everything else runs on the CPU with a per-layer handoff. That
matches the hardware exactly: the NPU is a matrix engine ([02](02-the-command-stream.md)), and
the [bandwidth wall](04-the-bandwidth-wall.md) means **the NPU wins prefill, and decode stays
CPU-bound** (NPU 23.2 GB/s vs CPU 22.9 GB/s — a wash for the memory-bound decode phase).

## The wins (measured, 16 GB ROCK 5B+, warm, first run discarded)

| Change | Effect |
|---|---|
| **MoE experts on the NPU** (batched `MUL_MAT_ID`, int8) | **5.02 → 0.518 s/layer (9.7×)** |
| Correct device type + KV placement (gemma-3-1B decode) | **2.39 → 17.32 t/s (7.2×)** |
| `Q4_0` routed through the W8A8 int8 pipeline | the 9.7× above — see quant rule below |
| Per-channel INT4 attention scales (`RKNPU_PERCHAN`) | perplexity **+43% → +5.0%** |
| Glue-op locality (keep ops on-device) | inter-device traffic **13.9 → 3.3 GB/req**; graph splits **482 → 196** |
| `ggml-rpc` weight transfer fast path | **38 → 280 MB/s** (wire-limited on 2.5 GbE) |
| `performance` governors | **+32%** decode |
| **30B production sweep** | baseline PP 8.66 / TG 6.5 → **PP 21.1 / TG 9.6** (the deployed config) |

## The quantisation-eligibility rule (the load-bearing one)

The NPU only accepts a few GGUF types, and only **symmetric** precisions
(W16A16 / W8A8 / W4A4). Eligible: **F16, Q8_0, Q6_K, Q4_0**. The trick that unlocks the 9.7×:
**route `Q4_0` through the `W8A8_STANDARD` (int8) pipeline** rather than a 4-bit path — because
the NPU matmul only *batches* in int8:

- NPU matmul ceiling (K=2048, N=768, 3 cores): **INT8 reaches 1563 GFLOPS at M=1024; INT4 is
  flat at ~120 GFLOPS — it does not batch.** So int4 weights, int8 compute.
- MoE per-layer, same kernel: CPU 5.02 s → NPU naive 4.94 s → batched int4 2.85 s → **batched
  int8 0.518 s.**

Perplexity cost is real and worth knowing (Qwen3-30B-A3B Q4_0): W8A8 top-8 **3.5447**; W8A8
top-4 (deployed) **4.0619**; W4A4 per-channel top-4 **4.2669 (+5.0%)**; W4A4 *per-block* top-4
**5.8114 (+43.1%)** — the per-channel scales are what make 4-bit attention usable at all.

## What actually runs (single board unless noted; PP=prefill, TG=decode, t/s)

| Model | Quant (size) | PP | TG |
|---|---|---:|---:|
| Qwen3-30B-A3B | Q4_0 (17.3 GB) | **21.1** | **9.6** |
| Qwen3.6-35B-A3B | Q4_0 (20.8 GB) | 18.5 | 4.07 |
| gemma-4-E4B | Q8_0 (8.2 GB) | 38.2 | 4.04 |
| gemma-3-1B | Q8_0 (1 GB) | 201.6 | 17.0 |
| gpt-oss-120b | Q8_0 (**60 GB — 4× board RAM**, multi-board) | — | 0.88, *coherent* |
| Qwen3.6-27B dense | Q8_0, 3 boards, tensor-parallel | — | **1.05** (vs 0.53 layer-split) |

**Cross-board tensor parallelism** (`rktp`: row-split matmuls, N-shard) beats naive layer
pipelining **2.0×** on decode for a dense model that doesn't fit one board — but the alignment
boundary is unforgiving (`LOCFRAC` off the boundary silently corrupts output).

## Platform constants worth pinning to the wall

- CPU bandwidth **22.9 GB/s**, NPU **23.2 GB/s** (peak 29.7), **CPU+NPU concurrent 27.3 GB/s**
  — the shared bus again ([04](04-the-bandwidth-wall.md)).
- **A55 cores cost 56%** on decode: `-t 8` gives 4.26 t/s vs `-t 4` at 9.66. Pin to the A76s.
- **Residency is the dominant decode variable** — same model, same board, 9.15 vs 6.00 t/s
  (+52%) depending only on what's paged in.
- Board↔board over 2.5 GbE tops out at **280 MB/s** (the tensor-parallel wire limit).

## Honest caveats — figures we withdrew

Keeping these visible is the point of the [method](../METHOD.md):

- A hybrid-decode **"+84%"** claim was **retracted** — page-cache residency inflated it.
  Controlled re-test: ~5.6 t/s; no clean ratio is quoted because it wasn't re-measured. Only
  the *mechanism* stands (the fused Gated-DeltaNet path silently disables without
  `LLAMA_RECURRENT_ON_CPU=1`).
- A flash-attention **"break-even at ~92 tokens"** was withdrawn — it's depth-dependent.
- An **"~11 GB/s"** NPU-bandwidth number was measured on too small a matrix; corrected to 23.2.
- A **"+37% from disabling glue"** was a cache-ordering artifact; retracted.
- A speculative-decoding **"defaults are 2× better"** was intra-prompt redundancy; retracted.
- **General rule (from the fork's own README):** *"any number whose harness cannot be shown to
  use distinct prompts is an upper bound, not a measurement."* Residency alone swings results
  up to 52%.
- **Negatives:** speculative decoding *loses* in every config on sparse MoE (−8% to −14.6%);
  the Mali GPU is 22–32× slower than CPU; more CPU threads always lose; power was never
  measured.

---

Next: [11 — Linear-attention serving](11-linear-attention-serving.md) — the CPU-side story,
for the models that don't need the NPU at all.
