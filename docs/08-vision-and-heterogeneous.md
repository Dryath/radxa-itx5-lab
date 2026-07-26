# 08 — Vision on the NPU (where the silicon finally wins)

Everything so far has been "the NPU can't beat the CPU for LLM decode." True — for **decode**.
But a **vision tower** is a different animal, and here the NPU earns its transistors. This is
also where the whole board comes together: **vision on the NPU, the MoE language model on the
CPU, at the same time.**

## Why vision is the NPU's game

A vision transformer (ViT) encoder is **dense matmuls** on a model small enough to stay
resident. Contrast with the MoE LLM story ([07](07-running-models.md)):

| | MoE LLM | ViT encoder |
|---|---|---|
| weights per forward | ~15 GB of experts (won't fit) | ~850 MB, **fits the 2.75 GB IOMMU** ([00](00-the-hardware.md)) |
| bound by | memory bandwidth | compute — exactly what the NPU is *for* |
| verdict on NPU | loses (cold-stage every token) | **wins** |

So the NPU's IOMMU size wall — the thing that kills MoE-on-NPU — is a non-issue for a small
dense tower. The weights load once and stay put.

## The result · `[MEASURED]`

Porting a full ViT tower (LFM2.5-VL-1.6B's SigLIP-style encoder, 27 layers, GEMMs → NPU via
3-core TP + M-batching; LayerNorm / softmax / GELU / bias on the CPU):

- **Full 27-layer tower: 4.38 s on the NPU vs 13.3 s on the CPU → 3.0×** (2.3× vs a 10.3 s
  reference image), **bit-exact to the fp32 CPU oracle** (rel-L2 ~1.6e-2, i.e. fp16-on-NPU
  matches fp32-on-CPU within quant noise).
- Getting there was the usual story: a naive Python-driven integration was **72 s (0.1×,
  slower than CPU)** — dominated by CPU-side layout/glue, not NPU compute. Moving attention
  onto the NPU and, critically, **rewriting the per-layer orchestration in C** (the Python
  numpy glue was ~68% of each layer) is what turned 0.1× into 3×. The NPU was always fast;
  the *plumbing* was the cost.

Two constraints worth stealing: the feature-map **M must be a multiple of 4** (M=49/50 give
garbage; pad to 48/52/…/64), and **M is CBUF-capped at ~64** per dispatch — tile above that.

**Honest status:** what's proven is the **encoder** (image → embeddings). The last mile —
bridging those embeddings into the LLM for end-to-end **OCR / image→text** — is scoped but
not finished. So: NPU vision encoder, measured and bit-exact; full OCR pipeline, the next
build.

## The point of it all: heterogeneous, concurrent

Here's the payoff that ties the whole repo together. On this board:

- **Dense vision → the NPU** (compute-bound, fits the IOMMU, 3× faster than CPU).
- **MoE language → the CPU** (bandwidth-bound, and the CPU already saturates the bus —
  [07](07-running-models.md)).

And they can run **at the same time**, because of the FENCE_OUT finding from
[05](05-kernel-and-tuning.md): a **non-blocking NPU submit hands the CPU back in ~9 µs**
while the NPU spends its ~280 µs/dispatch computing. That "disappointing" result (it wasn't
the 3× decode win we projected) turns out to be *exactly* the primitive you want here — it's
what lets the CPU keep decoding language tokens while the NPU chews on an image. The
projected-decode-lever became the **concurrency-lever**, and concurrency is the actual
architecture.

Do mind the shared bus ([04](04-the-bandwidth-wall.md)): CPU+NPU throughput doesn't add up
perfectly (~0.76×) when both are hammering memory simultaneously. But a **vision-encode
burst overlapped with CPU decode** isn't two memory-saturating loads at the same instant —
it's the good case, and it's why "put each workload on the hardware it suits, and overlap
them" is the design this board rewards.

---

← Back to the [README](../README.md). If you build the OCR last mile or measure the overlap,
we'd genuinely love a PR.
