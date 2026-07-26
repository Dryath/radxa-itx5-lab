# 00 — The hardware (know your enemy)

The RK3588's NPU is marketed as "6 TOPS." That number is technically true and almost
entirely useless. Here's what's actually under the hood.

## What it is

Three independent NPU cores, ~2 TOPS each (int8), that turn out to be a **derivative of
NVIDIA's open NVDLA design**. This is the single most useful fact in this whole repo:
when a register does something inexplicable, the answer is usually in the
[NVDLA hardware manual](https://nvdla.org/), not any Rockchip doc. The activation LUTs, the
cube/surface data layout, the SDP/CDP pipeline — all NVDLA lineage.

Each core is really two engines:

- **CNA** — the convolution/matrix engine. In LLM workloads it runs every projection in
  **GEMM (fully-connected) mode**, and it's where **99.9% of NPU compute time** goes.
- **DPU** — a post-processing unit. Despite early appearances (and a lot of our own wasted
  time — see [02](02-the-command-stream.md)), in generation it does almost nothing
  interesting: ~0.1% of the time, mostly a layout transform.

Supported types: int4 / int8 / int16 / fp16 / bf16 / tf32. Per-cycle peak examples from the
TRM: 1024×3 int8 MACs/cycle, 2048×3 int4. Internal buffer: 384 KiB × 3.

## Where it lives

Not where you'd think. It is **not** `/dev/rknpu`. It's a **DRM device** —
`/dev/dri/card1` (+ `card0`) — and everything you do to it is a DRM ioctl. See
[01](01-the-interface.md). Rockchip's kernel driver (`rknpu`, **GPL, public**) is the map;
read it.

## The two walls

Every optimization in this repo is a reaction to one of these two constraints. Internalize
them now and the rest of the docs will feel obvious.

### Wall #1 — LPDDR5 bandwidth (~25 GB/s, shared)

The DDR delivers **~25 GB/s of effective wall-time bandwidth** — roughly *half* the ~50
GB/s theoretical peak — and it is **shared across the CPU, GPU, and NPU** through one
memory controller. LLM decode reads a lot of weights per token, so decode speed is
governed almost entirely by this number, not by how many TOPS the NPU has. One A76 core
nearly saturates it on its own. Two masters running at once don't add up cleanly (CPU+NPU
concurrency ≈ 0.76×). This is the boss fight; it gets its own doc: [04](04-the-bandwidth-wall.md).

### Wall #2 — the 32-bit IOMMU (~2.75 GB usable per context)

The NPU's IOMMU is 32-bit, giving each context a **~2.75 GB usable address window**
(IOVAs are stored sign-extended in a u64 — always mask `& 0xFFFFFFFF`). The driver exposes
up to 16 domains. Consequences that shaped everything:

- A dense model whose weights fit in ~2.75 GB can live resident on the NPU. A big MoE whose
  experts total ~15 GB **cannot**, and pays a re-staging tax every forward pass — which is
  why "run the MoE on the NPU" keeps losing (see [06](06-the-graveyard.md)).
- **IOMMU domains do not reclaim on process exit.** Crash a run and the domain leaks;
  you reboot to clear it. Ask us how we know.

## The TOPS reality check

Rated 6 TOPS is int8-convolution marketing. In real LLM matmuls, a single core with the
best trick we found (M-batching, [03](03-going-fast.md)) reaches **~1.4 TOPS (~23%
utilization)**. Most of the time the chip sits at **0.17–0.75% utilization** — not because
it's slow, but because it's **starved**, waiting on memory and on a CPU that spins in a
completion poll. The NPU is rarely the bottleneck. The memory bus always is.

---

Next: [01 — The interface](01-the-interface.md) — how you actually talk to the thing.
