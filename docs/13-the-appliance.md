# 13 — The appliance: a whole board as a LAN LLM node

Everything else in this repo is components. This is where they run together: a single **ROCK 5
ITX (32 GB)** turned into a self-contained **LAN LLM appliance** — a llama.cpp router serving
swappable models over the network, with NPU-backed tools and the memory discipline from
[12](12-agent-memory.md). No cloud, nothing leaves the box.

## The router

Built on the serving fork from [11](11-linear-attention-serving.md). It holds one main model
resident and swaps it on demand, under two rules that keep a 32 GB board from falling over:

- **Memory-guarded swaps.** A load is refused (or an eviction forced) before it would OOM the
  board, not after. On a device with no swap headroom to spare, "check first" is the whole game.
- **Pin the model you actually use.** The router's LRU can evict a cold model, but a **pinned**
  main model is exempt — so the thing serving your requests never gets swapped out from under
  you by a one-off background call.

Serving defaults that come straight from the [bandwidth wall](04-the-bandwidth-wall.md) and
[running-models](07-running-models.md):

- **A76-pinned, `-t 4`** — the A55s cost you ([10](10-the-npu-backend.md)).
- A **32K working window** by default (131K available as a preset) — big enough to be useful,
  small enough that prefill time stays sane.
- **`Q8_0` K-cache** to halve KV bytes, and a **trimmed-`Q4_0`-head GGUF** ([11](11-linear-attention-serving.md))
  to shave the decode matmul. Bytes-per-token, everywhere.

## NPU-backed tools

The appliance exposes board tools to its agent harness, and the expensive one runs on the NPU
rather than the CPU — the [vision-on-NPU](08-vision-and-heterogeneous.md) result put to work:

- **`ocr_image`** is a **bare-metal PP-OCRv4 pipeline on the NPU** — image → text without
  touching the language model or the CPU's LLM budget. This is the "dense vision belongs on the
  NPU, the LLM stays on the CPU" split from [08](08-vision-and-heterogeneous.md), shipped.
- A small **LFM2.5 subagent worker** handles cheap background tasks so the main model isn't
  interrupted for them — the perception-discipline principle ([12](12-agent-memory.md)) applied
  to *which model* answers: cheapest capable worker first.

## Choosing the main model: bake it off, don't guess

Which model sits resident is a **measured** decision, not a vibe. The appliance ships with a
quantisation/model bake-off: run the candidates on the *real* prompt the appliance actually uses
(not a generic benchmark), at the quant levels the board can afford, and pick on delivered
quality-per-byte. Some candidates that look good on paper lose on this board's real prompt — the
only way to know is to run them here. (This is [METHOD](../METHOD.md) rule 2: the target is
fixed, the choice is the measurement's.)

## The idle window is part of the design

An appliance is on all night doing nothing — so that's where the expensive work goes
([12](12-agent-memory.md)): RTC-scheduled nightly jobs to consolidate the day's episodic memory
into long-term, let unreferenced traces decay, and run any batch maintenance. The board's own
schedule is a feature; the architecture is built to use it.

---

Next: [14 — Cartridges](14-cartridges.md) — one base model, many hot-swappable domains.
