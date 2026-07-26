# 06 — The graveyard (good ideas, honest autopsies)

Negative results are the most valuable thing in a reverse-engineering repo and the rarest
thing published. Here's what we tried that *didn't* work, with the measurement that killed
it, so you don't have to re-dig these graves.

## ☠️ Dense models on the NPU

**Verdict: 0.2 tok/s NPU vs 44.3 tok/s CPU** (dense 0.5B). A dense model must re-stream and
dequantize *all* its weights every token; the fill overhead buries any compute win. The NPU
streaming path only pays off for **MoE**, where 2–4 active experts mean you move 2–4 GB
instead of everything. Dense is the worst case. Don't.

## ☠️ Multi-core dispatch chaining ("Path A")

The idea: chain multiple ops across cores in one dispatch to amortize submits. **Verdict:
wedges the NPU** — hard enough to need a reboot (`RKNPU_ACT_RESET` didn't clear it). After
fixing the obvious missing PC-chain register writes, a targeted probe showed cores 1 & 2
**silently drop the second task's `DPU_DST_BASE_ADD` latch** — a **hardware errata below the
regcmd interface** (core 0 sets raw-status bit 31; cores 1 & 2 don't). You cannot fix this
from software. **Do not chain dispatches.** Use TP3 + M-batching instead
([03](03-going-fast.md)).

## ☠️ Attention offload to the GPU (Mali G610, Vulkan)

**Verdict: slower than CPU at every sequence length.** seq=25: 0.025 ms CPU vs 10.0 ms GPU
(0.003×). seq=4000: 21.6 ms vs 166.2 ms (0.130×). Cause: 16 sequential shader dispatches at
~500 µs launch overhead each, a local size of 64 on an 80-core GPU, and fp32 K/V. And the
reframe that made it moot: **attention is only ~5% of per-token wall** at typical positions
anyway. (Shader kept around for a possible long-context future >500 tokens, but it's not the
lever.)

## ☠️ Speculative decoding

**Verdict: not viable here (yet).** The economics need the draft-to-target cost ratio
`r ≤ 0.20` for the headline 2–4×. Best available draft managed **r = 0.38** (draft 29.72
tok/s vs target 11.3), and small drafts hit a ~30 tok/s floor because per-dispatch overhead
is ~65% of a small model's token time. At realistic acceptance rates you land between a 26%
*loss* and a 10% gain — not worth the complexity until a genuinely faster small draft (or
lower dispatch overhead) exists.

## ☠️ int8 KV-cache storage

**Verdict: −7%** (at ~340 tokens of context). The dequant-on-read overhead exceeds the
memory-bandwidth savings for KV. Keep KV in f16.

## ☠️ Multi-context concurrency

**Verdict: 1.05×** for two contexts (essentially nothing). The NPU is a single serial
device. The way to add throughput is **M-batching within one context**, not parallel
contexts. (A concurrent *scheduler* that batches multiple users' tokens into shared
dispatches, on the other hand, *does* work — measured **1.95× at 16 users**, with per-user
latency flat at ~85 ms/tok. The trick is sharing dispatches, not running the device twice.)

## ☠️ All-8-core decode

**Verdict: 2.4× slower than 4×A76.** The A55 little cores add sync-barrier and bandwidth
contention with no throughput to show for it. Pin to the big cluster and stop there.

## ☠️ KleidiAI on the A76

**Verdict: −2% decode.** It wants `i8mm`/SME instructions the A76 doesn't have, rejects all
559 tensors, and *silently disables* the CPU weight-repack (18645 MiB → 0) while it's at it.
Leave it **off** on RK3588.

## ☠️ DDR overclock

**Verdict: parked.** ~+14% theoretical, muted in practice, and "not worth the brick risk"
for a lab you depend on. The overclock blob exists; it stays on the shelf.

---

## The pattern

Notice the theme: almost every dead end died to the **bandwidth wall** or the **completion-
poll latency** or the **IOMMU size limit** — the same three constraints from
[00](00-the-hardware.md) and [04](04-the-bandwidth-wall.md). Once you internalize those,
you can predict which clever idea is doomed *before* spending three days on it. We didn't
have that intuition at the start. Now you do.

← Back to the [README](../README.md) · or start again at [00 — The hardware](00-the-hardware.md).
