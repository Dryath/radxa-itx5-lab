# 11 — Linear-attention serving: making the A76s earn their keep

The [NPU backend](10-the-npu-backend.md) is for prefill-heavy work. For everything else the
answer is the tuned CPU path ([07](07-running-models.md)) — and the biggest gains there come
from **picking architectures the A76 likes** and then shaving the CPU-side overhead the
generic build leaves on the table. This is a set of patches on top of
[ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) (Georgi Gerganov and contributors)
that do exactly that.

## Why linear attention, on this board

A hybrid **linear-attention MoE** is close to ideal for a bandwidth-bound board: linear
attention carries a **constant-size recurrent state** (no KV cache growing with context to feed
the [bandwidth wall](04-the-bandwidth-wall.md)), and the MoE reads only active experts per
token. The specific model here is **inclusionAI's [Ring-mini-linear-2.0](https://huggingface.co/inclusionAI)**
(`bailingmoe-linear`): **Lightning-Attention-2 on 16 of 20 layers** via `ggml_gated_linear_attn`,
with the `bailingmoe2` MoE block unchanged. The architecture and reference implementation are
inclusionAI's; the work here is wiring it into llama.cpp (logits match the reference to 4 dp)
and the serving patches below.

## The patches (measured; RK3588, 4×A76, `-t 4`)

**Prefill — batch the MoE routing.** llama.cpp runs each routed expert row as a separate gemv;
grouping 4 rows into a `block_q8_0x4` gemm turns the A76's biggest prefill cost (routed matmul
≈ 47% of it) into a batched op:

| model | pp512 off → on |
|---|---|
| LFM2.5-8B-A1B | 62.4 → **86.8 (+39%)** |
| Ring-mini-linear-2.0 | 91.3 → **110.9 (+21%)** |
| Granite-4.0-H-Tiny | 66.6 → 78.5 (+18%) |

Decode unchanged (it's not gemm-bound). Gated by `GGML_MMID_GEMM`.

**Decode — update the SSM state in place.** The Mamba2 recurrent state was being copied back
each step; updating it in place in the cache removes a whole memory pass:

- Granite-4.0-H-Tiny tg64 **17.76 → 20.19 (+13.7%)**; granite-350m **66.10 → 71.09 (+7.5%)**.
- Under the hood: **cycles/token 515M → 451M, IPC 1.25 → 1.40, backend stalls 65% → 61%** —
  and perplexity identical (6.1230). Gated by `LLAMA_SSM_INPLACE`.

**Long context — window the attention.** An interleaved sliding-window-attention (iSWA) graph
for hybrid models keeps decode flat as context grows instead of paying for the full span:

| context | tg32 off → on (window 2048) |
|---|---|
| 16K | 13.57 → 17.03 |
| 32K | 10.78 → 17.05 |
| **64K** | **7.25 → 17.00 (2.34×)** |

Prefill stays flat. (Also needed: seeding `rope_freq_base_train_swa` when SWA is forced, or
high-theta models emit garbage — a 100× rope-theta drop otherwise.)

**Decode — trim the LM head.** Computing logits over a kept-row subset (then gathering back to
full vocab) shrinks the decode matmul: Ring-mini-linear **157,184 → 89,028 rows**, tg64
25.04 → 26.87. Small but free.

**Memory — back big allocations with hugepages.** ≥16 MB allocations aligned to 2 MB and
`MADV_HUGEPAGE`'d (`GGML_HUGEPAGES=1`) trims TLB pressure: Ring-mini-linear 21.74 → 22.64 tg.
Modest, and off by default.

## The serving layer

The same fork carries the multi-model router that the [appliance](13-the-appliance.md) runs on,
and two fixes there matter more than their size suggests:

- **Core-affinity, done right.** libgomp inherits a *single-core* mask when the router forks a
  worker, so every OpenMP thread piles onto one core — the same `isolcpus` footgun from
  [05](05-kernel-and-tuning.md), now inside the server. Re-widening spawn-thread affinity to the
  full `GOMP_CPU_AFFINITY` set before the fork: **0.41 → 1.30 t/s** on RK3588 with
  `isolcpus=4-7`. This is the difference between "the board is broken" and "the board is fine."
- **Context checkpoints in slot save/restore** (`.ckpt` sidecar). Recurrent/hybrid models can't
  reconstruct their state from the token cache alone, so a slot restore used to force a full
  re-prefill. Persisting the state means a post-restore resend re-evaluates **4 tokens instead
  of thousands** — the thing that makes a swappable-model server usable for these architectures.

Plus the plumbing hybrids need for speculative decoding: a draft vocab that is a strict prefix
of the target (so a deliberately trimmed drafter is legal), and recurrent partial rollback so a
rejected draft doesn't corrupt the SSM state.

## Caveats

Numbers are single-board, mostly `-fa 0`, and several are on Granite/LFM2 test models rather
than the named linear model. Every patch is gated behind an env var and **off by default** —
they're opt-in tuning, not silent behaviour changes. No figures in this line of work have been
withdrawn, but the [same residency caveat](10-the-npu-backend.md) applies: quote decode numbers
only from harnesses with distinct prompts.

---

Next: [12 — On-device agent memory](12-agent-memory.md) — architecture above the tokens.
