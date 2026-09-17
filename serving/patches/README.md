# llama.cpp serving patches for RK3588

The actual diffs behind [`docs/11`](../../docs/11-linear-attention-serving.md) — the CPU-side
serving optimizations for running (linear-attention / hybrid / MoE) models on the A76 cores.
Each is a small, self-contained change to **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)**
(MIT). They're **our own authored changes**, not a fork you have to track: apply the ones you
want to your own checkout and move on.

These were authored against a llama.cpp master around **Sept 2026**. On a different base,
`git apply --3way` (or `patch -p1`) will usually still land them — they touch small, localised
regions.

```bash
cd your-llama.cpp
git apply --3way /path/to/serving/patches/01-moe-gemm-grouped-mmid.patch
```

## What each one does (measured on RK3588, 4×A76, `-t 4`)

| Patch | Effect | Off-by-default gate |
|---|---|---|
| `01-moe-gemm-grouped-mmid` | Batch 4 routed MoE expert rows into one `q8_0x4` gemm instead of a gemv per row. **Prefill +18–39%** (LFM2.5-8B-A1B pp512 62.4→86.8). | `GGML_MMID_GEMM=0` disables |
| `02-mamba2-inplace-ssm-state` | Update the Mamba2 SSM state in place in the cache; removes a copy-back pass. **Decode +7–14%**; IPC 1.25→1.40. | `LLAMA_SSM_INPLACE=0` |
| `03-trimmed-lm-head` | Compute logits over a kept-row vocab subset, gather back to full vocab. Shrinks the decode matmul. | `output_trim` / `output_trim_map` |
| `04-ggml-thp-hugepages` | Back ≥16 MB aligned allocations with 2 MB transparent hugepages. Trims TLB pressure. | `GGML_HUGEPAGES=1` (opt-in) |
| `05-granitehybrid-iswa-graph` | Interleaved sliding-window-attention graph for hybrids. **Decode flat as context grows: 2.34× at 64K.** | — |
| `06-lfm2moe-sliding-window` | Read `attention.sliding_window` so the iSWA path is reachable for lfm2moe (was dead code). | `--override-kv` |
| `07-lfm2-rope-swa-seed` | Seed `rope_freq_base_train_swa` when SWA is forced — prevents a 100× rope-theta drop / garbage. | — |
| `08-granitehybrid-recurrent-rollback` | Gate `llm_arch_supports_rs_rollback` for Granite hybrid, so speculative decode can roll a rejected draft back without corrupting SSM state. | — |
| `09-speculative-prefix-draft-vocab` | Allow a draft vocab that is a strict prefix of the target's — enables a deliberately vocab-trimmed drafter. | — |
| `10-server-context-checkpoint-sidecar` | Persist recurrent/hybrid context in slot save/restore (`.ckpt` sidecar). A restore re-evaluates a handful of tokens instead of a full re-prefill. | — |
| `11-router-spawn-thread-affinity` | Re-widen spawn-thread affinity to the full `GOMP_CPU_AFFINITY` set before forking a worker — fixes libgomp piling all OMP threads on one core. **0.41 → 1.30 t/s** under `isolcpus=4-7`. | — |
| `12-router-pin-models-lru-exempt` | Exempt a pinned model from the router's LRU eviction, so the model you're serving isn't swapped out by a one-off background call. | `~/.config/llm-router/pinned` |
| `13-model-bailingmoe-linear` | Add the **`bailingmoe-linear`** architecture (**inclusionAI's [Ring-mini-linear-2.0](https://huggingface.co/inclusionAI)** — Lightning-Attention-2 on 16/20 layers + `bailingmoe2` MoE). The architecture and reference are inclusionAI's; this is the llama.cpp integration (logits match the reference to 4 dp). | — |

## Credit & license

- These patches modify **llama.cpp** (Georgi Gerganov and contributors), MIT — the patches are
  offered under the same terms.
- Patch `13` integrates **inclusionAI's** `bailingmoe-linear` / Ring-mini-linear architecture;
  the model design and reference implementation are theirs.
- The numbers are single-board, mostly `-fa 0`, several on Granite/LFM2 test models — read the
  [residency caveat](../../docs/10-the-npu-backend.md) before quoting any decode figure.
