# 05 — Kernel & tuning (the config that unlocks it, and the traps)

You can do a surprising amount from userspace, but a few kernel choices are the difference
between "works" and "fails with a cryptic errno." Board here: Radxa ROCK 5 ITX+, custom
Radxa BSP `6.1.84-rk2410`.

## The kernel options that actually matter

### `CONFIG_ROCKCHIP_RKNPU_DRM_GEM=y` — the one that breaks everything if wrong
The NPU buffers must go through **DRM GEM**, not DMA-heap. Build the driver in DMA-heap mode
and *every* NPU allocation fails with `errno 22` (`EINVAL`) — from the vendor runtime *and*
from your own raw-DRM code. If nothing can allocate, check this first.

### `CONFIG_ROCKCHIP_RKNPU_FENCE=y` (needs `SYNC_FILE`) — the reason to run a custom kernel
Stock `-8`-style images tend to return `FENCE_OUT EINVAL`, which forces every submit onto
the blocking IRQ path. Since decode is completion-latency bound (the CPU spins 70–82% of
wall — [01](01-the-interface.md)), a *working* `FENCE_OUT` lets you get a real fence fd and
**busy-poll** it instead of sleeping in `wait_event_timeout`. We **confirmed the mechanism
live** (submit returns 0, real fence_fd, `poll` POLLIN, bit-exact result).
**[projected] ~3× decode** from switching the completion path — but see the promissory-note
box below; the end-to-end decode win is not yet a measured number.

### `PROC_FS` / `DEBUG_FS` on
For NPU load/monitoring visibility. Cheap, worth it.

### `CMDLINE_EXTEND` — a missing arm64 Kconfig entry
The `CMDLINE_EXTEND` option was never ported to arm64 (the C implementation exists in
`fdt.c` / `kaslr_early.c`, but the Kconfig entry doesn't). `kernel/patches/` re-adds it.
Belt-and-suspenders — in practice params get set via `/etc/kernel/cmdline` anyway.

### Per-board CMA (boot cmdline, not baked in)
`rk_dma_heap_cma=<N>G` sizes the pool that holds resident weights (e.g. 12G on a 32 GB
board for an 8B model across two IOMMU domains; 8G on a 24 GB board for smaller models).
**CMA size does not affect bandwidth or tok/s** — it only governs how much model can stay
resident. Don't expect a speedup from a bigger pool.

> Build gotcha: `FENCE`/`PROC_FS` love to silently revert to `# not set` when a dependency
> is unmet or the `.config` is stale. Always `grep` the built `.config` afterwards.

## Tuning levers, ranked by payoff

| Rank | Lever | Effect |
|---:|---|---|
| 1 | **Pin to A76 (cores 4–7)** | +53% decode ([03](03-going-fast.md)) |
| 2 | **int8 dispatch** | +70–87% ([03](03-going-fast.md)) |
| 3 | **FENCE busy-poll + shallow idle** | ~3× decode **[projected]** |
| 4 | **Governors → performance** (esp. `dmc`) | first-token-after-idle only; without it the DDR idles down to 534 MHz |
| 5 | DDR overclock | ~+14% theoretical, muted in practice; not worth the risk (see [06](06-the-graveyard.md)) |
| 6 | THP (transparent hugepages) | **inert** here — 0 huge pages materialized under `defer+madvise` without an explicit `madvise(MADV_HUGEPAGE)` in code |

Also: **`Q4_0` beats `Q4_K_M`** on this A76 on *both* prefill and decode — the K-quant
unpack cost dominates. Use plain `Q4_0` (and `Q4_0_4x4` repack for ~+2% more) unless you
have a specific reason not to.

## ⚠️ The `isolcpus` landmine (a 19× footgun)

`isolcpus=4-7 irqaffinity=0-3` buys **+3–4%** decode... and can cost you **19×** if you're
not careful. With `GGML_OPENMP=ON`, `--cpu-mask` / `--cpu-strict` are **silently ignored**,
and `isolcpus` removes the load balancer — so every OpenMP thread piles onto cpu4 and you
get **0.41 tok/s where you expected 8.33.**

Rules if you touch `isolcpus`:
1. Never add it without shipping OpenMP pinning in the *same* change
   (`GOMP_CPU_AFFINITY="4 5 6 7"` + `taskset -c 4-7`).
2. Verify with **PSR** (processor the thread actually ran on), not `%CPU`, and do it
   **during generation**, not during load.

Honestly? The +3–4% usually isn't worth the risk. Pinning ([03](03-going-fast.md)) gets you
most of it safely.

## The promissory-note box

The `FENCE_OUT` busy-poll is the biggest *unrealized* win in the repo. The mechanism is
confirmed; the ~3× is an extrapolation from where the time goes (70–82% in the completion
path), not a measured decode delta. If you build on this, **measure the end-to-end tok/s**
before quoting a number — and please send it back, we'd love to know.

---

Next: [06 — The graveyard](06-the-graveyard.md) — the good ideas that didn't survive contact
with the silicon.
