# Benchmarks — measure your own silicon

The harness behind the numbers in [`docs/04`](../../docs/04-the-bandwidth-wall.md). Run
these first on any RK3588 board so you're arguing from *your* measurements, not ours.
Pin to the A76 cluster and set governors to `performance` first (see
[`rk3588-perf.sh`](rk3588-perf.sh)).

| File | Measures | Build |
|---|---|---|
| `membw.c` | DDR bandwidth (STREAM Copy/Scale/Triad) vs thread count — find your real ceiling | `gcc -O3 -fopenmp membw.c -o membw` |
| `cpu_matmul_bench.c` | CPU matmul (Q4_0 × F32) per-token cost at various shapes | `gcc -O3 -fopenmp cpu_matmul_bench.c -o cpu_matmul_bench -lm` |
| `vk_membw.c` + `vk_membw.comp` | GPU (Mali) bandwidth solo, CPU solo, and both at once (the shared-bus tax) | needs Vulkan; compile the `.comp` with `glslangValidator` |
| `concurrency_probe.c` | NPU + CPU running together → the concurrency additivity factor (~0.76×) | needs an NPU dispatch backend (see note) |
| `rk3588-perf.sh` | Pins A76, sets `performance` governors incl. `dmc` — run before benchmarking | `bash rk3588-perf.sh` |

## Expected shape of results

- `membw`: peaks around **~23–25 GB/s at ~2 threads**, and *drops* if you add more — the
  bus is the bottleneck, not the cores.
- `concurrency_probe`: NPU-solo + CPU-solo throughput does **not** add up when run together
  (~0.76×) — the point of [docs/04](../../docs/04-the-bandwidth-wall.md).

## Note on `concurrency_probe.c`

It drives the NPU with a real matmul loop while running a STREAM triad on the A76s. That
requires an **NPU dispatch backend** providing `npu_drm.c` / `npu_matmul_kernel.c` (e.g.
Matharu's [`rk3588-npu`](https://github.com/mtx512/rk3588-npu)). The build comment in the
file shows the include/link shape; wire in whichever backend you're using.
