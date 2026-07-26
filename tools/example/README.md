# npu_gemm_test.c — a minimal raw-DRM CNA matmul

The smallest end-to-end thing that proves the point: a **16×16 fp16 matrix multiply run
directly on the NPU** via DRM ioctls, no vendor library, output compared against a CPU
reference. If this runs and matches, you're talking to the silicon.

## How it works

It builds the register command buffer from a **known-working template** (captured with the
[tracer](../tracing/)) and patches only the IOVA and dimension registers — the minimum to
retarget the op at your own buffers. See [docs/02](../../docs/02-the-command-stream.md) for
the regcmd format this relies on.

## Dependencies

- The reverse-engineered NPU headers (`npu_cna.h`, `npu_hw.h`, `rknpu-ioctl.h`, …). These
  are **not vendored** — see [`CREDITS.md`](../../CREDITS.md); obtain them from the upstream
  RE projects (Matharu / allbilly) or the Rockchip GPL driver UAPI.
- A weight blob to multiply against. The default path is `weights.gguf`; pass your own as
  the first argument.
- A kernel built with `ROCKCHIP_RKNPU_DRM_GEM=y` (see [docs/05](../../docs/05-kernel-and-tuning.md)).

## Build & run

```bash
gcc -O2 -o npu_gemm_test npu_gemm_test.c -I. -lm
./npu_gemm_test [weights.gguf]
```

Note the CNA DMA quirk baked into the example: it forces a **low IOVA** (allocating a large
buffer so the allocation lands below the ~512 MB address the CNA DMA seems to require).
That's not a bug in your code — it's the hardware.
