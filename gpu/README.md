# LLM ops on the Mali-G610 via Vulkan compute

A small, self-contained Vulkan compute library for the RK3588's **Mali-G610** GPU: the matmul and
transformer ops you need to run an LLM layer on the GPU instead of the CPU or NPU. Written from
scratch (no RKNN, no vendor GPU stack) — a single Vulkan context plus SPIR-V compute shaders.

## What's here

| Shader | Op |
|---|---|
| `shaders/sgemm.comp` | the matmul (pre-loaded weight path) |
| `shaders/attn.comp` | attention |
| `shaders/rms_norm.comp` | RMSNorm |
| `shaders/rope.comp` | rotary position embedding |
| `shaders/swiglu.comp` | SwiGLU FFN activation |

`vk_ctx.c` is the Vulkan context (device, queue, shader loading); `gpu_ops.c` is the op
dispatch; `weight_layout.h` is the shared weight-layout struct. The compiled `.spv` are checked
in alongside the `.comp` sources so it runs without a shader compiler.

## The one genuinely nice trick: UMA zero-copy

The RK3588 is a **unified-memory** part — CPU and GPU share the same LPDDR5. So every buffer uses
`HOST_VISIBLE | HOST_COHERENT` memory, and **the GPU reads the exact physical pages the CPU
dequantised into — no staging copy, no explicit transfer.** On a discrete GPU you'd pay a PCIe
copy; here you just hand the GPU a pointer. That's the RK3588 hardware handing you a free
optimisation, and this code takes it.

## Honest caveat: it lost to the CPU

Read [`docs/06`](../docs/06-the-graveyard.md) before you get excited. For these ops on this board,
the **Mali GPU is slower than 4×A76 NEON** — the shared bandwidth ([04](../docs/04-the-bandwidth-wall.md))
means the GPU has no memory advantage, and per-dispatch launch overhead hurts. This library is
here as a **working, reusable reference** (and for workloads where the CPU is otherwise busy — the
heterogeneous story in [08](../docs/08-vision-and-heterogeneous.md)), not as a recommendation to
run your LLM on the Mali. It's honest code with an honest result.

## Building

Needs the Vulkan SDK / `libvulkan`. The `.spv` are pre-compiled; to rebuild them from source,
`glslangValidator -V shaders/x.comp -o shaders/x.spv`. Point `GPU_SHADER_DIR` (compile-time define,
default `./shaders`) — or the runtime `dir_override` — at wherever the `.spv` live.
