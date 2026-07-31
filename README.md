# rk3588-npu-lab

> One girl and her AI friends, going crazy in a backroom, tearing a Radxa ROCK 5's
> NPU down to bare metal. Why? We're not entirely sure. This repo is the aftermath of
> deciding the vendor SDK was optional: reverse-engineered DRM ioctls, zero vendor libs,
> pure madness — with receipts.

It turns out you can drive the RK3588's neural engine yourself, straight over its DRM
ioctl interface, with **no `librknnrt.so`, no `librkllmrt.so`, nothing** from Rockchip's
userspace stack. This repo is the field notes, the tracer, the benchmarks, and the
hard-won numbers from doing exactly that — so the next person doesn't have to start from a
blank `strace` at 2am.

Everything here is about **getting the most out of the silicon**. No frameworks, no
magic, just the hardware and what it will actually do when you ask nicely (and in the
right register order).

**Just here to run a model?** Jump to [docs/07](docs/07-running-models.md) for the "run a
MoE, not a dense model — here's the recipe" short version. The current shape of the board:
**dense vision on the NPU (3× the CPU, measured), the MoE language model on the CPU, running
concurrently** — each workload on the hardware it actually wins on.

## The one-paragraph technical version

The RK3588 NPU is an **NVDLA-derived** 3-core accelerator exposed as a **DRM device**
(`/dev/dri/card1`), not the `/dev/rknpu` you'd expect. All inference is DRM ioctls
(`SUBMIT`/`ACTION`/`MEM_*`) pushing **register command streams** to a matrix engine (CNA)
that runs projections in GEMM mode and hardware-decompresses w8a8 weights on the fly. We
traced the protocol with an `LD_PRELOAD` shim, decoded the regcmd format, and rebuilt
inference from scratch on top of it. The headline lessons: **matmul batches for free up to
~32× on one core** (the CNA's spatial dimension is idle at M=1), **3-core tensor-parallel
works** by cycling `core_mask`, **pinning to the A76 cluster is +53%**, and — the theme
that governs everything — **decode is bound by LPDDR5 bandwidth (~25 GB/s, shared across
CPU/GPU/NPU), not by compute.** The NPU is usually *starved*, not busy.

## Start here

| Doc | What's in it |
|---|---|
| [`docs/00-the-hardware.md`](docs/00-the-hardware.md) | The RK3588 NPU, its NVDLA heritage, the 3 cores, and the two walls (bandwidth + IOMMU) |
| [`docs/01-the-interface.md`](docs/01-the-interface.md) | The DRM ioctl map, GEM buffers, and the per-token hot path (and why it spins) |
| [`docs/02-the-command-stream.md`](docs/02-the-command-stream.md) | Decoding register command buffers; CNA vs DPU; how to trace your own |
| [`docs/03-going-fast.md`](docs/03-going-fast.md) | The levers that worked: M-batching, TP3, A76 pinning, int8 — with numbers + method |
| [`docs/04-the-bandwidth-wall.md`](docs/04-the-bandwidth-wall.md) | The bandwidth law, the evidence, and why the answer was a second board |
| [`docs/05-kernel-and-tuning.md`](docs/05-kernel-and-tuning.md) | The kernel config that matters, and the tuning landmines (looking at you, `isolcpus`) |
| [`docs/06-the-graveyard.md`](docs/06-the-graveyard.md) | Good ideas that died, and the measurements that killed them |
| [`docs/07-running-models.md`](docs/07-running-models.md) | **Just want to run an LLM?** Why MoE beats dense, quant advice, the deploy recipe + real tok/s |
| [`docs/08-vision-and-heterogeneous.md`](docs/08-vision-and-heterogeneous.md) | Vision on the NPU (3× CPU, measured) + running it alongside the CPU LLM |
| [`docs/09-multimodal-grafting.md`](docs/09-multimodal-grafting.md) | **(exploratory)** Where this thinking leads: a CfC "connective tissue" binding frozen models into associative memory — evocation, not generation |

## What's in the box

```
rk3588-npu-lab/
├─ docs/          # the field notes (start above)
├─ tools/
│  ├─ tracing/    # npu_hook.c — the LD_PRELOAD shim that decodes NPU ioctls live
│  ├─ example/    # npu_gemm_test.c — a minimal raw-DRM CNA matmul you can run
│  └─ bench/      # the hardware benchmark harness (bandwidth, matmul, concurrency)
├─ kernel/        # config rationale + the CMDLINE_EXTEND patch (GPLv2)
└─ CREDITS.md     # the giants whose shoulders this stands on
```

## The hardware

- **Board:** Radxa ROCK 5 ITX+ · **SoC:** Rockchip RK3588
- **NPU:** 3× cores, ~2 TOPS each (~6 TOPS int8 rated), NVDLA-derived, on `/dev/dri/card1`
- **CPU:** 4× Cortex-A76 @ 2.304 GHz + 4× Cortex-A55 · **GPU:** Mali-G610
- **RAM:** LPDDR5 — ~25 GB/s effective, and *shared*. Remember that number; it wins every argument.

## The boring-but-important bit

This is **independent interoperability research**. Nothing here redistributes Rockchip's
proprietary software — no vendor `.so`, no binaries, no patched libraries. What's
documented is the *hardware interface* (an ioctl protocol and a register layout), learned
by observing a device we own, in the long tradition of writing free drivers for hardware
whose vendor stack you'd rather not depend on. Rockchip's kernel NPU driver is itself
**GPL** and public; this work builds on that and on the prior community RE credited in
[`CREDITS.md`](CREDITS.md).

"RKNPU", "RKLLM", "Rockchip", and "Radxa" are trademarks of their respective owners, used
here only to say true things about their products. This repo is **not affiliated with or
endorsed by** Rockchip or Radxa. Everything is provided as-is, for research and
interoperability, under the MIT license (see [`LICENSE`](LICENSE)).

## Standing on shoulders

None of this started from zero — see [`CREDITS.md`](CREDITS.md) for the people who mapped
this territory first: **Jasbir Matharu** (`rk3588-npu`), **allbilly** (`npu` / `ops_reg`),
**Martin Chang** (NVDLA LUT analysis), the **NVDLA** project itself, **ggml**, and the
**Panfrost/Asahi** crews whose accelerator-RE playbook we shamelessly borrowed.
