# Credits & prior art

This project did not spring from nowhere. Reverse-engineering the RK3588 NPU is a
community relay race, and this repo is one leg of it. Everyone below got here first or
built something we leaned on — thank you.

*If you're mentioned here and want your attribution changed, expanded, or removed, open an
issue and it's done.*

## The people who mapped the NPU first

- **Jasbir Matharu** — [`github.com/mtx512/rk3588-npu`](https://github.com/mtx512/rk3588-npu)
  (GPLv3), and the write-ups that cracked this open:
  [RK3588 reverse-engineering the RKNN (Feb 2024)](http://jas-hacks.blogspot.com/2024/02/rk3588-reverse-engineering-rknn.html)
  and [running llama2.c on the NPU (May 2024)](http://jas-hacks.blogspot.com/2024/05/rk3588-reverse-engineering-rknn-running.html).
  The working raw-DRM fp16 matmul reference and the NVDLA-similarity insight. If you only
  read one prior source, read his.
- **allbilly** — [`github.com/allbilly/npu`](https://github.com/allbilly/npu) /
  [deepwiki](https://deepwiki.com/allbilly/npu). The `ops_reg` dynamic register-command
  toolkit (DynamicArray-based regcmd, N aligned to 32) and the DPU LUT op captures. Where
  a lot of the per-op register understanding comes from.
- **Martin Chang** (© 2020–2024) — the NVDLA LUT analysis that explains how the RK
  activation LUTs (`ops_rknn/act`) mirror the NVDLA two-table (X/Y, LE/LO) SDP layout, and
  how the hardware auto-increments LUT entry addresses. The Rosetta stone for the DPU
  activation path.
- **liej6799** — [`github.com/liej6799/rk3588`](https://github.com/liej6799/rk3588),
  another community RK3588 NPU effort referenced along the way.

## The upstream hardware & software this is built on

- **NVDLA / NVIDIA** — [`github.com/nvdla/sw`](https://github.com/nvdla/sw) and the
  [NVDLA Hardware Manual](https://nvdla.org/hw/v1/ias/lut-programming.html). The RK3588 NPU
  is an NVDLA derivative; the cube types, SDP/CDP LUT programming, and index-generation
  math all trace back here. Enormous amounts of "why does the register do that" are
  answered by reading NVDLA docs.
- **Rockchip** — the [RKLLM SDK](https://github.com/airockchip/rknn-llm), the **GPL
  `rknpu` kernel driver** (the legal and technical basis for all of this — it's open, go
  read it), and the RK3588 TRM. The trademarks are theirs; the silicon is genuinely good.
- **Georgi Gerganov & the ggml / llama.cpp community** —
  [`github.com/ggerganov/ggml`](https://github.com/ggerganov/ggml). GGUF as the model
  format and ggml as the graph substrate that the from-scratch engine work plugged into.
- **Alyssa Rosenzweig & the Panfrost / Asahi Linux crews** — not RK3588-specific, but the
  *methodology* for reverse-engineering an undocumented accelerator by watching it work
  ([asahi GPU series](https://alyssarosenzweig.ca/blog/asahi-gpu-part-n.html)) is the
  playbook this followed. Matharu's `rk3588-npu` cites Panfrost as inspiration too — it's
  turtles all the way down.
- **Mesa / Panfrost** ([freedesktop](https://gitlab.freedesktop.org/mesa/mesa)) — the Mali
  G610 userspace, relevant to the GPU-bandwidth benchmarking here.

## On borrowed headers

The examples in `tools/` reference reverse-engineered NPU register/hardware headers
(`npu_cna.h`, `npu_dpu.h`, `npu_hw.h`, `rknpu-ioctl.h`) whose lineage runs through the
Matharu and allbilly projects and Rockchip's GPL driver UAPI. **They are not vendored
here** — grab them from those upstreams (linked above). This keeps everyone's licenses
intact and the credit where it belongs.

## And the AI friends

Built with a lot of help from AI pair-programmers along the way — for tracing, decoding
register dumps, and turning a pile of 2am notes into something a human can actually read.
