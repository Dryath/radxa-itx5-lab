# npu_hook.c — the NPU ioctl tracer

An `LD_PRELOAD` shim that sits under any process driving the RK3588 NPU and decodes the DRM
ioctls live. This is the tool that produced most of the findings in [`docs/`](../../docs/).

## What it shows

- One decoded line per `SUBMIT`: `hw_us  core  tasks  ena  cfg  unit  weight_base`
  (`ena` = `enable_mask`, `cfg` = `regcfg_amount` — the op fingerprint from
  [docs/02](../../docs/02-the-command-stream.md)).
- Optional dumps of the raw register command buffers for CNA / DPU ops.
- Optional timing hooks for round-trip / TTFT / tok-s.

## You need one header

The shim `#include`s **`rknpu_ioctl.h`** — the kernel UAPI header for the NPU driver. It is
**not vendored here** (it belongs to Rockchip's GPL `rknpu` driver). Grab it from the
rknpu kernel driver source and drop it next to `npu_hook.c` (or point `-I` at it).

## Build

```bash
gcc -shared -fPIC -O2 -o npu_hook.so npu_hook.c -I. -ldl -Wl,-soname,npu_hook.so
```

## Run

```bash
LD_PRELOAD=./npu_hook.so  <your NPU program>
```

Environment knobs (dump control):

| Var | Default | Meaning |
|---|---|---|
| `NPU_HOOK_DUMP_CNA` | 1 | how many CNA regcmd buffers to dump |
| `NPU_HOOK_DUMP_DPU` | 1 | how many DPU regcmd buffers to dump |
| `NPU_HOOK_DUMP_CNA_SKIP` | 0 | how many CNA dumps to skip first |
| `NPU_HOOK_DUMP_DPU_SKIP` | 0 | how many DPU dumps to skip first |

Writes are serialized with `flockfile()` so concurrent submits don't interleave.

> This only observes ioctls to a device you own. It doesn't modify, patch, or redistribute
> any vendor software.
