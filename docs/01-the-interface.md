# 01 — The interface (it's DRM all the way down)

The first surprise: there is no special NPU syscall API. You open a DRM render node and
send ioctls, exactly like you would to a GPU. The vendor userspace libraries are just a
(large, closed) wrapper around this. We threw the wrapper away.

## The device

RKLLM/RKNN open **`/dev/dri/card0` and `/dev/dri/card1`**; `card1` is the NPU. All buffers
go through **DRM GEM** (this requires the kernel built with `ROCKCHIP_RKNPU_DRM_GEM=y` — see
[05](05-kernel-and-tuning.md); get this wrong and every allocation fails with `EINVAL`).
GEM mappings show up in `/proc/self/maps` as `rw-s` regions with offset 0 — handy for
correlating a buffer with the regcmd IOVAs that reference it.

## The ioctl map

DRM type `0x64` (`'d'`), `nr ≥ 0x40`:

| nr | Name | Struct size | Calls / inference | What it does |
|---:|---|---:|---:|---|
| `0x40` | **ACTION** | 8 B | ~1,000,000 | get/set freq/volt/bandwidth — **and completion poll** |
| `0x41` | **SUBMIT** | 104 B | ~1,000,000 | submit a task batch to a core |
| `0x42` | MEM_CREATE | 48 B | 16 / session | allocate a GEM buffer |
| `0x43` | MEM_MAP | 16 B | 16 / session | map it |
| `0x44` | MEM_DESTROY | 16 B | 16 / session | free it |
| `0x45` | **MEM_SYNC** | 32 B | ~2,000,000 | DMA cache flush (CPU↔NPU) |

(Names/values as seen in the kernel UAPI header, e.g. `RKNPU_SUBMIT=0x01`,
`RKNPU_ACTION=0x00` at the driver level; the DRM `nr` adds the `0x40` base.)

## The per-token hot path

For every token, per NPU op, the sequence is:

```
MEM_SYNC (flush inputs → NPU)
SUBMIT   (kick off the CNA task)
ACTION   (GET_DRV_VERSION — but really: block until the job completes)
MEM_SYNC (flush outputs → CPU)
```

The important word is **block**. `ACTION` here is **not** an async pipeline kick — it's a
**synchronous completion poll**. The counts give it away: in one trace, **63,209 ACTION
calls vs 63,207 SUBMIT** — one poll per submit. The CPU issues the work and then *spins
waiting for it*.

## Why this is the whole ballgame

Measured on a 2048×2048 op: wall time **~222 µs**, of which the **ioctl round-trip is ~164
µs** and the actual useful compute is **~1.4 µs**. That's **0.63% utilization**. Across
decode, the ioctl/completion path is **70–82% of wall time**.

So the NPU spends most of its life finished-and-idle while the CPU is stuck in a blocking
wait, coming out of deep idle, eating wakeup latency (p99 is 2–5× p50). This single
observation sent us chasing two things:

1. **Do more work per submit** — batch matmuls so each expensive round-trip carries 32×
   the payload. That's [M-batching](03-going-fast.md), and it's the big win.
2. **Free the CPU while the NPU works** — a non-blocking `FENCE_OUT` submit hands the CPU
   back in ~9 µs instead of blocking for the ~280 µs the NPU takes. It works on a `FENCE=y`
   kernel (measured) — but plot twist: measuring it showed the warm dispatch is
   *compute-bound*, so it buys **concurrency, not decode speed**. The autopsy is in
   [05](05-kernel-and-tuning.md).

---

Next: [02 — The command stream](02-the-command-stream.md) — what's actually inside a SUBMIT.
