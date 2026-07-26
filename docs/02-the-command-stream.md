# 02 — The command stream (reading the NPU's mind)

A `SUBMIT` points at a **register command buffer** — the actual program the NPU core runs.
Decode this and you can synthesize your own operations from scratch. This is the part that
felt like magic the first time it worked.

## The regcmd format

The command buffer is an array of **`{u32 packed_addr, u32 value}`** pairs. For each pair:

- `bits[15:0]` of `packed_addr` = the **register offset**
- `(packed_addr & 0xFFFF) >> 12` = the **block selector** — which sub-unit the write targets:
  `PC / CNA / CORE / DPU / DPU_RDMA / PPU / DDMA / SDMA / GLOBAL`

At the task level, ops are packed as `(op << 48) | (value << 16) | reg`, little-endian.

A practical gotcha that cost real time: the `op_idx` field the SDK "should" set to identify
an op type **is always 0** — the vendor code never populates it. Don't discriminate on it.
Use the pair **`(enable_mask, regcfg_amount)`** instead — that actually tells you what kind
of op you're looking at.

## Two fingerprints worth memorizing

| Op | `enable_mask` | `regcfg_amount` | ~time |
|---|---|---|---|
| **CNA GEMM** (a projection) | `0x0d` | 108 | ~960 µs |
| **standalone DPU** | `0x18` | 69 | ~30 µs |

CNA is the workhorse. And here's the elegant part: the CNA **hardware-decompresses w8a8
weights on the fly** (`dcomp_ctrl` / `dcomp_regnum` / `dcomp_addr0`) as they stream from the
weight buffer — the CPU never touches the decompression. Weights live at a CNA-internal
IOVA base you'll see everywhere: **`wt_iova = 0x02000000 + tensor_byte_offset`**.

## You can read the model architecture straight off the silicon

Because the regcmd carries the shapes, you can reconstruct a model's architecture purely by
watching it run — no config file needed:

- `datain_channel = 4095` → hidden size **4096**
- 252 CNA ops per forward pass ÷ 7 projections per layer → **36 layers**
- → (in one captured case) a Llama-3-8B-shaped model, cross-checked on a 0.5B where
  `datain_channel = 895` → 896, matching a Qwen2.5-0.5B.

The `lm_head` is recognizable too: it's the CNA op with `tasks = 21` (the vocab tiled into
21 kernels), always the last op of a pass, and on an 8B it's **~8% of total inference time**.

## 3-core tensor parallelism

Splitting an op across all three cores is done by issuing **three single-core SUBMITs with
`core_mask` cycling `0x1 → 0x2 → 0x4`**; they run in parallel and wall-time ≈ one core's
time. The slices aren't perfectly even (e.g. a 896-wide output splits `304 + 304 + 288`), so
pad `N` to a multiple of 48 to keep the cores balanced. More in [03](03-going-fast.md).

## The DPU plot twist (a lesson in staying honest)

We initially believed the DPU did the interesting attention work — K-RoPE and a
flash-attention softmax via a lookup table. It was a clean, satisfying theory. It was also
**wrong**, and we're leaving the gravestone up as a reminder.

Probing the standalone DPU op (`enable_mask = 0x18`) with a linear pattern showed it's a
**strided copy** — emit 32 elements, skip 32, verbatim, no LUT, no arithmetic — a layout
transformer, not an attention primitive. A second pass over 60 generation-phase dumps
confirmed they all share **one identity int32→fp16 post-process pipeline**. The real
conclusion: **attention runs on the CPU** (which is exactly what the vendor runtime does
too). The DPU standalone op only appears in *prefill*, twice per KV-head per layer, on the
two KV-holding cores.

Moral: a beautiful theory that survives three days and dies to one targeted probe is the RE
process working correctly. Trust the probe, not the story.

## How to trace your own

`tools/tracing/npu_hook.c` is an `LD_PRELOAD` shim that intercepts the DRM ioctls (and
optionally the vendor API for timing), filters to the NPU render fd, and prints a decoded
line per SUBMIT: `hw_us  core  tasks  ena  cfg  unit  weight_base`. Point it at any process
that drives the NPU and watch the command stream go by. Build/run notes are in
[`tools/tracing/`](../tools/tracing/). Everything above was learned with it.

---

Next: [03 — Going fast](03-going-fast.md) — the levers that actually moved the needle.
