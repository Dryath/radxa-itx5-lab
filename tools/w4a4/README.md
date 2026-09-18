# W4A4 Qwen3-0.6B for the RK3588 int4 datapath

> *Context: this is the quantisation toolkit behind the symmetric-precision / per-channel-scale
> discussion in [`docs/10`](../../docs/10-the-npu-backend.md) — the NPU wants symmetric W4A4 with
> per-kernel scales, and this is the pipeline that produces it (rotate → quantise → pack → probe).
> The board-specific `run_*.py` orchestration scripts and the pre-generated binary probe fixtures
> aren't shipped; the command sequence below regenerates everything from a base model.*

A 4-bit-weight, 4-bit-activation build of Qwen3-0.6B, with the rotation baked
into the weight bank so the NPU kernel stays a pure int4 matmul — no runtime
Hadamard, no per-layer fixup op.

Started on Qwen3-1.7B (same base as the shipping int8 model, so int4-vs-int8
would have been a true A/B) and moved to 0.6B for build speed. The scripts run
either unchanged — hidden 1024 is still a power of two, so the Sylvester
Hadamard stays exact. Shapes are 0.6B: K in {1024, 3072}, N in
{1152, 2112, 3072} after padding.

Scale form is **per-kernel** (one fp32 per row, applied after the accumulate),
matching the shipping int8 bank. A hardware constraint, not a preference: the
int4 dispatch sets `datain_channel = K` and accumulates every channel into one
int32 before anything is read out, so there is no per-K-group readout and
group-32 scales are not consumable. A tile is not a scale.

## What this does and does not settle

**It does not fix the half-kernel fault.** That was already chased further
here than this artifact reaches. `verify_native_layout.c` proved
`(N/64, K/32, 64, 32)` k-inner correct at **256/256 exact** through the
vendor's own dispatch, with `io.B.dims` reporting `[1,2,64,32]`, and our
kernel fed that exact layout still failed 1/1536. Layout is settled and
eliminated; the open variable is **register configuration**. The 2912-test
single-bit sweep returning zero hits is consistent with that — and with the
fault not being a single bit anywhere in the stream.

**What it does give the register hunt** is real model data at more than one
shape. Every probe so far has been all-ones controls or single geometries.
This emits actual quantised Qwen3 weight banks at three distinct kernel counts
and two distinct K, each with exact integer goldens, so a register field that
scales with N is separable from one that scales with K by diffing rather than
by inferring from output garbage.

Formats follow the **verified** triple, not the header:

| tensor | verified form |
|---|---|
| A | normal `(M,K)`, nibble-packed row-major |
| B | native `(N/64, K/32, 64, 32)`, k-inner |
| C | int16 `(M,N)`, row-major |

The header's native A `(K/32,M,32)` and C `(N/8,M,8)` forms are emitted too,
but as alternates, clearly labelled.

## What the kit found on silicon

Running these banks on the board surfaced a second, previously unseen fault —
and it was only visible because the kit ships more than one K.

**The block-prefix fault.** At N=1152 the per-64-block correct counts are
`[32]*6` at K=512 and K=1024, but `[32,32,32,0,0,0]` at K=2048: half the
blocks stop producing output entirely. Reproduced with random data on the
board, so it is a hardware/config fault and not a property of any bank —
in particular not of this model's (collapsed) W4A4 values.

**It is fixed by HMULT**, the H-axis multiplier that already had to be 2
(dispatch `M_hw = HMULT*M`, read rows `0, HMULT, 2*HMULT, ...`). Measured
minimum HMULT:

| K | 512 | 1024 | 1536 | 1792 | 2048 | 2560 | 3072 |
|---|---|---|---|---|---|---|---|
| min HMULT | 2 | 2 | 2 | 3 | 3 | 4 | 4 |

It is the **ratio**, not absolute `Mt_pad`: at K=2048, `M=4/h4` and `M=8/h2`
give the same `Mt_pad=16`, yet the first works and the second fails.

Ruled out along the way, each by direct measurement rather than argument:

- the `entries` rule — `entries=16` at K=2048 is correct, every other value
  is worse
- the H axis being quantised to powers of two — HMULT=3 works at K=2048
- `min HMULT = max(2, K/512)` — K=1536 needs only 2
- `min HMULT = max(2, ceil(K/512)-1)` — predicts 5 at K=3072, measured 4

### The lower bound: min HMULT = ceil(K/768), CONFIRMED

Health-gated (see below), every point fits:

| K | 1024 | 1536 | 1792 | 2048 | 2304 | 2560 | 3072 |
|---|---|---|---|---|---|---|---|
| observed min | 2 | 2 | 3 | 3 | 3 | 4* | 4 |
| `ceil(K/768)` | 2 | 2 | 3 | 3 | 3 | 4 | 4 |

(*h3 at 2560 is marginal, 2/3 — right at the boundary.) Reads as coverage:
~768 channels per H row, `ceil(K/768)` rows to cover K.

This rule was retracted once and then reinstated. The retraction was correct
at the time — it had been fitted to single readings of a system that turned
out to be non-deterministic. The readings were noise; the rule was not.

### The health gate — why every earlier reading was unreliable

**NPU state crosses process boundaries.** A failing shape leaves the device
degraded, and the next run of a *different* shape reads wrong — in both
directions. Fresh processes do not isolate it, and repeat counts cannot
detect it because every rep is equally contaminated.

Two conclusions flipped once measurement was gated on a proven-healthy
device:

- K=3072 h6 read as passing ungated -> fails every rep from healthy
- K=512 h4 read as failing in-process -> clean in isolation

So: run a known-good shape, require 3/3 cores, and only then measure. Repeat
after, because a config that fails degrades the device for whatever runs
next.

### There is also an upper bound, and it tightens with K

| K | result |
|---|---|
| 2048 | h3..h6 all pass |
| 2304 | h3, h4, h6 pass; h5 marginal |
| 3072 | **only h4** — h5, h6, h7, h8 all fail |

That is why the table looked non-monotonic. h8 failing kills "power of two";
h6 passing at 2304 kills parity; h6 exceeding h5 in feature bytes kills a
capacity ceiling. Two bounds, and the upper one is not yet explained.

Failing cores are always 1 and 2, never core 0 — structural, not a race.

Monotonicity above it does not. With repeats:

    K=3072  h4: 3/3 3/3 3/3   solid
            h5: 1/3 1/3 1/3   reproducibly bad
            h6: 2/3 3/3 2/3   flaky

h5 failing while h4 and h6 pass kills every monotonic coverage rule.

**And then the failing-core log reclassified most of it.** K=512 h4 fails on
all three cores on the FIRST run of a process, then passes on every later
run. That is warm-up / state carryover: because each rep was a fresh process,
one bad first dispatch reads as 2/3. So

- K=3072 h5 = 0/3 every rep -> fails warm too, REAL
- K=3072 h6 = 1/3, 2/3, 2/3 -> the fresh-process warm-up signature

If h6 is clean once warm, K=3072 is h4 ok / h5 fail / h6 ok: parity above the
coverage bound, not a capacity ceiling. That also retires the byte-count
puzzle, since h6 is larger than the h5 that reproducibly fails.

**Every HMULT point measured as a fresh process may carry this artifact**,
including the ones the fit was derived from. Re-measure with one warm-up
dispatch discarded before trusting any of it — and the same question applies
to single-reading conclusions elsewhere in the int4 work.

## Telling staleness apart from silence

`probe_flaky.py` emits N generations of one shape with different goldens:

    C[m][n] == (n - N//2) + TAG*(g+1)

### The probe data itself was a variable — and a fault

`encode_sum` builds each column as a prefix of +-1 followed by zeros, to make
the golden readable. That makes the bank pathological: 86% of each column is
one contiguous zero block, and target 0 produces an entirely zero kernel.

On silicon that provoked a **data-dependent fault**. Same shape, same config,
byte-identical packing (the board's own packer reproduced these bytes exactly),
different values, different behaviour:

    board's random weights, M=2 K=512 N=1152 h2 : [32]*6 all cores       PASS
    sparse probe weights,   same shape & config : core2 [32,32,32,32,0,0] FAIL

Layout, indexing, M and HMULT were all eliminated on the board.

**Both obvious mechanisms are now refuted.**

- *Feature zeros* (`fc_skip_en`, "when one pixel feature data is 0 the
  corresponding weight data is not fetched"): padding value 0, 1, 7 and
  random all give identical block patterns. Dead.
- *Weight sparsity*: dense is **worse**, not better. Sparse loses core 2's
  last two blocks; dense loses cores 1 and 2. The pathological probe data was
  partly masking a wider fault rather than causing one.

A third failure mode also fell out: dense + random padding gives core 1
`[28,28,28,29,0,0]` -- partial lane corruption inside a surviving block,
distinct from both "clean" and "block absent". Padding content does not
decide whether a block survives, but can corrupt lanes within one that does.

The fault is real, data-sensitive, and explained by neither mechanism. It
reproduces at K=512 on kit data where random weights pass, and the dense case
matches the block-prefix pattern seen on the real Qwen banks at K=2048 --
so probably one fault, not two.

So every probe now ships in **two encodings**, identical in shape, layout and
readback, differing only in density:

| encoding | all-zero kernels | median longest zero-run | column sums |
|---|---|---|---|
| `sparse` | 1 | 1757 / 2048 | exact |
| `dense` | 0 | 3 / 2048 | exact |

`--encoding dense` (default) is representative; `--encoding sparse` is the one
that exposes the fault. Run both — density is the variable, and a result from
one alone means little.

The uncomfortable general point: every int4 conclusion measured on synthetic
weights may not hold on real ones. Same class of blindness as single-K and
single-N.

**Wrap the health gate around it.** A degraded device and a stale buffer
produce the same wrong lanes; only the bracketing health runs tell them
apart. Each kit ships a `health/` probe (known-good shape, untagged golden).
`classify_flaky.py` takes `health_pre=` and `health_post=`, refuses to
interpret anything if `health_pre` is degraded (exit 2), and flags when the
measured config degraded the device on the way out.

The gate reads the **per-64-block pattern, not a lane total**. A kernel with
the half-kernel fault can only ever get 32 of every 64 lanes, so gating on
all-N lanes assumes a working kernel and could never pass. Uniform 64/block is
fully healthy, uniform 32/block is healthy given the known fault, anything
ragged is a degraded device.

For the third failure mode the classifier splits each block: lanes 0-31 are
the ones that should survive the half-kernel fault, 32-63 are expected dead.
A block reading 28 means four lanes that should have survived did not, and
the classifier reports **where inside the block** they sit -- clustered
positions mean a fixed sub-lane group is dropping, scattered means
corruption rather than geometry.

Pre-fill the output with `POISON_fill.i16.bin` (0xAAAA, unproducible by any
int4 dot product), dispatch each generation into the same buffer without
clearing, then re-dispatch gen0 LAST. Every lane self-classifies:

| lane reads | meaning |
|---|---|
| poison | never written — a geometry/HMULT question |
| zero | covered, nothing fetched |
| another generation's golden | **stale** — buffer reused before the previous dispatch retired |
| correct | fine |
| anything else | misaddressed or garbage |

Staleness is the one hypothesis a repeat-count experiment can never confirm,
because a stale value from an identical previous run looks correct. Different
goldens per run breaks that tie. `classify_flaky.py` also compares gen0-first
against gen0-last in the same process and names the verdict: more correct
lanes on the last dispatch is warm-up; equal and failing is a real fault.

Also worth logging: **which core fails.** Always-core-N is structural; moving
between cores is a race.

## Two findings that are new here

**K=32 probes are structurally blind to N-tiling.** With a single K-tile,
`(N/64,K/32,64,32)` and `(N/32,K/32,32,32)` serialise to *identical bytes* —
there is no second k-block to interleave the n-blocks with. Any probe run at
K=32 cannot distinguish 64-kernel from 32-kernel grouping no matter what else
is varied. `test_layouts.py` asserts both the aliasing at K=32 and the
divergence at K=64. Probes here use K ∈ {64,128}; one K=32 probe is kept as an
explicit negative control.

**int16 output has a real accumulator ceiling.** If int4 output is genuinely
int16, K is bounded: worst case is `64*K` (8×8 per term), so K=2048 can reach
131072 and K=6144 393216 — 4× and 12× past int16. Measured on this model's
actual quantised weights, headroom is **7.8×–15×**, so it does not overflow at
any shape here. But that margin is data-dependent, not structural. A kernel
that trusts int16 output needs a K-split or a saturation check before it meets
real activations. `verify_kit.py` prints measured max |C| and worst case per
shape.

## The model build

```
rotate.py            fuse RMSNorm scales into the linears, bake R1 (hidden
                     2048) and R2 (head 128) randomized Hadamards into the
                     weights, untie lm_head from embed_tokens
verify_rotation.py   prove rotated == original as a function (also used to
                     gate smoothing, which must be equally exact)
smooth.py            offline per-channel outlier migration for down_proj and
                     o_proj -- the layers rotation cannot reach
quantize.py          GPTQ int4, group 32 along K, act_order OFF
eval_ppl.py          wikitext2 ppl for fp16 / W4A16 / W4A4
pack_rk3588.py       emit banks + goldens; --all writes the deployable bank
classify_output.py   board-side: name the layout the hardware actually used
verify_kit.py        integrity-check an emitted kit before trusting it
test_layouts.py      self-checks on the layout primitives
probe_ksweep.py      K-sweep kernel-id probes (HMULT-independent banks)
probe_flaky.py       generation-tagged probes: staleness vs silence, warm-up
classify_flaky.py    board-side classifier for the above
```

Pre-generated probe sets: `ksweep/` (12 K values at N=1152) and
`flaky_K{512,1024,2048,3072}/`. Both regenerable from the scripts.

### Rotation

Bare RMSNorm commutes with an orthonormal Q exactly, because
`mean((xQ)²) == mean(x²)`. An elementwise scale does not, so the scales are
fused into the consuming linear first and set to 1. Then, with `y = x @ W.T`
and W stored `[out, in]`:

- readers of the residual (q, k, v, gate, up): `W ← W @ Q`
- writers to the residual (o, down): `W ← Qᵀ @ W`
- `embed_tokens ← E @ Q`
- `lm_head ← (E * g_final) @ Q` — differs from embed, so the weights **must**
  be untied

R2 is a per-head Hadamard on head_dim=128, fused into v_proj's output rows and
o_proj's input columns. It is free because attention output `softmax(QKᵀ)V` is
linear in V. `q_norm`/`k_norm` act on head_dim before RoPE and are untouched.

R3/R4 (online rotations) are excluded by choice — they would cost a runtime
Hadamard per layer.

### Why R4 cannot be made offline, and what replaces it

`down_proj`'s input is the elementwise product `silu(gate) * up`. A rotation
does not commute through an elementwise product, so R4 is inherently a runtime
op — no algebra folds it into the weights.

Division by a per-channel vector *does* commute through an elementwise
product. So the same outlier problem has an offline answer (`smooth.py`):

```
down(x) = ((x / s) @ (Wd * s).T)            x = silu(g) * u
        = (silu(g) * (u / s)) @ (Wd * s).T
```

`1/s` folds into `up_proj`'s rows, `s` into `down_proj`'s columns, with
`s_j = max|X_j|^a / max|W_:,j|^(1-a)`. The same trick handles `o_proj` by
folding into `v_proj`'s rows — under GQA the scale must be shared across the
query heads mapping to each KV head, so it is reduced with a max over the
sharing group. Runtime cost: none. The kernel stays a pure int4 matmul.

Measured at alpha=0.65 on Qwen3-0.6B, exactly equivalent in fp16 (100% top-1
agreement against the unsmoothed model):

| target | outlier ratio before | after |
|---|---|---|
| `down_proj` | 29.2 | 2.5 |
| `o_proj` | 2.7 | 1.6 |

### Why this was necessary

Without smoothing, true W4A4 collapses. Qwen3-0.6B, wikitext2:

| config | ppl |
|---|---|
| W16A16 (rotated fp16) | 19.55 |
| W4A16 group-32 | 21.91 |
| W4A16 per-kernel (shipping form) | 24.14 |
| W4A8 group-32 | 22.17 |
| **W4A4** | **199.99** |
| W4A4 + `o_proj`@A8 | 157.15 |
| W4A4 + `down_proj`@A8 | 33.55 |
| W4A4 + both@A8 | 29.99 |

Activations at 8 bits are nearly free (+0.26). At 4 bits the model breaks, and
`down_proj` alone accounts for 200 -> 33.5. Fifteen quantisation levels cannot
span a 29x per-channel dynamic range — which is exactly what smoothing removes.

The A8 rows are diagnostics only. The target is a genuine W4A4: 4-bit
activations on every linear, no fallback.

### What per-kernel scales cost

Per-kernel is not a choice, it is the consumable form: the int4 dispatch sets
`datain_channel = K` and accumulates every channel into one int32 before
anything is read out, so there is no per-K-group readout. Measured cost:

| scale form | scales per kernel | W4A16 ppl |
|---|---|---|
| group 32 along K | 32 (K=1024) / 96 (K=3072) | 21.91 |
| per-kernel | 1 | 24.14 |

+2.22 ppl for a 32x to 96x coarsening. That is mild for the size of the
change, and consistent with the claim that the baked-in rotation already
flattened the per-channel ranges that normally force group scales -- the
lab's own retrieval measurement saw a shared orthogonal R lift int4 recall
from ~55% to 99% top-5 for the same reason.

Not proven, though: there is no unrotated per-kernel control here, so the
attribution to rotation is suggestive rather than demonstrated. The clean
test is per-kernel GPTQ on an unrotated model -- one quantize plus one eval.

Group size 32 along K is not an accuracy preference — it is the hardware's
K/32 tile. One scale per (kernel, K-tile) lines the dequant term up with the
1024-byte weight atom. For the same reason `act_order` is **off**: it helps
perplexity but scrambles which K indices share a group, and groups have to
stay contiguous in K.

### Padding

int4 N is 64-byte aligned; the board additionally wants N a multiple of 48.
`lcm(48,64) = 192`, so banks are padded to a multiple of 192 with zero
kernels. `N_zero_kernels` is recorded in every `meta.json`.

## Probe kit

```
w4a4_probe/
  manifest.json
  L00.k_proj/      K=2048  N=1024 -> 1152
  L00.q_proj/      K=2048  N=2048 -> 2112
  L00.gate_proj/   K=2048  N=6144 -> 6144
  L00.down_proj/   K=6144  N=2048 -> 2112     <- different K
  probe.kernelid_K{64,128}_N{192,256,384}/
  probe.kernelid_K32_N192/                    <- blind control
```

Per shape: `B_nat_n64` (verified) plus `B_nat_n32`, `B_nat_n16`,
`B_nat_n64_kmaj` and `B_nat_n64.i4hi` as alternates; `scales.f32`;
activations at M=1 and M=8; goldens as exact integer products, no float
tolerance.

### kernel-id probe

Built so the correct answer is `C[m][n] == n - N//2`, for every m. Add `N//2`
back to each output lane and you read off which kernel the hardware actually
put there.

```
python3 classify_output.py w4a4_probe/probe.kernelid_K64_N192 hw_out.bin
```

| recovered indices | meaning |
|---|---|
| `0,1,2,3,...` | correct |
| `0..31,0..31,...` | 32 kernels supplied where the atom wants 64 |
| `0,2,4,...` | nibble order swapped, or int8 element width assumed |
| `0..63` then garbage | one 1024-byte weight fetch landed, rest stale |

Both rows must be identical (A is all-ones on both). If they differ, the M
axis is wrong before the N axis is worth reading.

## Reproducing

Needs torch + transformers + safetensors + datasets, and a CUDA card for
GPTQ (built on a 6 GiB GTX 1660 SUPER; the model lives on CPU and one decoder
layer at a time goes to the GPU).

```bash
python rotate.py          --src Qwen3-1.7B     --dst Qwen3-1.7B-rot
python verify_rotation.py --src Qwen3-1.7B     --rot Qwen3-1.7B-rot
python quantize.py        --model Qwen3-1.7B-rot --out Qwen3-1.7B-w4 \
                          --nsamples 128 --seqlen 1024
python eval_ppl.py        --model Qwen3-1.7B-w4  --act 4
python pack_rk3588.py     --model Qwen3-1.7B-rot --out w4a4_probe \
                          --gptq Qwen3-1.7B-w4/int4_weights.npz --all
python verify_kit.py      w4a4_probe
```

`--all` additionally writes `bank.i4.bin` (every linear in the verified
layout, one flat file), `bank_scales.f32.bin` and `bank_index.json` — the
deployable artifact. `embed_tokens` stays fp16; it is a gather, not a matmul.

## Build cost, measured

On a GTX 1660 SUPER (6 GiB, WSL2 — note WSL2 exposes only one GPU even when
the host has two):

| model | GPTQ per layer | 28 layers |
|---|---|---|
| Qwen3-0.6B, 64×512 | ~11 s (H 2.8 / quant 4.2 / out 2.3) | ~5 min |
| Qwen3-1.7B, 128×1024 | ~60 s projected | ~30–45 min |

**Tokenising the calibration set dominated the first run** — 2.5M tokens of
wikitext2 train, single-threaded, ~40 minutes, with no output the whole time.
That looked exactly like a hung GPTQ and was not. `get_calib` now caches the
id tensor to `/tmp/wikitext2_train_ids.pt`, and `quantize.py` prints per-stage
timing (`H` / `quant` / `out`) so cost is attributable rather than guessed.

## Verified numbers

- R1 orthonormality residual `1.1e-16`
- Qwen3-1.7B rotated vs original: top-1 **100%**, relative max diff `1.0e-3`
- Qwen3-0.6B rotated vs original: top-1 **100%**, relative max diff `1.6e-3`
  (both are fp16 storage rounding, not rotation error)
- probe kit integrity: all layout inversions and integer goldens pass

Both model sizes build from these scripts unchanged — 0.6B has hidden 1024,
still a power of two, so the Sylvester Hadamard stays exact.
