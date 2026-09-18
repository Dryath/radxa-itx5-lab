# W4A4 Qwen3-1.7B for the RK3588 int4 datapath

A 4-bit-weight, 4-bit-activation build of Qwen3-1.7B, with the rotation baked
into the weight bank so the NPU kernel stays a pure int4 matmul — no runtime
Hadamard, no per-layer fixup op.

Qwen3-1.7B specifically because it is the same base as the shipping int8
the shipping int8 model (TP3, 10.9 tok/s). Identical tokenizer, identical shapes, identical
plumbing, so int4-vs-int8 is a true A/B and not a model comparison wearing a
precision costume.

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
verify_rotation.py   prove rotated == original as a function
quantize.py          GPTQ int4, group 32 along K, act_order OFF
eval_ppl.py          wikitext2 ppl for fp16 / W4A16 / W4A4
pack_rk3588.py       emit banks + goldens; --all writes the deployable bank
classify_output.py   board-side: name the layout the hardware actually used
verify_kit.py        integrity-check an emitted kit before trusting it
test_layouts.py      self-checks on the layout primitives
```

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
Hadamard per layer. The cost is real and shows up in every GPTQ report:
`down_proj` is the worst layer in the model, because its input still carries
the raw SwiGLU outliers. If W4A4 there is unacceptable, run that one layer at
A8 before reaching for a runtime rotation.

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
