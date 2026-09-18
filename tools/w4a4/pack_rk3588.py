"""Step 3 -- emit int4 weight banks in every candidate RK3588 layout.

The vendor documents an int4 B native layout of (N/64, K/32, 64, 32) but
librknnrt rejects every int4 matmul type ("Unsupport type bits 0"), so that
layout has never executed on silicon. It is a hypothesis, not ground truth.

So we do not pick one. Each shape is emitted in ALL candidate layouts over
identical logical data, with one golden output per shape. The bare-metal
kernel runs them and the hardware says which layout it wanted.

Why n=64 is the leading hypothesis
----------------------------------
    int4  (N/64, K/32, 64, 32)  -> 64*32 nibbles = 1024 bytes / tile
    int8  (N/32, K/32, 32, 32)  -> 32*32 bytes   = 1024 bytes / tile
    fp16  (N/16, K/32, 16, 32)  -> 16*32*2 bytes = 1024 bytes / tile

The weight-fetch atom is a constant 1024 bytes across every precision; only
the kernel count inside it changes. A bank built on the int8 grouping puts
32 kernels where the int4 atom expects 64 -- which is exactly the reported
"hardware takes the first 32 kernels of every 64-kernel block".

Shapes emitted give three distinct kernel counts and two distinct K, so a
field that scales with N is separable from one that scales with K.

Usage:
    python pack_rk3588.py --model Qwen3-1.7B-rot --out w4a4_probe
"""

import argparse, json, os, sys
import numpy as np
# NOTE: safetensors is imported lazily inside the model-reading helpers only. Generating
# probe kits needs no model, and requiring it here broke probe generation on the board.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (quant_int4_groupwise, dequant_int4_groupwise, pack_nibbles,
                    a_native, b_native, c_native, pad_to)

KTILE = 32          # HARDWARE weight tile along K. Fixed by the silicon.
GROUP = 32          # SCALE granularity along K. Default; --group 0 means one
                    # scale per kernel, which is all a kernel that accumulates
                    # the whole of K in a single pass can actually apply.
                    # KTILE and GROUP are independent -- a tile is not a scale.
NTILE_CANDIDATES = [64, 32, 16]
LAYERS = [
    ("model.layers.0.self_attn.k_proj.weight",  "L00.k_proj"),
    ("model.layers.0.self_attn.q_proj.weight",  "L00.q_proj"),
    ("model.layers.0.mlp.gate_proj.weight",     "L00.gate_proj"),
    ("model.layers.0.mlp.down_proj.weight",     "L00.down_proj"),
]


def w(path, arr):
    arr.tofile(path)
    return os.path.basename(path), int(arr.nbytes)


def emit_shape(outdir, name, Bq, scale, n_orig, n_mult, ms=(1, 8)):
    """Bq: (K,N) int8 in [-8,7]. scale: (N, K/GROUP) f32."""
    os.makedirs(outdir, exist_ok=True)
    K, N = Bq.shape
    files = {}

    files["B_normal.i4lo.bin"] = w(f"{outdir}/B_normal.i4lo.bin", pack_nibbles(Bq))[1]

    for nt in NTILE_CANDIDATES:
        if N % nt:
            continue
        nat = b_native(Bq, ntile=nt, ktile=KTILE)
        files[f"B_nat_n{nt}.i4lo.bin"] = w(f"{outdir}/B_nat_n{nt}.i4lo.bin",
                                           pack_nibbles(nat))[1]
        if nt == 64:   # only the leading hypothesis gets the extra variants
            files["B_nat_n64.i4hi.bin"] = w(f"{outdir}/B_nat_n64.i4hi.bin",
                                            pack_nibbles(nat, hi_first=True))[1]
            km = b_native(Bq, ntile=nt, ktile=KTILE, k_major=True)
            files["B_nat_n64_kmaj.i4lo.bin"] = w(f"{outdir}/B_nat_n64_kmaj.i4lo.bin",
                                                 pack_nibbles(km))[1]

    files["scales.f32.bin"] = w(f"{outdir}/scales.f32.bin",
                                np.ascontiguousarray(scale, dtype=np.float32))[1]

    rng = np.random.default_rng(0xA4A4)
    for M in ms:
        A = rng.integers(-8, 8, size=(M, K)).astype(np.int8)
        C = A.astype(np.int32) @ Bq.astype(np.int32)          # exact integer golden
        # VERIFIED pair (verify_native_layout.c, 256/256 through vendor dispatch):
        #   A = normal (M,K) nibble-packed row-major
        #   C = int16  (M,N) row-major
        # The header's native A/C forms are kept only as alternates.
        files[f"A_m{M}.i4lo.bin"] = w(f"{outdir}/A_m{M}.i4lo.bin", pack_nibbles(A))[1]
        files[f"C_m{M}_normal.i16.bin"] = w(f"{outdir}/C_m{M}_normal.i16.bin",
                                            C.astype(np.int16))[1]
        files[f"A_m{M}_nat_k32.i4lo.bin"] = w(f"{outdir}/A_m{M}_nat_k32.i4lo.bin",
                                              pack_nibbles(a_native(A, 32)))[1]
        files[f"C_m{M}_normal.i32.bin"] = w(f"{outdir}/C_m{M}_normal.i32.bin", C)[1]
        files[f"C_m{M}_nat_n8.i16.bin"] = w(f"{outdir}/C_m{M}_nat_n8.i16.bin",
                                            c_native(C, 8).astype(np.int16))[1]

    meta = {
        "name": name, "M_variants": list(ms), "K": int(K), "N": int(N),
        "N_original": int(n_orig), "N_pad_multiple": n_mult,
        "N_zero_kernels": int(N - n_orig),
        "group_size": GROUP or int(K),
        "scale_form": "per-kernel" if not GROUP else f"per {GROUP} K",
        "int4_range": [-8, 7], "nibble_order_default": "low_nibble_is_even_index",
        "derived_registers": {
            "datain_channel": int(K),
            "weight_kernels": int(N),
            "weight_bytes_per_kernel": int(K // 2),
            "b_tile_bytes_all_precisions": 1024,
            "b_tiles_n64": [int(N // 64), int(K // KTILE)] if N % 64 == 0 else None,
            "b_tiles_n32_int8_convention": [int(N // 32), int(K // 32)],
            "total_B_bytes": int(K * N // 2),
            "c_out_documented": "int16, (N/8, M, 8)",
            "c_out_int8_convention": "int32, (N/4, M, 4)",
        },
        "files": files,
    }
    json.dump(meta, open(f"{outdir}/meta.json", "w"), indent=2)
    return meta


def encode_sum(targets, K):
    """B column values in [-8,7] over K rows summing exactly to each target."""
    t = np.asarray(targets, dtype=np.int64)
    assert np.abs(t).max() <= 8 * K, "target outside int4 sum range"
    base = np.floor_divide(t, K)
    rem = t - base * K                                  # 0 <= rem < K
    col = np.repeat(base[:, None], K, axis=1)
    idx = np.arange(K)[None, :]
    col = col + (idx < rem[:, None]).astype(np.int64)
    assert col.min() >= -8 and col.max() <= 7, "encoding overflowed int4"
    assert (col.sum(1) == t).all()
    return col.T.astype(np.int8)                        # (K, N)


def encode_sum_dense(targets, K, seed=0xD3):
    """Same exact column sums as encode_sum, but DENSE pseudo-random values.

    encode_sum builds each column as a prefix of +-1 followed by zeros, so the
    bank is mostly zeros, highly correlated across columns, and contains an
    all-zero kernel at target 0. That is pathological data, and it provoked a
    data-dependent fault on silicon that dense random weights do not: same
    shape, same layout, byte-identical packing, different values, different
    behaviour. `fc_skip_en` ("when one pixel feature data is 0 the
    corresponding weight data is not fetched") is the obvious suspect.

    This keeps the readback property -- column n still sums to target n, so
    A=all-ones still yields C[m][n] == target -- while looking like real
    weights. Pair it with encode_sum to isolate density as the variable.
    """
    t = np.asarray(targets, dtype=np.int64)
    n = t.size
    rng = np.random.default_rng(seed)
    # A large target forces a DC level of ~t/K on every entry, which eats the
    # int4 range. Shrink the random spread to whatever is left rather than
    # overflowing -- the point is non-sparse, not maximally varied.
    qmax = int(np.abs(t // K).max()) + 1
    spread = max(1, min(4, 6 - qmax))
    u = rng.integers(-spread, spread + 1, size=(n, K)).astype(np.int64)
    d = t - u.sum(1)
    q, r = np.divmod(d, K)                      # r in [0, K)
    u += q[:, None]
    ranks = rng.random((n, K)).argsort(1)
    u += (ranks < r[:, None]).astype(np.int64)
    assert (u.sum(1) == t).all(), "dense encoding lost the target sum"
    assert u.min() >= -8 and u.max() <= 7, (
        f"dense encoding overflowed int4: [{u.min()}, {u.max()}]")
    return u.T.astype(np.int8)                  # (K, N)


def emit_kernelid(root, K, N):
    """Self-describing probe: golden C[m,n] == n - N//2, for every m.

    Read the hardware's output, add N//2, and each lane tells you which
    kernel index actually landed there. A bank mis-tiled at 32 instead of 64
    shows up as lanes repeating in blocks rather than counting."""
    d = f"{root}/probe.kernelid_K{K}_N{N}"
    os.makedirs(d, exist_ok=True)
    tgt = np.arange(N) - N // 2
    Bq = encode_sum(tgt, K)                             # (K,N)
    A = np.ones((2, K), dtype=np.int8)
    C = A.astype(np.int32) @ Bq.astype(np.int32)
    assert (C[0] == tgt).all()

    files = {"B_normal.i4lo.bin": w(f"{d}/B_normal.i4lo.bin", pack_nibbles(Bq))[1]}
    for nt in NTILE_CANDIDATES:
        if N % nt == 0:
            files[f"B_nat_n{nt}.i4lo.bin"] = w(
                f"{d}/B_nat_n{nt}.i4lo.bin",
                pack_nibbles(b_native(Bq, ntile=nt, ktile=KTILE)))[1]
    files["A_m2_ones.i4lo.bin"] = w(f"{d}/A_m2_ones.i4lo.bin", pack_nibbles(A))[1]
    files["C_m2_normal.i16.bin"] = w(f"{d}/C_m2_normal.i16.bin", C.astype(np.int16))[1]
    files["A_m2_ones_nat_k32.i4lo.bin"] = w(f"{d}/A_m2_ones_nat_k32.i4lo.bin",
                                            pack_nibbles(a_native(A, 32)))[1]
    files["C_m2_normal.i32.bin"] = w(f"{d}/C_m2_normal.i32.bin", C)[1]
    files["C_m2_nat_n8.i16.bin"] = w(f"{d}/C_m2_nat_n8.i16.bin",
                                     c_native(C, 8).astype(np.int16))[1]
    json.dump({
        "name": f"kernelid_K{K}_N{N}", "K": K, "N": N, "M": 2,
        "invariant": "C[m][n] == n - N//2 for all m",
        "read_back": "kernel_index = C[m][n] + N//2",
        "diagnoses": "lane->kernel mapping, N tiling, nibble order, K tiling",
        "files": files,
    }, open(f"{d}/meta.json", "w"), indent=2)
    return d


def emit_full_bank(root, mdl, gptq, n_mult):
    """The deployable artifact: every linear, one flat file, leading layout.

    bank.i4.bin is a concatenation of (N/64, K/32, 64, 32) nibble-packed
    tensors; bank_index.json gives byte offsets. Scales go in a parallel file
    as fp32 (N, K/32) so the kernel can fold them into the int32 accumulator.
    """
    import torch
    from safetensors import safe_open
    bank_p = os.path.join(root, "bank.i4.bin")
    scale_p = os.path.join(root, "bank_scales.f32.bin")
    idx, off, soff = [], 0, 0
    shards = [os.path.join(mdl, f) for f in sorted(os.listdir(mdl))
              if f.endswith(".safetensors")]
    names = []
    for s in shards:
        with safe_open(s, framework="pt") as f:
            names += [k for k in f.keys()
                      if k.endswith(".weight") and (".mlp." in k or
                         ".self_attn." in k) and "norm" not in k]
    names = sorted(set(names), key=lambda k: (int(k.split(".")[2]), k))

    with open(bank_p, "wb") as bf, open(scale_p, "wb") as sf:
        for s in shards:
            with safe_open(s, framework="pt") as f:
                keys = [k for k in names if k in f.keys()]
                for k in keys:
                    W = f.get_tensor(k).float().numpy()
                    stem = k[:-len(".weight")]
                    if gptq is not None and f"{stem}.q" in gptq:
                        Wq, sc = gptq[f"{stem}.q"], gptq[f"{stem}.scale"]
                    else:
                        Wq, sc = quant_int4_groupwise(W, GROUP or W.shape[1], axis=1)
                    B = Wq.T.copy()                       # (K,N)
                    n_orig = B.shape[1]
                    B, _ = pad_to(B, 1, n_mult)
                    sc, _ = pad_to(sc, 0, n_mult)
                    nat = pack_nibbles(b_native(B, 64, KTILE))
                    scf = np.ascontiguousarray(sc, dtype=np.float32)
                    bf.write(nat.tobytes())
                    sf.write(scf.tobytes())
                    idx.append({"name": stem, "K": int(B.shape[0]),
                                "N": int(B.shape[1]), "N_original": int(n_orig),
                                "bank_offset": off, "bank_bytes": int(nat.nbytes),
                                "scale_offset": soff,
                                "scale_bytes": int(scf.nbytes)})
                    off += int(nat.nbytes)
                    soff += int(scf.nbytes)
    json.dump({"layout": "(N/64, K/32, 64, 32) nibble-packed, low nibble even",
               "group": GROUP or "per-kernel", "n_pad_multiple": n_mult,
               "scales": "fp32 (N, K/32), parallel file",
               "note": "embed_tokens stays fp16 -- it is a gather, not a matmul",
               "total_bytes": off, "tensors": idx},
              open(os.path.join(root, "bank_index.json"), "w"), indent=2)
    print(f"full bank {off / 2**20:.1f} MiB over {len(idx)} tensors -> bank.i4.bin")


def main():
    from safetensors import safe_open
    global GROUP
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-1.7B-rot")
    ap.add_argument("--out", default="w4a4_probe")
    ap.add_argument("--n-mult", type=int, default=192,
                    help="lcm(48 board constraint, 64 int4 B tile)")
    ap.add_argument("--gptq", default=None,
                    help="int4_weights.npz from quantize.py; uses those exact "
                         "int4 values instead of re-deriving them by RTN")
    ap.add_argument("--group", type=int, default=GROUP,
                    help="K indices per scale; 0 = one scale per kernel")
    ap.add_argument("--all", action="store_true",
                    help="also emit the full deployable bank: every linear in "
                         "the leading (N/64,K/32,64,32) layout, one flat file")
    a = ap.parse_args()
    GROUP = a.group if a.group > 0 else 0

    here = os.path.dirname(os.path.abspath(__file__))
    mdl = a.model if os.path.isabs(a.model) else os.path.join(here, a.model)
    root = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
    os.makedirs(root, exist_ok=True)

    import torch
    shards = [os.path.join(mdl, f) for f in sorted(os.listdir(mdl))
              if f.endswith(".safetensors")]
    want = {k for k, _ in LAYERS}
    tensors = {}
    for s in shards:
        with safe_open(s, framework="pt") as f:
            for k in f.keys():
                if k in want:
                    tensors[k] = f.get_tensor(k).float().numpy()
    missing = want - set(tensors)
    assert not missing, f"missing from {mdl}: {missing}"

    gptq = np.load(a.gptq) if a.gptq else None
    if gptq is not None:
        print(f"using GPTQ int4 values from {os.path.basename(a.gptq)}")

    shapes = []
    for key, name in LAYERS:
        W = tensors[key]                                  # [out=N, in=K]
        stem = key[:-len(".weight")]
        if gptq is not None and f"{stem}.q" in gptq:
            Wq, scale = gptq[f"{stem}.q"], gptq[f"{stem}.scale"]
            src = "gptq"
        else:
            Wq, scale = quant_int4_groupwise(W, GROUP or W.shape[1], axis=1)
            src = "rtn"
        err = np.abs(dequant_int4_groupwise(Wq, scale, GROUP or W.shape[1], 1) - W)
        rel = err.max() / (np.abs(W).max() + 1e-9)

        Bq = Wq.T.copy()                                  # (K, N)
        n_orig = Bq.shape[1]
        Bq, _ = pad_to(Bq, 1, a.n_mult)                   # zero kernels on the tail
        sc, _ = pad_to(scale, 0, a.n_mult)
        d = os.path.join(root, name)
        m = emit_shape(d, name, Bq, sc, n_orig, a.n_mult)
        m["quant_source"] = src
        m["max_rel_err"] = float(rel)
        json.dump(m, open(f"{d}/meta.json", "w"), indent=2)
        shapes.append(m)
        print(f"{name:16s} K={m['K']:5d} N={n_orig:5d}->{m['N']:5d} "
              f"bank={m['derived_registers']['total_B_bytes']/2**20:6.2f} MiB "
              f"{src} rel_err={rel:.4f}")

    # K must be >= 64. At K=32 there is only one k-block, so (N/64,K/32,64,32)
    # and (N/32,K/32,32,32) serialise to identical bytes and the probe cannot
    # see the bug at all. K=32 is kept once, explicitly, as that negative control.
    probes = [emit_kernelid(root, k, n) for k, n in
              ((64, 192), (64, 256), (64, 384), (128, 192), (128, 384))]
    probes += [emit_kernelid(root, 32, 192)]              # blind control
    for p in probes:
        print(f"probe            {os.path.basename(p)}")

    if a.all:
        emit_full_bank(root, mdl, gptq, a.n_mult)

    json.dump({
        "source_model": os.path.basename(mdl),
        "quant": {"weights": "int4 symmetric",
                  "scale_form": "per-kernel" if not GROUP else f"group {GROUP} along K",
                  "activations": "int4 symmetric per-token (runtime)"},
        "rotation": "baked into the weight bank; no runtime Hadamard",
        "constraints": {"N_multiple": 48, "K_multiple": 32,
                        "int4_B_tile_N": 64, "N_pad_multiple_used": a.n_mult},
        "layout_status": "UNVALIDATED -- vendor runtime rejects all int4 "
                         "matmul types; layouts here are candidates to test",
        "candidates": {
            "b_nat_n64": "(N/64, K/32, 64, 32)  vendor-documented int4",
            "b_nat_n32": "(N/32, K/32, 32, 32)  int8 convention, likely current bug",
            "b_nat_n16": "(N/16, K/32, 16, 32)  fp16 convention",
            "b_nat_n64_kmaj": "(N/64, K/32, 32, 64) inner pair swapped",
            "i4hi": "high nibble holds the even index",
        },
        "shapes": [{k: s[k] for k in ("name", "K", "N", "N_original")} for s in shapes],
        "probes": [os.path.basename(p) for p in probes],
    }, open(f"{root}/manifest.json", "w"), indent=2)
    print(f"\n-> {root}/manifest.json")


if __name__ == "__main__":
    main()
