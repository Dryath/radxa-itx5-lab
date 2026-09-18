"""Step 1 -- offline rotation of Qwen3-1.7B for W4A4.

Fuses every RMSNorm scale into the linear that consumes it, then bakes a
randomized Hadamard into the weight bank itself. Nothing is left for the
NPU kernel to do at runtime: it stays a pure int4 matmul.

  R1  hidden dim 2048 (= 2^11, exact Sylvester)
        readers of the residual (q,k,v,gate,up):  W <- W @ Q
        writers to the residual (o,down):         W <- Q.T @ W
        embed_tokens:                             E <- E @ Q
        lm_head:                          (E * g_final) @ Q   [must untie]

      Bare RMSNorm commutes with Q exactly: mean((xQ)^2) == mean(x^2)
      because Q is orthonormal. That is why the scales must be fused out
      first -- an elementwise g does not commute with a rotation.

  R2  head dim 128 (= 2^7), fused into v_proj rows / o_proj columns.
      Free because attention output softmax(QK^T)V is linear in V.

R3/R4 (online rotations) are deliberately excluded -- they would cost a
runtime Hadamard per layer. Consequence: down_proj's input still carries
the raw SwiGLU outliers. That is the layer to watch in eval.

Usage:  python rotate.py --src Qwen3-1.7B --dst Qwen3-1.7B-rot
"""

import argparse, json, os, shutil, sys
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import randomized_hadamard

SEED_R1 = 1337
SEED_R2 = 4242


def load_state(src):
    """Read every shard into one dict of numpy float32 arrays.

    Goes through torch because the checkpoint is bfloat16 and numpy has no
    bfloat16 dtype -- safe_open(framework="np") would refuse it.
    """
    import torch
    idx = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx):
        shards = sorted(set(json.load(open(idx))["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    sd = {}
    for s in shards:
        with safe_open(os.path.join(src, s), framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                sd[k] = t.float().numpy() if t.dtype.is_floating_point else t.numpy()
    return sd


def chunked_matmul(A, Q, chunk=8192):
    """(A.astype(f64) @ Q) row-chunked, returns float64. For the 152k-row embed."""
    out = np.empty((A.shape[0], Q.shape[1]), dtype=np.float64)
    for i in range(0, A.shape[0], chunk):
        out[i:i + chunk] = A[i:i + chunk].astype(np.float64) @ Q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Qwen3-1.7B")
    ap.add_argument("--dst", default="Qwen3-1.7B-rot")
    ap.add_argument("--no-r2", action="store_true", help="skip the head-dim rotation")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    src = a.src if os.path.isabs(a.src) else os.path.join(here, a.src)
    dst = a.dst if os.path.isabs(a.dst) else os.path.join(here, a.dst)
    os.makedirs(dst, exist_ok=True)

    cfg = json.load(open(os.path.join(src, "config.json")))
    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    HD = cfg.get("head_dim", H // cfg["num_attention_heads"])
    NH = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    out_dtype = np.dtype(a.dtype)

    print(f"hidden={H} layers={L} head_dim={HD} heads={NH}/{NKV} -> {dst}")

    Q = randomized_hadamard(H, SEED_R1)          # (H,H) float64
    Hh = None if a.no_r2 else randomized_hadamard(HD, SEED_R2)

    # sanity: orthonormality is the entire correctness argument
    err = np.abs(Q.T @ Q - np.eye(H)).max()
    assert err < 1e-10, f"Q not orthonormal: {err}"
    print(f"R1 orthonormality residual {err:.2e}")

    sd = load_state(src)
    new = {}

    def rot_block_rows(W, B):
        """Per-head rotation on the OUTPUT axis: rows of each 128-block."""
        o, i = W.shape
        return (W.reshape(o // B.shape[0], B.shape[0], i)
                 .transpose(0, 2, 1) @ B).transpose(0, 2, 1).reshape(o, i)

    def rot_block_cols(W, B):
        """Per-head rotation on the INPUT axis: columns of each 128-block."""
        o, i = W.shape
        return (W.reshape(o, i // B.shape[0], B.shape[0]) @ B).reshape(o, i)

    # ---- embeddings -------------------------------------------------
    E = sd["model.embed_tokens.weight"]                     # [vocab, H]
    g_final = sd["model.norm.weight"].astype(np.float64)    # [H]
    new["model.embed_tokens.weight"] = chunked_matmul(E, Q).astype(out_dtype)
    # lm_head absorbs the final norm scale, so it differs from embed -> untie
    lm_src = sd.get("lm_head.weight", E)
    new["lm_head.weight"] = chunked_matmul(
        lm_src.astype(np.float64) * g_final[None, :], Q).astype(out_dtype)
    new["model.norm.weight"] = np.ones(H, dtype=out_dtype)
    print("embed + lm_head rotated (untied)")

    # ---- per layer --------------------------------------------------
    for li in range(L):
        p = f"model.layers.{li}."
        g_in = sd[p + "input_layernorm.weight"].astype(np.float64)
        g_post = sd[p + "post_attention_layernorm.weight"].astype(np.float64)

        for name, g in (("self_attn.q_proj", g_in), ("self_attn.k_proj", g_in),
                        ("self_attn.v_proj", g_in), ("mlp.gate_proj", g_post),
                        ("mlp.up_proj", g_post)):
            W = sd[p + name + ".weight"].astype(np.float64) * g[None, :]  # fuse norm
            if Hh is not None and name.endswith("v_proj"):
                W = rot_block_rows(W, Hh)                                # R2 out
            new[p + name + ".weight"] = (W @ Q).astype(out_dtype)        # R1 in

        Wo = sd[p + "self_attn.o_proj.weight"].astype(np.float64)
        if Hh is not None:
            Wo = rot_block_cols(Wo, Hh)                                  # R2 in
        new[p + "self_attn.o_proj.weight"] = (Q.T @ Wo).astype(out_dtype)

        Wd = sd[p + "mlp.down_proj.weight"].astype(np.float64)
        new[p + "mlp.down_proj.weight"] = (Q.T @ Wd).astype(out_dtype)

        new[p + "input_layernorm.weight"] = np.ones(H, dtype=out_dtype)
        new[p + "post_attention_layernorm.weight"] = np.ones(H, dtype=out_dtype)
        # q_norm / k_norm act on head_dim before RoPE and are left alone
        for n in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            if p + n in sd:
                new[p + n] = sd[p + n].astype(out_dtype)
        if li % 7 == 0 or li == L - 1:
            print(f"  layer {li:2d}/{L - 1} done")

    for k, v in sd.items():
        if k not in new and k != "model.embed_tokens.weight":
            new[k] = v.astype(out_dtype) if v.dtype.kind == "f" else v

    save_file(new, os.path.join(dst, "model.safetensors"),
              metadata={"format": "pt"})
    np.savez(os.path.join(dst, "rotations.npz"), R1=Q.astype(np.float32),
             R2=(Hh.astype(np.float32) if Hh is not None else np.zeros(0, np.float32)),
             seed_r1=SEED_R1, seed_r2=SEED_R2)

    cfg["tie_word_embeddings"] = False
    cfg["torch_dtype"] = a.dtype
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "generation_config.json"):
        s = os.path.join(src, f)
        if os.path.exists(s):
            shutil.copy(s, dst)

    total = sum(v.nbytes for v in new.values())
    print(f"wrote {len(new)} tensors, {total / 2**30:.2f} GiB -> {dst}")
    print("rotations.npz holds R1/R2 so packing and the board reference agree")


if __name__ == "__main__":
    main()
