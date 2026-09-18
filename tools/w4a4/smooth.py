"""Step 1b -- offline per-channel smoothing for the layers rotation cannot reach.

R4 (a Hadamard on down_proj's input) cannot be made offline: that input is the
elementwise product silu(gate) * up, and a rotation does not commute through an
elementwise product. Division by a per-channel vector DOES commute through it,
so the same outlier problem has an offline answer:

    down(x) = ((x / s) @ (Wd * s).T)          x = silu(g) * u
            = (silu(g) * (u / s)) @ (Wd * s).T

so 1/s folds into up_proj's ROWS and s folds into down_proj's COLUMNS. Nothing
is left at runtime; the kernel stays a pure int4 matmul.

s_j = max|X_j|^alpha / max|W_:,j|^(1-alpha), the SmoothQuant migration factor:
alpha=0 leaves activations alone, alpha=1 flattens them completely and dumps
all the difficulty on the weights. For W4A4 the weights are only 4 bits too,
so the balance point matters -- sweep it.

Usage:
    python smooth.py --model Qwen3-0.6B-rot --out Qwen3-0.6B-rs --alpha 0.65
"""

import argparse, json, os, shutil, sys, time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


TARGETS = ("mlp.down_proj", "self_attn.o_proj")


def calibrate(model, tok, dev, nsamples, seqlen):
    """Per-channel absmax of every smoothable input, over calibration text."""
    stats = {}

    def hook(name):
        def f(_m, inp):
            x = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).amax(0).float()
            stats[name] = torch.maximum(stats[name], x) if name in stats else x
        return f

    hs = [m.register_forward_pre_hook(hook(n))
          for n, m in model.named_modules()
          if isinstance(m, nn.Linear) and n.endswith(TARGETS)]

    ids = torch.load("/tmp/wikitext2_train_ids.pt")
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _ in range(nsamples):
            i = torch.randint(0, ids.numel() - seqlen - 1, (1,), generator=g).item()
            model(ids[i:i + seqlen].unsqueeze(0).to(dev))
    for h in hs:
        h.remove()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-0.6B-rot")
    ap.add_argument("--out", default="Qwen3-0.6B-rs")
    ap.add_argument("--alpha", type=float, default=0.65)
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    mdl = a.model if os.path.isabs(a.model) else os.path.join(here, a.model)
    out = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(a.device)

    tok = AutoTokenizer.from_pretrained(mdl)
    model = AutoModelForCausalLM.from_pretrained(mdl, dtype=torch.float16).to(dev)
    model.eval()
    model.config.use_cache = False

    t = time.time()
    stats = calibrate(model, tok, dev, a.nsamples, a.seqlen)
    print(f"calibrated {len(stats)} down_proj inputs in {time.time() - t:.0f}s")

    cfg = model.config
    HD = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads
    RATIO = NH // NKV

    def factor(act, wmax, alpha):
        s = (act.clamp(min=1e-5).pow(alpha) /
             wmax.clamp(min=1e-5).pow(1 - alpha)).clamp(min=1e-5)
        return s / s.mean()

    sd = model.state_dict()
    report = []
    for name, act_max in sorted(stats.items()):
        li = name.split(".")[2]
        act_max = act_max.clamp(min=1e-5)

        if name.endswith("mlp.down_proj"):
            up = f"model.layers.{li}.mlp.up_proj.weight"
            dn = f"model.layers.{li}.mlp.down_proj.weight"
            Wd = sd[dn].float()                              # [hidden, inter]
            s = factor(act_max.to(Wd.device), Wd.abs().amax(0), a.alpha)
            sd[up] = (sd[up].float() / s.unsqueeze(1)).half()   # rows  <- 1/s
            sd[dn] = (Wd * s.unsqueeze(0)).half()               # cols  <- s
            eff = s
            tag = "down_proj"
        else:
            # o_proj input is concat over QUERY heads; v_proj has only KV heads.
            # Under GQA each kv head feeds RATIO query heads, so one scale must
            # serve all of them -- reduce with max over the sharing group.
            vp = f"model.layers.{li}.self_attn.v_proj.weight"
            op = f"model.layers.{li}.self_attn.o_proj.weight"
            Wo = sd[op].float()                              # [hidden, NH*HD]
            aq = act_max.to(Wo.device).view(NKV, RATIO, HD).amax(1)      # [NKV,HD]
            wq = Wo.abs().amax(0).view(NKV, RATIO, HD).amax(1)           # [NKV,HD]
            s_kv = factor(aq.reshape(-1), wq.reshape(-1), a.alpha)       # [NKV*HD]
            s_q = (s_kv.view(NKV, 1, HD).expand(NKV, RATIO, HD)
                       .reshape(-1))                                     # [NH*HD]
            sd[vp] = (sd[vp].float() / s_kv.unsqueeze(1)).half()  # v rows <- 1/s
            sd[op] = (Wo * s_q.unsqueeze(0)).half()               # o cols <- s
            eff = s_q
            tag = "o_proj"

        before = (act_max.max() / act_max.mean()).item()
        sm = act_max.to(eff.device) / eff
        after = (sm.max() / sm.mean()).item()
        report.append({"layer": int(li), "target": tag,
                       "outlier_ratio_before": round(before, 1),
                       "outlier_ratio_after": round(after, 1)})

    for t in ("down_proj", "o_proj"):
        rs = [r for r in report if r["target"] == t]
        if rs:
            b = np.mean([r["outlier_ratio_before"] for r in rs])
            af = np.mean([r["outlier_ratio_after"] for r in rs])
            print(f"alpha={a.alpha}  {t:10s} mean outlier ratio {b:6.1f} -> {af:5.1f}")

    save_file({k: v.contiguous().cpu() for k, v in sd.items()},
              os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(mdl, "config.json")))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "generation_config.json", "rotations.npz"):
        p = os.path.join(mdl, f)
        if os.path.exists(p):
            shutil.copy(p, out)
    json.dump({"alpha": a.alpha,
               "targets": ["mlp.down_proj input", "self_attn.o_proj input"],
               "folded_into": {"down_proj": "up_proj rows (1/s), down_proj cols (s)",
                               "o_proj": "v_proj rows (1/s), o_proj cols (s); "
                                         "scale shared across GQA query group"},
               "runtime_cost": "none", "layers": report},
              open(os.path.join(out, "smooth_report.json"), "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
