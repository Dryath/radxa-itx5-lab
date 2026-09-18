"""Step 2 -- GPTQ int4 over the rotated model.

Group size 32 along the input dim mirrors the hardware's K/32 weight tile.
CAVEAT: a tile is not a scale. Applying a per-group scale requires the
accumulator to be read out per K-group; a kernel that accumulates all of K
in one int4 matmul can only apply ONE scale per kernel. The shipping int8
bank is per-row for exactly that reason. Use --group 0 to build that form
and compare -- do not assume group 32 is consumable.

act_order is deliberately OFF. Reordering columns by Hessian diagonal helps
perplexity but scrambles which K indices share a group, and the groups have
to stay contiguous in K or they no longer match the tiling.

Runs layer-by-layer: model lives on CPU, one decoder layer at a time on the
GPU. Peak GPU is one layer plus the largest Hessian (down_proj 6144^2 fp32
= 151 MiB), which fits the 6 GiB card with room to spare.

Usage:
    python quantize.py --model Qwen3-1.7B-rot --out Qwen3-1.7B-w4 \
        --nsamples 128 --seqlen 1024
"""

import argparse, gc, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GROUP = 32
BLOCK = 128          # GPTQ column block
PERCDAMP = 0.01


def get_calib(tok, nsamples, seqlen, seed=0, cache="/tmp/wikitext2_train_ids.pt"):
    """wikitext2 train, packed into nsamples x seqlen token windows.

    Tokenising the whole split is ~2.5M tokens and takes minutes, so the id
    tensor is cached -- reruns should not pay for it twice.
    """
    if os.path.exists(cache):
        ids = torch.load(cache)
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        t = time.time()
        ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
        print(f"tokenised {ids.numel()} tokens in {time.time() - t:.0f}s", flush=True)
        torch.save(ids, cache)
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(nsamples):
        i = torch.randint(0, ids.numel() - seqlen - 1, (1,), generator=g).item()
        out.append(ids[i:i + seqlen].unsqueeze(0))
    return torch.cat(out, 0)


class GPTQ:
    """Standard GPTQ with a symmetric int4 group quantiser.

    group is the number of K indices sharing one scale. Note this is a
    CONSTRAINT ON THE KERNEL, not a free accuracy knob: applying a per-group
    scale requires the accumulator to be read out per K-group. A kernel that
    accumulates the whole of K in one int4 matmul can only apply one scale per
    kernel, i.e. group == in_features. Set --group 0 for that form.
    """

    def __init__(self, layer, group=GROUP):
        self.group = group if group > 0 else layer.weight.shape[1]
        self.layer = layer
        W = layer.weight.data
        self.rows, self.cols = W.shape          # [out=N, in=K]
        self.H = torch.zeros((self.cols, self.cols), device=W.device,
                             dtype=torch.float32)
        self.n = 0

    def add_batch(self, x):
        x = x.reshape(-1, x.shape[-1]).t().float()      # [K, tokens]
        t = x.shape[1]
        self.H *= self.n / (self.n + t)
        self.n += t
        self.H += (2.0 / self.n) * (x @ x.t())

    def quantize(self):
        W = self.layer.weight.data.float().clone()
        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = PERCDAMP * torch.mean(torch.diag(H))
        H[range(self.cols), range(self.cols)] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        G = self.group
        per_row = G >= self.cols
        Q = torch.zeros_like(W)
        scales = torch.zeros((self.rows, max(1, self.cols // G)),
                             device=W.device, dtype=torch.float32)
        if per_row:
            # one scale per kernel: fix it up front from the whole row, since
            # no block-local view can see the full reduction axis
            cur_s = (W.abs().amax(dim=1) / 7.0).clamp(min=1e-8)
            scales[:, 0] = cur_s

        for i0 in range(0, self.cols, BLOCK):
            i1 = min(i0 + BLOCK, self.cols)
            W1 = W[:, i0:i1].clone()
            Q1 = torch.zeros_like(W1)
            E1 = torch.zeros_like(W1)
            Hi = Hinv[i0:i1, i0:i1]

            for j in range(i1 - i0):
                col = i0 + j
                if not per_row and col % G == 0:
                    # BLOCK is a multiple of G and both are aligned, so a group
                    # never straddles a block: read it from W1, which carries
                    # the error correction applied so far.
                    g = W1[:, j:j + G]
                    s = (g.abs().max(dim=1).values / 7.0).clamp(min=1e-8)
                    scales[:, col // G] = s
                    cur_s = s
                w = W1[:, j]
                d = Hi[j, j]
                q = torch.clamp(torch.round(w / cur_s), -8, 7)
                Q1[:, j] = q
                err = (w - q * cur_s) / d
                W1[:, j:] -= err.unsqueeze(1) * Hi[j, j:].unsqueeze(0)
                E1[:, j] = err

            Q[:, i0:i1] = Q1
            W[:, i1:] -= E1 @ Hinv[i0:i1, i1:]

        deq = Q * (cur_s.unsqueeze(1) if per_row
                   else scales.repeat_interleave(G, dim=1))
        err = (deq - self.layer.weight.data.float()).pow(2).sum().sqrt()
        rel = (err / self.layer.weight.data.float().pow(2).sum().sqrt()).item()
        self.layer.weight.data = deq.to(self.layer.weight.dtype)
        return Q.to(torch.int8).cpu().numpy(), scales.cpu().numpy(), rel

    def free(self):
        self.H = None
        gc.collect()
        torch.cuda.empty_cache()


def find_linears(module):
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-1.7B-rot")
    ap.add_argument("--out", default="Qwen3-1.7B-w4")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--group", type=int, default=GROUP,
                    help="K indices per scale; 0 = one scale per kernel "
                         "(required if the kernel accumulates all of K in "
                         "a single int4 matmul)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    mdl = a.model if os.path.isabs(a.model) else os.path.join(here, a.model)
    out = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(a.device)

    tok = AutoTokenizer.from_pretrained(mdl)
    model = AutoModelForCausalLM.from_pretrained(mdl, dtype=torch.float16)
    model.eval()
    model.config.use_cache = False
    layers = model.model.layers
    print(f"{len(layers)} layers, calib {a.nsamples}x{a.seqlen}")

    tcal = time.time()
    calib = get_calib(tok, a.nsamples, a.seqlen)
    print(f"calibration ready in {time.time() - tcal:.0f}s", flush=True)

    # ---- capture the input to layer 0 -------------------------------
    inps = torch.zeros((a.nsamples, a.seqlen, model.config.hidden_size),
                       dtype=torch.float16)
    cache = {"i": 0, "kw": None}

    class Catcher(nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, x, **kw):
            inps[cache["i"]] = x.squeeze(0).cpu()
            cache["i"] += 1
            cache["kw"] = kw
            raise ValueError

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = Catcher(layers[0].to(dev))
    for s in range(a.nsamples):
        try:
            model(calib[s:s + 1].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].mod.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    kw = {k: v for k, v in cache["kw"].items() if k != "past_key_value"}
    kw.pop("past_key_values", None)
    outs = torch.zeros_like(inps)

    packed, report = {}, []
    t0 = time.time()
    for li in range(len(layers)):
        layer = layers[li].to(dev)
        lin = find_linears(layer)
        gptq = {n: GPTQ(m, a.group) for n, m in lin.items()}
        handles = [m.register_forward_pre_hook(
            (lambda name: lambda _m, inp: gptq[name].add_batch(inp[0].data))(n))
            for n, m in lin.items()]

        th = time.time()
        with torch.no_grad():
            for s in range(a.nsamples):
                layer(inps[s:s + 1].to(dev), **kw)
        torch.cuda.synchronize()
        th = time.time() - th
        for h in handles:
            h.remove()

        tq = time.time()
        rels = {}
        for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                  "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                  "mlp.down_proj"):
            if n not in gptq:
                continue
            q, s_, rel = gptq[n].quantize()
            gptq[n].free()
            packed[f"model.layers.{li}.{n}"] = (q, s_)
            rels[n] = round(rel, 5)

        torch.cuda.synchronize()
        tq = time.time() - tq

        to = time.time()
        with torch.no_grad():
            for s in range(a.nsamples):
                outs[s] = layer(inps[s:s + 1].to(dev), **kw)[0].squeeze(0).cpu()
        torch.cuda.synchronize()
        to = time.time() - to

        layers[li] = layer.cpu()
        del layer, gptq
        gc.collect()
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        report.append({"layer": li, "rel_err": rels})
        worst = max(rels, key=rels.get)
        print(f"layer {li:2d}  worst {worst} {rels[worst]:.4f}  "
              f"down {rels.get('mlp.down_proj', 0):.4f}  "
              f"| H {th:.1f}s quant {tq:.1f}s out {to:.1f}s "
              f"| total {time.time() - t0:.0f}s", flush=True)

    model.save_pretrained(out)
    tok.save_pretrained(out)
    np.savez_compressed(
        os.path.join(out, "int4_weights.npz"),
        **{f"{k}.q": v[0] for k, v in packed.items()},
        **{f"{k}.scale": v[1] for k, v in packed.items()})
    json.dump({"group": a.group, "act_order": False, "sym": True, "bits": 4,
               "nsamples": a.nsamples, "seqlen": a.seqlen,
               "layers": report}, open(os.path.join(out, "gptq_report.json"), "w"),
              indent=2)
    print(f"\nsaved fake-quant model + int4_weights.npz -> {out}")


if __name__ == "__main__":
    main()
