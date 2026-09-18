"""Step 4 -- wikitext2 perplexity for fp16 / W4A16 / W4A4.

W4A4 is simulated: weights are already fake-quantised in the checkpoint, and
activations are quantised per-token symmetric int4 by a forward pre-hook on
every linear the NPU would execute. That mirrors what the bare-metal kernel
does -- one scale per token row, folded into the int32 accumulator on the way
out -- without needing the kernel to exist yet.

    python eval_ppl.py --model Qwen3-1.7B-w4 --act 4
    python eval_ppl.py --model Qwen3-1.7B-rot --act 16    # W16A16 control
"""

import argparse, os, sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


def quant_act(x, bits):
    """Per-token symmetric int4/int8. Last dim is the reduction dim K."""
    if bits >= 16:
        return x
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / qmax
    return (x / s).round().clamp(-qmax - 1, qmax) * s


def install(model, bits, include_head=False, raise_to=None, raise_bits=8,
            only=None):
    """raise_to: iterable of projection names run at raise_bits instead of bits.

    Used to price the fallback for layers whose input keeps outliers that the
    offline rotation cannot reach -- down_proj above all, since its input is
    the raw SwiGLU product and only an online (R4) rotation would flatten it.
    """
    raise_to = set(raise_to or ())
    n, raised = 0, 0
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if not (any(t in name for t in TARGETS) or (include_head and "lm_head" in name)):
            continue
        if only and not any(o in name for o in only):
            continue            # isolation mode: leave everything else in fp16
        b = bits
        if any(r in name for r in raise_to):
            b = raise_bits
            raised += 1
        m.register_forward_pre_hook(lambda _m, inp, b=b: (quant_act(inp[0], b),))
        n += 1
    if raised:
        print(f"  {raised} linears raised to A{raise_bits}: {sorted(raise_to)}")
    return n


@torch.no_grad()
def ppl(model, ids, seqlen, dev, limit=None):
    n = ids.numel() // seqlen
    if limit:
        n = min(n, limit)
    nlls = []
    for i in range(n):
        chunk = ids[0, i * seqlen:(i + 1) * seqlen].unsqueeze(0).to(dev)
        out = model(chunk, labels=chunk)
        nlls.append(out.loss.float() * (seqlen - 1))
        if (i + 1) % 10 == 0:
            cur = torch.exp(torch.stack(nlls).sum() / ((i + 1) * (seqlen - 1)))
            print(f"  {i + 1}/{n}  ppl {cur.item():.3f}", flush=True)
    return torch.exp(torch.stack(nlls).sum() / (n * (seqlen - 1))).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-1.7B-w4")
    ap.add_argument("--act", type=int, default=4, help="activation bits (4/8/16)")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--head", action="store_true", help="also quantise lm_head input")
    ap.add_argument("--raise-to", default="", help="comma list of projections to "
                    "run at --raise-bits instead of --act, e.g. down_proj")
    ap.add_argument("--raise-bits", type=int, default=8)
    ap.add_argument("--only", default="", help="isolation mode: apply --act "
                    "ONLY to these projections; everything else stays fp16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    mdl = a.model if os.path.isabs(a.model) else os.path.join(here, a.model)
    dev = torch.device(a.device)

    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(mdl)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids

    model = AutoModelForCausalLM.from_pretrained(mdl, dtype=torch.float16).to(dev)
    model.eval()
    model.config.use_cache = False
    rt = [s for s in a.raise_to.split(",") if s]
    only = [s for s in a.only.split(",") if s]
    n = install(model, a.act, a.head, rt, a.raise_bits, only) if a.act < 16 else 0
    print(f"{os.path.basename(mdl)}  A{a.act}  hooks on {n} linears  "
          f"seqlen {a.seqlen}")
    p = ppl(model, ids, a.seqlen, dev, a.limit)
    tag = (f"A{a.act}[only {','.join(only)}]" if only else
           f"A{a.act}" + (f"+{'/'.join(rt)}@A{a.raise_bits}" if rt else ""))
    print(f"\nPPL  {os.path.basename(mdl)}  W4{tag}  =  {p:.4f}")


if __name__ == "__main__":
    main()
