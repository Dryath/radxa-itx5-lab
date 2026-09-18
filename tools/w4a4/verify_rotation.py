"""Prove the rotated model is the same function as the original.

If this does not pass, nothing downstream means anything: every quantisation
result would be measuring a broken rotation rather than 4-bit error.
"""
import argparse, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="Qwen3-1.7B")
ap.add_argument("--rot", default="Qwen3-1.7B-rot")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
S, R = os.path.join(here, a.src), os.path.join(here, a.rot)

tok = AutoTokenizer.from_pretrained(S)
text = ("The RK3588 NPU exposes an int4 datapath at 2048 MACs per cycle per "
        "core, which the vendor runtime never enabled.")
ids = tok(text, return_tensors="pt").input_ids


def logits(path):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        out = m(ids).logits.float()
    del m
    return out


a_l = logits(S)
b_l = logits(R)
d = (a_l - b_l).abs()
scale = a_l.abs().max()
cos = torch.nn.functional.cosine_similarity(          # float64: the float32
    a_l.reshape(-1).double(), b_l.reshape(-1).double(), dim=0).item()
top1 = (a_l.argmax(-1) == b_l.argmax(-1)).float().mean().item()

print(f"tokens              {ids.shape[1]}")
print(f"logit max |diff|    {d.max():.5f}   (logit scale {scale:.2f})")
print(f"relative            {d.max() / scale:.2e}")
print(f"cosine              {cos:.8f}")
print(f"top-1 agreement     {top1 * 100:.2f}%")
ok = cos > 0.9999 and top1 == 1.0
print("\nROTATION VERIFIED -- same function" if ok else "\nFAILED -- do not proceed")
sys.exit(0 if ok else 1)
