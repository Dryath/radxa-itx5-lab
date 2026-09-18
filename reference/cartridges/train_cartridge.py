"""
Train a domain "cartridge" (a LoRA adapter) — a clean-room reference scaffold.

  >>> CLEAN-ROOM NOTE <<<
  Written fresh and generic from the design in docs/14 (by Claude, at the author's request).
  NOT extracted from any engine, and it contains NO domain data of any kind — you bring your own
  corpus. It's the *shape* of a cartridge trainer, small enough to read.

  >>> REFERENCE SCAFFOLD <<<
  This requires `torch transformers peft datasets` and is NOT executed in this repo's CI (no GPU
  here). Read it as the correct pattern; run it on your own box / a short cloud burst.

docs/14: a small resident BASE model + hot-swappable per-domain LoRA "cartridges". This trains
one cartridge — tens of MB — for one narrow domain, leaving the base weights untouched. Keep it
narrow; capacity per base is a thing you measure, not assume (METHOD.md).
"""
from __future__ import annotations
import argparse


def train_cartridge(base_model: str, corpus_path: str, out_dir: str,
                    rank: int = 16, alpha: int = 32, epochs: int = 1,
                    target_modules=("q_proj", "v_proj")):
    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer, DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)

    # the cartridge itself: a small low-rank adapter on a couple of projections
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=list(target_modules)))
    model.print_trainable_parameters()          # tens of MB, not tens of GB

    # BYO corpus: a text file / jsonl of *your* narrow domain (this repo ships none)
    ds = load_dataset("text", data_files=corpus_path)["train"]
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=1024), batched=True,
                remove_columns=["text"])

    Trainer(
        model=model,
        args=TrainingArguments(output_dir=out_dir, num_train_epochs=epochs,
                               per_device_train_batch_size=2, gradient_accumulation_steps=8,
                               learning_rate=2e-4, bf16=True, logging_steps=20, save_strategy="epoch"),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()

    model.save_pretrained(out_dir)              # <- this is the cartridge (adapter only)
    tok.save_pretrained(out_dir)
    print(f"cartridge saved to {out_dir}  (base weights untouched)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a domain LoRA cartridge (docs/14).")
    ap.add_argument("--base", required=True, help="base model id/path (stays resident)")
    ap.add_argument("--corpus", required=True, help="your domain text/jsonl (BYO — none ships here)")
    ap.add_argument("--out", required=True, help="where to write the cartridge (adapter)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    train_cartridge(a.base, a.corpus, a.out, rank=a.rank, epochs=a.epochs)
