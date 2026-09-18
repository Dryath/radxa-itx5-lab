"""
Hot-swap domain cartridges on one resident base — a clean-room reference scaffold.

  >>> CLEAN-ROOM NOTE <<<
  Written fresh and generic from docs/14 (by Claude, at the author's request). No engine code,
  no domain data. The pattern only.

  >>> REFERENCE SCAFFOLD <<<
  Requires `torch transformers peft`; not executed in CI here (no GPU). Correct pattern, run it
  yourself.

docs/14: the base stays resident (your working set); cartridges are the paged domains. Load the
base ONCE, attach several adapters, and switch between them per request in microseconds — the
same "hold the resident thing, page the cold thing" logic as the appliance router (docs/13), one
level down inside the model.
"""
from __future__ import annotations


class CartridgeRuntime:
    def __init__(self, base_model: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(base_model)
        self._base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
        self.model = None            # becomes a PeftModel once the first cartridge is loaded
        self._loaded: set[str] = set()

    def load_cartridge(self, name: str, path: str) -> None:
        """Attach a cartridge (adapter) under a name. Cheap — tens of MB — so keep several."""
        from peft import PeftModel
        if self.model is None:
            self.model = PeftModel.from_pretrained(self._base, path, adapter_name=name)
        elif name not in self._loaded:
            self.model.load_adapter(path, adapter_name=name)
        self._loaded.add(name)

    def use(self, name: str) -> None:
        """Switch the active cartridge. This is the hot-swap — no reload, no base copy."""
        self.model.set_adapter(name)

    def use_base_only(self) -> None:
        """Disable all cartridges — answer from the base's general operators alone."""
        if self.model is not None:
            self.model.disable_adapter_layers()

    def generate(self, prompt: str, cartridge: str | None = None, **kw) -> str:
        if cartridge is not None:
            self.use(cartridge)
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**ids, **kw)
        return self.tok.decode(out[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Illustrative usage (won't run without the deps + your own trained cartridges):
    #
    #   rt = CartridgeRuntime("your/base-model")
    #   rt.load_cartridge("domain_a", "./cartridges/domain_a")
    #   rt.load_cartridge("domain_b", "./cartridges/domain_b")
    #   print(rt.generate("...", cartridge="domain_a"))   # base + A
    #   print(rt.generate("...", cartridge="domain_b"))   # same base, swapped to B
    #   rt.use_base_only()                                # general operators, no domain
    print(__doc__)
