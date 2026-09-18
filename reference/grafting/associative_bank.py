"""
Cross-modal associative bank — a clean-room reference implementation.

  >>> CLEAN-ROOM NOTE <<<
  Written fresh and generic from the *mechanism* described in docs/09 (by Claude, at the
  author's request). NOT extracted from any engine, and NOT the private substrate — this is a
  minimal, honest, runnable illustration of the idea, nothing more. The docs/09 substrate is
  explicitly unbuilt research; this file exists so the *concept* is legible in ~80 lines.

The idea (docs/09): give frozen single-modality encoders a shared weight bank with EMPTY
off-diagonal space, then let Hebbian co-activation fill that space to bind the modalities. Cue
one modality and the bank evokes the others: read "cat" and the stored *visual* cat and the
*sound* of the word fire. It's an auto-associative (Hopfield-style) net with block structure and
k-WTA sparsity — evocation from stored experience, not generation.

This toy uses random unit vectors as stand-ins for real encoder outputs. numpy only.
"""
from __future__ import annotations
import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def kwta(x: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest-magnitude entries, zero the rest (sparse coding)."""
    if k >= x.size:
        return x
    keep = np.argpartition(np.abs(x), -k)[-k:]
    y = np.zeros_like(x)
    y[keep] = x[keep]
    return y


class AssociativeBank:
    def __init__(self, mod_dims: dict[str, int], k: int | None = None):
        # block-diagonal layout: one contiguous slice of the state per modality
        self.mods = list(mod_dims)
        self.slices, off = {}, 0
        for m in self.mods:
            self.slices[m] = slice(off, off + mod_dims[m]); off += mod_dims[m]
        self.D = off
        self.k = k or max(4, self.D // 12)          # sparsity: ~8% active by default
        self.W = np.zeros((self.D, self.D), np.float32)

    def _state(self, parts: dict[str, np.ndarray]) -> np.ndarray:
        x = np.zeros(self.D, np.float32)
        for m, v in parts.items():
            x[self.slices[m]] = _unit(np.asarray(v, np.float32).reshape(-1))
        return x

    def bind(self, parts: dict[str, np.ndarray], lr: float = 1.0) -> None:
        """Co-activate all modalities of one concept and write the outer product into the bank.
        The off-diagonal blocks are what get filled — that's the cross-modal binding. Symmetric,
        so any direction of recall comes free."""
        x = kwta(self._state(parts), self.k)
        self.W += lr * np.outer(x, x)

    def evoke(self, cue: dict[str, np.ndarray], steps: int = 3) -> dict[str, np.ndarray]:
        """Drive the bank with a partial cue (one or two modalities) and let the recurrence fill
        the rest. Returns the evoked vector per modality."""
        x = kwta(self._state(cue), self.k)
        for _ in range(steps):
            x = kwta(_unit(self.W @ x), self.k)
        return {m: x[self.slices[m]].copy() for m in self.mods}


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dims = {"vision": 96, "audio": 64, "text": 48}
    bank = AssociativeBank(dims)

    N = 40                                            # 40 concepts, each a code per modality
    concepts = [{m: _unit(rng.standard_normal(d)) for m, d in dims.items()} for _ in range(N)]
    for c in concepts:
        bank.bind(c)

    # cue with VISION ONLY; measure how well the evoked AUDIO/TEXT match the true concept
    hits, cos_aud = 0, []
    for ci, c in enumerate(concepts):
        ev = bank.evoke({"vision": c["vision"]})
        # nearest concept by evoked audio block
        sims = [float(_unit(ev["audio"]) @ c2["audio"]) for c2 in concepts]
        if int(np.argmax(sims)) == ci:
            hits += 1
        cos_aud.append(float(_unit(ev["audio"]) @ c["audio"]))
    print(f"  concepts: {N}   cue: vision only")
    print(f"  cross-modal recall (evoked audio -> right concept): {hits}/{N} = {hits/N:.0%}  "
          f"(chance {1/N:.0%})")
    print(f"  mean cosine of evoked audio to the true audio: {np.mean(cos_aud):+.3f}")
