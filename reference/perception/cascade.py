"""
Perception cascade — a clean-room reference implementation.

  >>> CLEAN-ROOM NOTE <<<
  Written fresh and generic from the design principle in docs/12 (by Claude, at the author's
  request). NOT extracted from any engine. It's the *pattern*, ~60 lines. Adapt freely.

The principle (docs/12): cheapest deterministic read first, expensive neural read last, each at
its own cadence. Most inputs never reach a model. And when a ground-truth label already exists
(a transcript, a calendar, a schedule), use it instead of asking a model to re-derive it.

This is a tiny scheduler: an ordered list of readers by cost, each returning (label, confidence)
or None. The first reader confident enough wins; only the stragglers fall through to the
expensive one. It tracks where work actually landed, so you can see the model was rarely needed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import time


@dataclass
class Reader:
    name: str
    fn: Callable[[object], Optional[tuple[str, float]]]   # -> (label, confidence) or None
    threshold: float = 0.7                                # min confidence to accept its answer
    cost_ms: float = 0.0                                  # nominal cost, for the cadence budget


@dataclass
class Cascade:
    readers: list[Reader]                                 # cheapest first
    resolved_by: dict[str, int] = field(default_factory=dict)

    def read(self, x, ground_truth: Optional[str] = None) -> dict:
        # label-from-ground-truth: if you already know the answer, don't run anything
        if ground_truth is not None:
            self.resolved_by["ground_truth"] = self.resolved_by.get("ground_truth", 0) + 1
            return {"label": ground_truth, "confidence": 1.0, "by": "ground_truth", "cost_ms": 0.0}
        spent = 0.0
        for r in self.readers:
            spent += r.cost_ms
            out = r.fn(x)
            if out is not None and out[1] >= r.threshold:
                self.resolved_by[r.name] = self.resolved_by.get(r.name, 0) + 1
                return {"label": out[0], "confidence": out[1], "by": r.name, "cost_ms": spent}
        # nobody was confident: fall through to the last reader's best guess (or None)
        best = self.readers[-1].fn(x)
        self.resolved_by["unresolved"] = self.resolved_by.get("unresolved", 0) + 1
        return {"label": best[0] if best else None,
                "confidence": best[1] if best else 0.0, "by": "fallthrough", "cost_ms": spent}


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    import random
    rng = random.Random(0)

    # three readers, cheap -> expensive. Most events are "silence" and a threshold catches them.
    def cheap_energy_gate(x):           # a deterministic threshold — nearly free
        return ("silence", 0.99) if x["energy"] < 0.1 else None

    def mid_heuristic(x):               # a cheap heuristic — catches the easy positives
        if x["energy"] > 0.6 and x["periodic"]:
            return ("speech", 0.85)
        return None

    def expensive_model(x):            # the NN — only the ambiguous middle reaches it
        time.sleep(0.0)                # stand-in for real inference cost
        return ("speech" if x["periodic"] else "noise", 0.6 + 0.4 * rng.random())

    casc = Cascade([
        Reader("energy_gate", cheap_energy_gate, threshold=0.9, cost_ms=0.01),
        Reader("heuristic", mid_heuristic, threshold=0.8, cost_ms=0.2),
        Reader("model", expensive_model, threshold=0.7, cost_ms=40.0),
    ])

    # a realistic stream: mostly silence, some clear speech, a little genuinely ambiguous
    for _ in range(1000):
        roll = rng.random()
        if roll < 0.60:                       # silence
            x = {"energy": rng.uniform(0.0, 0.08), "periodic": False}
        elif roll < 0.90:                     # clear speech
            x = {"energy": rng.uniform(0.65, 1.0), "periodic": True}
        else:                                 # ambiguous middle
            x = {"energy": rng.uniform(0.1, 0.6), "periodic": rng.random() > 0.5}
        casc.read(x)

    print("  where 1000 events were resolved (cheapest-first):")
    for name, n in sorted(casc.resolved_by.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<14} {n:>4}  ({n/10:.0f}%)")
    reached = casc.resolved_by.get("model", 0) + casc.resolved_by.get("unresolved", 0)
    print(f"  -> the expensive model ran on only ~{reached/10:.0f}% of events")
