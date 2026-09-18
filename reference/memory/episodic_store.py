"""
Episodic store — a clean-room reference implementation.

  >>> CLEAN-ROOM NOTE <<<
  This is NOT extracted from any product or private engine. It was written fresh and
  generic, from the design principles in docs/12 alone (by Claude, at the author's request),
  so the *pattern* is usable without exposing any codebase's architecture. Adapt freely.

The pattern (docs/12): an EPISODIC store is context-conditioned, time-stamped "what happened,
when" memory — deliberately NOT merged across encounters (that's the *semantic* store's job).
Cheap enough for a single board: a bounded ring buffer of events, each reduced to a
low-dimensional key, recalled by kNN-cosine, with a versioned save so the whole store can roll
back. No vector database required.

numpy only.
"""
from __future__ import annotations
import json, time, pathlib
import numpy as np

SCHEMA_VERSION = 1


class EpisodicStore:
    def __init__(self, dim: int, capacity: int = 4096):
        self.dim = dim
        self.capacity = capacity
        self.keys = np.zeros((0, dim), np.float32)   # unit-normalised low-D keys
        self.meta: list[dict] = []                   # parallel: {t, payload, hits, last_used}

    # ---- write ----------------------------------------------------------------
    def add(self, key: np.ndarray, payload: dict, t: float | None = None) -> None:
        """Store one episode. `key` is a low-D vector (an embedding, or any projection of the
        event); `payload` is the structured 'what happened' you'll want to recall by."""
        k = _unit(np.asarray(key, np.float32).reshape(-1))
        assert k.shape[0] == self.dim, f"key dim {k.shape[0]} != {self.dim}"
        row = {"t": time.time() if t is None else float(t),
               "payload": payload, "hits": 0, "last_used": 0.0}
        if len(self.meta) >= self.capacity:
            self._evict_one()
        self.keys = np.vstack([self.keys, k[None, :]])
        self.meta.append(row)

    # ---- read -----------------------------------------------------------------
    def recall(self, query: np.ndarray, k: int = 5) -> list[tuple[float, dict]]:
        """kNN-cosine recall. Returns [(score, payload), ...] best-first. Recall counts as a
        reference, which is what protects an episode from decay (see `consolidate`)."""
        if len(self.meta) == 0:
            return []
        q = _unit(np.asarray(query, np.float32).reshape(-1))
        sims = self.keys @ q                          # keys are unit vectors -> cosine
        idx = np.argsort(-sims)[:k]
        out = []
        now = time.time()
        for i in idx:
            self.meta[i]["hits"] += 1
            self.meta[i]["last_used"] = now
            out.append((float(sims[i]), self.meta[i]["payload"]))
        return out

    # ---- consolidation & decay (the nightly idle job, docs/12) -----------------
    def consolidate(self, semantic_fold=None, decay_after_days: float = 30.0) -> dict:
        """Run this in the board's idle window. Optionally hand each episode to a `semantic_fold`
        callback (that's where 'this specific bear' becomes part of the general 'bear' in the
        *other* store), then drop episodes that were never referenced and have gone cold.
        Returns a small report. Decay is by AGE of last use, not by hit-count alone — decaying
        purely by usage loses recent-vs-old structure (a measured caveat)."""
        now = time.time()
        cutoff = now - decay_after_days * 86400.0
        keep_keys, keep_meta, folded, dropped = [], [], 0, 0
        for i, m in enumerate(self.meta):
            if semantic_fold is not None:
                semantic_fold(self.keys[i], m["payload"]); folded += 1
            cold = m["hits"] == 0 and m["t"] < cutoff
            if cold:
                dropped += 1
            else:
                keep_keys.append(self.keys[i]); keep_meta.append(m)
        self.keys = np.vstack(keep_keys) if keep_keys else np.zeros((0, self.dim), np.float32)
        self.meta = keep_meta
        return {"folded": folded, "dropped": dropped, "remaining": len(self.meta)}

    def _evict_one(self) -> None:
        # capacity pressure: drop the coldest (fewest hits, then oldest last-use)
        i = min(range(len(self.meta)), key=lambda j: (self.meta[j]["hits"], self.meta[j]["last_used"]))
        self.keys = np.delete(self.keys, i, axis=0)
        self.meta.pop(i)

    # ---- versioned save / load (roll the whole store back) --------------------
    def save(self, path: str) -> None:
        p = pathlib.Path(path); p.mkdir(parents=True, exist_ok=True)
        np.save(p / "keys.npy", self.keys)
        (p / "store.json").write_text(json.dumps(
            {"schema": SCHEMA_VERSION, "dim": self.dim, "capacity": self.capacity,
             "meta": self.meta}, indent=2))

    @classmethod
    def load(cls, path: str) -> "EpisodicStore":
        p = pathlib.Path(path)
        d = json.loads((p / "store.json").read_text())
        assert d["schema"] == SCHEMA_VERSION, f"schema {d['schema']} != {SCHEMA_VERSION}"
        s = cls(d["dim"], d["capacity"])
        s.keys = np.load(p / "keys.npy")
        s.meta = d["meta"]
        return s


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    store = EpisodicStore(dim=64, capacity=1000)
    # log a few "events", each a random 64-D key + a structured payload
    for i in range(200):
        store.add(rng.standard_normal(64), {"event": f"thing #{i}", "where": "room-a"})
    q = store.keys[42] + 0.05 * rng.standard_normal(64)     # a noisy cue for episode 42
    for score, payload in store.recall(q, k=3):
        print(f"  {score:+.3f}  {payload}")
    print("consolidate:", store.consolidate(decay_after_days=0.0))  # force-decay the unreferenced
