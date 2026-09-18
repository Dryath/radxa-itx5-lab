"""
Recurring-identity model — a clean-room reference implementation.

  >>> CLEAN-ROOM NOTE <<<
  Written fresh and generic from the design principles in docs/12 (by Claude, at the author's
  request). NOT extracted from any product or private engine. It carries no codebase's
  architecture — just the pattern. Adapt freely.

The pattern (docs/12): a SEMANTIC "who is this" model that starts anonymous and **promotes a
recurring signal to 'known'** once it's been seen enough. This is the natural shape for
recognising a returning voice / face / device without enrolling it up front — a running
centroid per identity, matched by cosine, promoted on recurrence. Pairs with `episodic_store.py`
(the episodic half); here we generalise across encounters on purpose.

numpy only.
"""
from __future__ import annotations
import numpy as np


class RecurringIdentity:
    def __init__(self, dim: int, match_threshold: float = 0.80, promote_after: int = 3):
        self.dim = dim
        self.match_threshold = match_threshold      # cosine to count as "the same one"
        self.promote_after = promote_after          # sightings before anonymous -> known
        self.centroids = np.zeros((0, dim), np.float32)
        self.count = []                             # sightings per identity
        self.label = []                             # None until named; may be set on promotion

    def observe(self, vec: np.ndarray, ground_truth_label: str | None = None) -> dict:
        """See one signal. If a ground-truth label is available (a transcript, a calendar, a
        prior enrolment) pass it — labelling from ground truth you already have beats guessing
        (docs/12, perception discipline). Returns the resolved identity."""
        v = _unit(np.asarray(vec, np.float32).reshape(-1))
        if len(self.count) == 0:
            return self._new(v, ground_truth_label)
        sims = self.centroids @ v
        i = int(np.argmax(sims))
        if sims[i] >= self.match_threshold:
            # same identity: fold the new sighting into its running centroid
            n = self.count[i]
            self.centroids[i] = _unit((self.centroids[i] * n + v) / (n + 1))
            self.count[i] = n + 1
            if ground_truth_label and not self.label[i]:
                self.label[i] = ground_truth_label
            return self._view(i, float(sims[i]))
        return self._new(v, ground_truth_label)

    def _new(self, v: np.ndarray, label) -> dict:
        self.centroids = np.vstack([self.centroids, v[None, :]])
        self.count.append(1)
        self.label.append(label)
        return self._view(len(self.count) - 1, 1.0)

    def _view(self, i: int, score: float) -> dict:
        known = self.count[i] >= self.promote_after or self.label[i] is not None
        return {"id": i, "known": known, "sightings": self.count[i],
                "label": self.label[i], "score": score}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ident = RecurringIdentity(dim=48, promote_after=3)
    speaker_a = _unit(rng.standard_normal(48))     # two "real" speakers
    speaker_b = _unit(rng.standard_normal(48))
    stream = [speaker_a, speaker_b, speaker_a, speaker_a, speaker_b, speaker_a]
    for s in stream:
        noisy = _unit(s + 0.05 * rng.standard_normal(48))   # same speaker, small variation
        r = ident.observe(noisy)
        tag = "A" if r["id"] == 0 else "B"
        print(f"  speaker {tag}: id={r['id']}  known={r['known']}  seen={r['sightings']}  cos={r['score']:+.2f}")
    # -> two identities emerge; speaker A crosses promote_after=3 and becomes 'known'
