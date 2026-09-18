# Reference implementations (clean-room)

Runnable, generic reference code for the patterns the docs describe. **Everything in `reference/`
is clean-room** — written fresh from the design conclusions (by Claude, at the author's request),
**not** extracted from any private engine. The conclusions came from a real system; the code here
was written from the conclusions alone, so the patterns are usable without exposing that system.
Each file repeats the note at the top. Take it and adapt it.

Most of it is **numpy-only and runs standalone** (`python <file>.py`); the cartridge scaffolds need
`transformers`/`peft` and are marked as reference-only (not run in CI here).

| Folder | What | Backs |
|---|---|---|
| [`memory/`](memory/) | `episodic_store.py` (ring buffer + kNN recall + consolidation/decay + versioned save) and `recurring_identity.py` (promote-on-recurrence). The two-tier semantic/episodic split. | [docs/12](../docs/12-agent-memory.md) |
| [`perception/`](perception/) | `cascade.py` — cheapest-deterministic-read-first, expensive-model-last, label-from-ground-truth. (Demo: the model runs on ~11% of events.) | [docs/12](../docs/12-agent-memory.md) |
| [`grafting/`](grafting/) | `associative_bank.py` — a minimal Hebbian + k-WTA cross-modal binding bank; cue one modality, evoke the others. (Demo: 88% cross-modal recall vs 2% chance.) | [docs/09](../docs/09-multimodal-grafting.md) |
| [`cartridges/`](cartridges/) | `train_cartridge.py` + `hotswap.py` — train a per-domain LoRA cartridge, hold one base resident and swap adapters. Reference scaffold (needs peft/transformers). | [docs/14](../docs/14-cartridges.md) |

> A reminder on the grafting one especially: docs/09 is an **unbuilt** research direction. The
> `associative_bank.py` here is a legibility toy for the *mechanism*, not a claim that the full
> thing is built. It cues one modality and recovers the others at recognition grade — exactly what
> docs/09 reports, and no more.
