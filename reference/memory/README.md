# Reference memory implementations (clean-room)

> **Clean-room note.** The code in this folder is a **fresh, generic, pattern-level
> reimplementation written by Claude** at the author's request — *not* extracted from any
> product or private engine. The design *conclusions* in [`docs/12`](../../docs/12-agent-memory.md)
> came from a real (private) system; the *code here* was written from those conclusions alone, so
> the patterns are usable and runnable without exposing that system's architecture. Take it and
> adapt it freely.

Why clean-room instead of a copy? The original implementations live inside a codebase whose
architecture is inseparable from a specific project — copying it verbatim would leak that project
even scrubbed. So these are written from the *idea*, generic on purpose. Same patterns, no
fingerprints.

## What's here (numpy only, each runs standalone: `python <file>.py`)

| File | Pattern (docs/12) |
|---|---|
| [`episodic_store.py`](episodic_store.py) | The **episodic** half — a bounded ring buffer of time-stamped events, each a low-D key, recalled by kNN-cosine, with **consolidation + age-based decay** (the nightly idle job) and a **versioned save** so the whole store rolls back. |
| [`recurring_identity.py`](recurring_identity.py) | The **semantic "who is this"** half — running centroids that start anonymous and **promote a recurring signal to 'known'** once seen enough; labels from ground truth when you have it. |

Together they're the **two-tier split** from [`docs/12`](../../docs/12-agent-memory.md): one store
that generalises across encounters (identity), one that binds specific episodes (what happened) —
because a single store measurably cannot do both. Neither needs a big model or a vector database;
both fit comfortably on a single board.
