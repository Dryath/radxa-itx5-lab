# 12 — On-device agent memory, shaped by the hardware

An always-on agent on a single board has to remember things across sessions without a
datacentre behind it. That constraint — bounded compute, bounded storage, a device that idles
most of the day — turns out to *specify* the memory architecture rather than merely limit it.
These are the design conclusions that survived measurement. They're stated as general
patterns; there's no product here, just what the substrate forced.

## 1. Two-tier memory is not optional

The single most load-bearing finding: **one generalising store cannot also bind episodes.**
A store built to answer *"what do I know about X"* generalises across contexts by construction
— which is exactly why it *can't* also answer *"what happened in that specific session."* We
measured this directly on a recurrent substrate: predecessor/context identity did not change
what was retrieved at any depth tested (equivalent to no-effect within a tight margin). The
architecture built **semantic** memory; it supplied **zero episodic** binding. (An earlier
reading that suggested otherwise turned out to be *stream position* leaking in, not content —
and was withdrawn.)

So split them, deliberately:

| | answers | conditioned on | shape |
|---|---|---|---|
| **Semantic** | "who is this / what do I know about them" | nothing — it generalises | an entity model that merges across encounters |
| **Episodic** | "what happened, when" | the surround it was laid down in | time-stamped, context-keyed, *not* merged |

**Do not expect one vector store to do both.** A cross-session "person model" and a
"what-happened-in-that-meeting" recall are different machines with different write rules. Build
two.

## 2. Write by consequence, not by salience

What gets kept should be ranked by **outcome/consequence**, not by how *loud* or *surprising*
the moment was. Salience is a cheap proxy that keeps the wrong things (every loud irrelevance)
and drops the quiet ones that mattered later.

And treat "affect" — if you have any notion of it — as a **modulation on the learning rule**,
not a scoreboard variable. A `mood += 0.1` counter you could delete without changing behaviour
is decorative; a signal that actually changes *what and how strongly you write* is real. On a
constrained device this is also just good engineering: the write policy is your only defence
against the store filling with noise, because you can't afford to keep everything.

## 3. Consolidation and decay are a scheduled idle job

There isn't budget to fold new experience into long-term memory *as it arrives* — the
consolidation step is far more expensive than perception (on the substrate, roughly an
order of magnitude). So don't try. Instead:

- **Consolidate during idle.** An always-on board has an RTC and hours of nothing to do.
  A nightly *"fold the day's episodes into long-term memory"* run is a natural fit — batch the
  expensive work into the window the hardware hands you for free.
- **Let unreferenced traces decay.** Memory you never touch again should fade. Decay isn't
  forgetting-as-failure; it's the counterpart of consolidation that keeps the store's size
  bounded and its contents relevant. (Caveat from measurement: if you decay by *usage* alone,
  you lose any recent-vs-old structure — you may want an explicit age trace, not just a hit
  counter.)

This is the memory analogue of the whole repo's thesis: the board's idle RTC schedule is a
*feature of the silicon*, so build the expensive path to run in it.

## 4. Perception discipline: cheapest read first, label from ground truth

For any sensing pipeline (audio, vision, events), stack the reads by cost and run each at its
own cadence:

1. **Cheapest deterministic read first** — a threshold, a diff, a checksum, a heuristic. Most
   frames/events are handled here and never reach a model.
2. **Expensive neural read last, and rarely** — only when the cheap layers can't resolve it,
   and at a lower cadence.

And the trick that saves the most work: **label from ground truth you already have** instead of
hand-tagging. If a transcript or a calendar or a schedule already knows who/what/when, use *it*
as the label source rather than asking a model (or a human) to re-derive it.

## The patterns worth copying

The concrete shapes behind the above, all cheap enough for a single board:

- **Episodic store:** a ring buffer of events, each projected to a **low-dimensional vector**,
  with **kNN-cosine recall** and a **versioned save** so you can roll the whole store back. Tiny
  and fast; no vector database required.
- **Event snapshots:** capture a compact, structured record of *what happened* at the moment it
  happens (the fields you'll want to recall by), not a raw dump you re-parse later.
- **Promote-on-recurrence:** an identity model that starts anonymous and **promotes a recurring
  signal to "known"** once it's seen enough — the natural shape for recognising a returning
  person/voice/device without enrolling them up front.

None of this needs a big model. It needs the *right two stores*, a write rule with teeth, and
the discipline to do the expensive work while the board sleeps.

**Runnable code:** [`reference/memory/`](../reference/memory/) has clean-room implementations of
both halves — `episodic_store.py` (ring buffer + kNN recall + consolidation/decay + versioned
save) and `recurring_identity.py` (promote-on-recurrence). numpy only, each runs standalone. They
were written fresh from the principles above (not lifted from any engine — see the note there), so
the patterns are usable without any project baggage.

---

Next: [13 — The appliance](13-the-appliance.md) — where all of this actually runs.
