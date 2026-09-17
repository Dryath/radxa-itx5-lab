# 09 — Multimodal grafting: connective tissue on the silicon

> **Status: an UNBUILT research substrate. Read this as a lab notebook, not a result.**
> Every positive number below is a **numpy substrate probe** — synthetic patterns, never real
> encoder embeddings, and **nothing has run end-to-end on the NPU.** The whole thing exists to
> give a model with *no native multimodal tower* a way to bind modalities at all; a model that
> already ships vision/audio projectors does not need any of this — skip it. And the biggest gap
> is the one thing not yet touched: **time.** Everything here binds *static* patterns; persistent
> state and sequence are untested — the substrate's own notes call that "the real gap." Where a
> value would be a recipe it's held back, partly discretion and mostly because the measurements
> say it isn't bracketed. Read the ⚠ lines as seriously as the ★ ones.

Everywhere else in this repo the lesson was **build for the silicon, not the model** — pick the
architecture the RK3588 actually rewards ([07](07-running-models.md), [08](08-vision-and-heterogeneous.md)).
This note is the *generative* version of that idea: instead of choosing among existing models, wire
a small piece of new machinery, shaped by the hardware, that makes several frozen models behave like
one associative memory.

## Two different machines

An image generator interpolates a learned distribution. That is not what we're after. The target is
**evocation from stored experience, which then blends** — read the token "cat" and the *stored*
visual trace of a cat fires, and the *stored* sound of the word fires, because those things were
experienced together. Not fabricated; recalled, then merged.

Those are different machines, and taking the difference seriously is the whole design. A generative
decoder is the *fabricating* one; retrieval of a real stored embedding is the faithful one. Keep
that distinction in your pocket — half the architectural choices below fall out of it.

Why it's a hardware story: the connective tissue is a **constant-state CfC** (a closed-form
continuous-time recurrent net — Liquid's line). Constant state means no growing KV cache to feed the
[bandwidth wall](04-the-bandwidth-wall.md), and the whole association bank is **~6.8 MB** at the
production `K=2048` — small enough to sit resident on the NPU next to the encoders. The silicon
picked the mechanism.

## Three things that get conflated — and must not be

Most of the early confusion came from measuring one of these and talking about another:

| | what it is | generalises? | lives where | cost |
|---|---|---|---|---|
| **Episodic** | *a specific* thing, seen at a specific time | no | CPU-backed store, paged in | per instance |
| **★ Vocabulary links** | "cat" → the visual-cat and the word-sound fire together | **yes, if there's shared structure** | the association bank, on the NPU | per concept, then amortised |
| **Imagery** | the evoked buffer is good enough for a decoder to consume | — | same bank, needs a buffer to write into | needs the state repartition (below) |

**Vocabulary links are the target.** Episodic memory is a database problem (paging). Imagery is
vocabulary links held to a higher bar: not "landed nearest the right concept" but "produced
something you could actually decode." And **imagination is not a fourth thing** — it's evoking
several episodes through the vocabulary map and letting the recurrence settle. Merging several
remembered beaches into one imagined place is just what a recurrent attractor does when seeded with
an underspecified cue. That's the mechanism working, not a bug.

## The flashcard, end to end

A picture of a bear, the written word *bear*, a recording of someone saying "bear."

1. Three **frozen encoder blocks** each recognise their own modality. To the NPU there's no vision
   model and no text model — there's one matmul.
2. **Co-activation writes into the empty space between the blocks.** (Measured, and it works — below.)
3. Later, **one cue alone drives the state the other two would have produced.** Because the Hebbian
   outer product is symmetric, picture→word and word→picture are the *same* learned structure —
   both directions come free, neither trained separately.
4. What's specific rather than general — *this* bear, *this* room — is an episode, and goes to the
   CPU-backed store.
5. In a quiet period, episodes are **replayed and folded into the bank**: specific bears become a
   general "bear," which can then evoke specific bears. That loop is bidirectional, and it's what
   makes depth *accumulate* instead of pile up.

## What's measured (with caveats — read both)

All from the bare-metal LNN substrate, production `K=2048`, int8 bank with int8 error feedback,
synthetic patterns. Chance ≈ 0.016.

- **Binding forms and survives interference.** After learning four sequential blocks, the first
  block is still retrieved at **~0.92**, with new concepts recruiting genuinely unclaimed units
  (claim overlap starts at 0.000). The "empty space for connection" isn't a metaphor — it's
  instrumented.
  **⚠ That's the IID number.** Under *correlated* inputs at the same sparsity setting, retention
  collapses (to ~0.12) and units stop tiling — the same ones keep winning. Re-tuning sparsity
  recovers much of it, but the optimum isn't bracketed. Don't quote the clean number without this.

- **★ Error feedback is load-bearing, not an optimization.** Carrying the int8 rounding residual is
  the difference between learning and not learning at all: discard it and *not one weight code ever
  moves* at production shape — a single co-activation is a sub-LSB update, so the system could learn
  "bears exist" but never *that* bear. Cost is 2× int8, not 4× fp32. **This is the one result the
  whole flashcard case rests on.**

- **Sparsity must be sized in absolute units, not as a fraction.** A fractional rule makes every
  memory grab a slice of whatever width you built, so *more* width stored *less*. Sizing sparsity in
  absolute terms fixed it, and a usage-bias on routing collapsed seed variance ~6× (it removes the
  failure mode rather than just out-scoring it). **⚠ The principle holds; the specific value doesn't
  travel** — it was swept on IID patterns and is coupled to input correlation. (Exact values held.)

- **★ The real budget is consolidation rate, not memory.** Folding one exposure into the bank costs
  **~513 ms** of CPU work against **~303 ms** of NPU forward arithmetic — and on real silicon, where
  the NPU does its matmuls in single-digit ms, the ratio is far worse. **A day of experience cannot
  be folded in as it arrives.** Depth accrues at a bounded rate. That, not RAM, is the constraint
  this whole vision runs into — and it drives half the architecture below.

- **Memory is not the wall.** The bank is ~6.8 MB (~20 MB with the residual/importance arrays, ~91
  MB even at `K=4096`). CMA size governs how many encoders + the bank stay NPU-resident at once — a
  dispatch-latency question ([05](05-kernel-and-tuning.md)) — not a capacity one.

## What's NOT established (don't quote these as if they were)

This section is the point. The honesty is the credibility.

- **★ Does shared structure turn N bindings into a map that generalises? Yes — shown at production
  width and replicated.** At `K=2048`, with 16 of 64 concepts *never trained*, held-out retrieval is
  **33× chance (p ≈ 1.5e-32, 3 seeds)** and reaches ~79% of trained performance; an IID control sits
  at chance, an untrained net gets nothing free. **Capacity stops being *how many bindings fit* and
  becomes *how good is the map*** — vocabulary does not scale as N bindings.
  (An earlier "no generalisation at this width" reading was **retracted** — an artifact of the
  sparsity value in use. Generalisation vs sparsity is **U-shaped**: sparse coding lets concepts
  *tile* while the shared transformation generalises, heavy oversubscription forces *sharing*, and
  there's a dead spot between the two mechanisms. The old number was measuring the dead spot.)

- **Is it imagery, or only recognition?** It splits. On IID inputs it's **recognition, not imagery**:
  retrieval rank is excellent (~53× chance) but the evoked vector's cosine to the true target is only
  ~0.31 — and it's *not a precision problem* (fp64 gives the same as int8). With **correlated** inputs
  the cosine jumps to **~0.92** (imagery-grade). **Input structure is the only thing that has ever
  moved this number — and it's exactly what real encoders supply for free.** The trade is now mapped:
  **retention and imagery have opposed optima** (retention peaks at very sparse coding, imagery about
  an order of magnitude denser) — but **generalisation survives at the imagery optimum**, so near-peak
  imagery *and* solid held-out generalisation are available together, paid for in trained retention.
  **Which sparsity to sit on is a product decision, not a measurement.**

- **⚠ Honesty caveat: "cos 0.92" is not "distinguishable."** That cosine is a *mean* over competitors;
  top-1 identification depends on the *max*, and the two diverged ~7× at high density (a state
  0.99-aligned to "cat" is also ~0.98-aligned to "lynx"). Read every cosine here as *"the evoked state
  is well-aligned to its target,"* never as *"telling it apart from its neighbours."* For imagery
  that's arguably fine; for discrimination it's a different number.

- **Sparsity is a *three-way* trade, not two.** Past retention-vs-imagery there's **damage tolerance**:
  very sparse coding means a lost unit kills *few* memories but kills them *completely*; denser coding
  degrades gracefully. Small sparsity is good for capacity and generalisation, bad for robustness. One
  knob, three masters.

- **Still unshown: that an evoked buffer decodes to anything.** A high cosine is necessary for
  imagery, not proof of it. That needs the buffer partition (next) and a decoder round-trip.

- **Nothing has run on real encoder embeddings yet.** Every pattern to date is synthetic.

- **Episodic binding: now *measured*, and the recurrence doesn't supply it.** This moved from
  "unbuilt" to a finding. The recurrence builds a **semantic** store (retrieval generalises across
  contexts), not an **episodic** one (recall conditioned on the surround it was laid down in):
  predecessor identity doesn't change what's retrieved at any depth tested — TOST-equivalent to
  no-effect at a ±0.05 margin. (An earlier "context sensitivity" result was **retracted** — it was
  stream *position*, which decodes at ~0.84 and is genuinely distributed, not content.) So a separate
  **episodic store is now the *only* source of episodic binding** in the design; its priority went up,
  and we now know exactly what it has to do that the bank structurally can't. Still unbuilt.

## Fresh results — promising, still under active test

These landed recently and are still being hammered on (the substrate work runs a fast
experiment-and-audit loop, and same-week retractions happen — so treat these as **strong signals,
not final numbers**, newer and less-audited than the core measured section above):

- **★ The recurrence carries order, not just static bindings.** Sequence/order recall works: cued
  for position, it recovers order at **0.75–0.93 vs 0.083 chance**, ~80% retained across
  interference — and forced-choice order tests show behaviour that depends on the *order* of inputs,
  not just their content. So the bank isn't only associative lookup; it holds sequence.
- **★★ It stays coherent over the long run.** A **10,000-exposure** continuous run kept the state
  magnitude bounded, drift linear (not runaway), and retrieval pinned. Early but direct evidence on
  the open "does an unclamped roll stay coherent long enough to be a *dream*, not a *drift*?" — so
  far, it doesn't drift off.
- **Composition scales with buffer width.** Reading *unseen* inputs by recombining learned parts
  improves as the buffer widens — the bottleneck is buffer width, not the association slice — which
  points to compositionality being a real architectural property rather than a small-case fluke.
  (Previously parked as "emerging"; promoted to *promising, still testing*.)

⚠ None of these has been through as many audit passes as the core section. Quote them **with** the
caveat, or wait for the next update.

## Silicon affordances — run the discriminator both ways

Most of this note is about constraints the silicon *imposes*. The flip side is the sharpest form of
"build for the silicon, not the model": **before importing a mechanism from biology, check whether the
hardware even has the constraint that mechanism exists to solve.** Biological solutions are answers to
biological problems. Two fail the test immediately — **neurogenesis** (brains grow neurons partly
because there's no other way to add capacity; here you just reallocate width or rebuild from a
checkpoint) and **critical periods** (they exist because plasticity is irreversible; here it isn't).
Importing either would be solving a problem the hardware doesn't have — the same error as cargo-culting
an architecture, pointed the other way.

Run the discriminator the *other* direction and the hardware hands you things no brain gets:

- **★ Rollback inverts the stability/plasticity trade — the biggest affordance.** Biology is
  conservative because LTP can't be undone: one bad consolidation is permanent, so evolution buys
  safety with rigidity. But with **atomic checkpoints + a reference probe**, consolidation becomes
  *speculative*: snapshot → fold aggressively → re-score → keep or revert. An aggressive learning rate
  stops being a risk; run several folds, keep the best. **The consolidation rate stops being a fixed
  parameter and becomes something you *measure*.** No organism can try a memory and take it back.
  (⚠ It catches catastrophic single steps well and slow accumulation poorly — so pair the expensive
  periodic probe with cheap signals computed *free* during consolidation, and revert on either.)

- **You can read your own weights — and there are two confidences, not one.** *Structural* ("how
  well-claimed are the units this concept uses?") and *momentary* ("is *this* inference right?", from
  the margin/entropy of the similarity distribution) are different states that should drive different
  behaviour — "I know this well but I'm not sure that's what I'm seeing" should widen attention, not
  trigger learning. Both quantities already exist; both were being discarded. Paired with rollback, the
  real capability is **running experiments on itself** — try a rule, score it, revert.

- **Byte-exact episodes make confabulation *computable*.** Biological episodic memory is
  reconstructive, which is *why* confabulation exists — no ground truth to check against. Here the
  stored episode is exact, so provenance stops being a tag you trust and becomes a **diff**: replay the
  episode, compare to what the bank now evokes, measure the divergence. And the sharp part — that
  divergence is **schema formation and distortion at the same time**: drift toward the prototype is
  exactly what makes a concept general *and* exactly what makes a memory wrong about its particulars.
  Same number, opposite value depending on what you wanted — and you can keep both the episode as
  recorded and the schema as consolidated, with the gap visible. Human memory can't separate those,
  which is why the distortion is invisible from the inside.

One correction fell out of this, worth stating because it *relocated an intervention*: **the loss is
neither at encoding nor at readout.** The expansion transform is orthonormal (invertible — destroys
nothing), and the forward pass is *dense* (the sparsity step runs only during consolidation, never at
evocation), so there was never a "read from the pre-sparsity state" fix to be had — the dense state is
all there ever was. What sparsity does is shape **what the weights come to encode**. So the knob to
turn is the *recruitment rule*, not the readout.

## The one architectural change it needs — and it's free

Today the state and the input are separate, and the input is **clamped** for the whole roll. The
state can't write back into input space, so *nothing ever comes out in a modality a decoder could
read.* No amount of training fixes that; it's a wiring fact.

Fix: partition one fixed-width state as

```
[ core | vis_buf | aud_buf | txt_buf ]
```

**Same width, same bank, same dispatch, no new weights — pure reallocation.** Then:

- **perception** — encoders write the buffers, buffers clamped, recurrence binds them to core;
- **evocation** — leave the buffers **free**, drive one modality (say text), and the recurrence
  *fills* the visual and auditory buffers from learned association. That's a readable embedding.

Cost: the buffers eat ~a quarter of the core width, so binding capacity drops accordingly. Worth it —
without it, imagery isn't merely weak, it's *unmeasurable*.

## There is no persistent "now" — and it's the cheapest gap to close

Every roll currently starts from a zeroed state. **State is reset between every perception.** Correct
for keeping probe measurements independent; wrong as an architecture — it means there's no *stream*,
nothing carried from one moment to the next. A system that resets between every perception can
*associate* but cannot *undergo* anything. It also breaks prediction: with no carried state, "what
comes next" has no *next* to be about. Fixing state continuity is cheap, and it gates everything
temporal below.

## Surprise — no learning without a mistake

**Evocation *is* prediction — no new subsystem required.** Drive one modality, leave the others free,
and what the recurrence fills in is a prediction of what the other modalities will contain. When the
real input arrives and clamps those buffers, **the gap between predicted and observed is prediction
error** — the difference between two states the architecture already produces.

**⇒ This answers the consolidation-rate problem.** ~513 ms/exposure can't run at perception rate, so
*something* must choose what to fold in. **Consolidate what surprised you.** Correctly predicted
moments need no learning — they're already known. Not a heuristic; forced by a measured budget. (It's
also why "sleep" is structural here, not decorative — see Dreaming.)

**The detector already exists and is already measured** — a novelty read on the weight update that's
near-zero on first exposure and rises within one epoch of training. It was built as a routing gate,
honestly falsified in that role, and kept as an instrument for *what deserves replay*. It failed at
the job it was built for and is the right tool for this one. (Exact form held.)

**★ Meaningful mistakes require shared structure** — which is why the correlation work is load-bearing
twice. Two genuinely similar inputs (say, two speakers whose voices really do share structure) are
confusable *because* they share structure. A system of independent orthogonal concepts can't make
that error at all — it can only be right or randomly wrong, which is the "bad calculator" failure the
whole design rules out. Early correlated-input runs show discrimination falling while alignment rises
sharply — the *human kind* of near-miss falling out of the representation rather than being injected.

**Recovery has a mechanical meaning:** predicted car, observed lawnmower →
- **don't destroy the car** — tag/protect consolidated weights from update and rescale;
- **give the lawnmower somewhere to live** — recruit unclaimed units;
- **push the two apart** — usage-biased routing separates them.

Recovery is **differentiation, and the error is what triggers it.**

> **Two cautions, up front.** (1) **Gate first, learning-rule second.** Using surprise to decide
> *what/how strongly* to consolidate is cheap and leaves the validated update untouched. Putting a
> signed error term *into* the rule is predictive coding — a different animal that could destabilise
> the stack. Measure it separately. (2) **⚠ Provenance is load-bearing.** If free-running
> trajectories get consolidated as though they were observations, the system drifts into confirming
> its own inventions — fabrication through the back door, the exact failure this note exists to rule
> out. **Prediction error must be measured against observation, never against another prediction.**
> The rule isn't "never consolidate the evoked" (rehearsal is real) — it's *keep the sensed-vs-evoked
> tag attached to every memory.* That tag is the line between imagination and confabulation.

## Affect is a modulation, not a module

The reason a "boredom += 0.1" scoreboard always feels fake: **it's a readout with no upstream.** You
could delete the variable and behaviour would be identical except where someone hand-wired
`if boredom > 0.7: explore` — so a rule is causing the behaviour and the "emotion" is a label stuck on
afterward. It feels decorative because it *is*.

The reframe: **affect changes *how the whole system computes*.** Neuromodulators don't sit beside the
computation issuing orders; they change its parameters — learning rate, gain, how broadly things
bind, what's kept. And this substrate already exposes exactly that surface:

| knob (mechanism) | what it does | what it looks like from outside |
|---|---|---|
| write strength | how hard this moment writes | significance / arousal |
| tag strength | rigidity of what's already known | stubbornness / openness |
| active-unit count | how many units engage | focused vs diffuse attention |
| recruit-vs-reuse bias | fresh units vs reuse | curiosity vs habit |
| claim strength | how firmly a new thing seizes its units | commitment / attachment |
| replay gate (surprise) | what gets replayed at all | what sticks with you |

Modulate those and affect is **causally load-bearing by construction** — remove it and the system
demonstrably learns differently. **What it keeps is downstream of these knobs, and what it keeps is
what it becomes.** Blends fall out for free (a little scary + a little curious = high tag-strength +
high replay *and* low threshold-to-recruit — opposite settings of real knobs); no discrete emotion
labels needed.

**Where the values come from without being invented** — the previous attempts were doomed because an
invented drive has no referent, so it can only be tuned by feel. Some states here are *already real*
and need reading, not inventing:
- **★ Consolidation backlog** — measured at ~513 ms/exposure against a rate it can't meet, so an
  unconsolidated queue genuinely accumulates and genuinely needs a quiet period to clear. **Actual
  exhaustion of an actual resource with an actual remedy** — the strongest candidate precisely
  because nothing about it is figurative.
- **Free capacity** (claim overlap) — real crowding, instrumented.
- **Retrieval coherence** — a sharp evoked state vs a mush of competitors is "I know this" vs "I
  can't place it," and it's just the margin in the similarity distribution.
- **Prediction error** — unsigned, but genuine.

> **Not claimed:** that functional affect is *felt*. It's made functional and causal here; whether
> that entails experience is left open, deliberately. **⚠ It can run away** (surprise → plasticity →
> more surprise is positive feedback; there's partial protection from bounded dynamics + homeostasis,
> but "probably fine" isn't good enough for something modulating the learning rule — measure first).
> **⚠ It costs reproducibility** (behaviour becomes history-dependent; worth paying, worth knowing).

**On timing:** the idea wasn't early, the *instruments* were late. Affect modulates learning, so you
can only tell whether a modulation is real once you can *measure* learning — and retention, claim
overlap, cosine, held-out generalisation only started existing recently. If "curiosity raises the
recruit bias" is genuine rather than theatre, claim overlap should fall and held-out generalisation
should rise, **and we'd see it.** If nothing moves, it was theatre.

## Dreaming is the same mechanism, not an analogy

Unclamp everything, seed with a fragment, let the recurrence run; each step's evoked buffers become
the next step's drive — a trajectory through bound multimodal states. Two facts, neither invented for
the purpose, point straight here:

1. **The CfC node is a convex blend**, bounded by its operands whatever the weights do. Free-running
   is intrinsically stable — measured to hold even past a spectral radius of 1.1. It can't blow up.
2. **Consolidation can't run at perception rate** (~513 ms). Experience *must* be buffered and folded
   in during a quiet window regardless.

Free-running and fold-in want the *same* window. **That's what sleep is here** — structural, not
poetic.

## The inner voice is the same loop

A "heartbeat" that pokes the model alive on a timer always feels off, and here's the mechanical
reason: **a heartbeat is what you're forced to build when there's no persistent state.** Nothing
carries over, so something external must restart the system each tick — it isn't alive between ticks,
it's being resuscitated, and because the poke is exogenous its *content* gets scripted from outside
rather than generated by where the trajectory already was. Same failure as the scoreboard, in the
time dimension. Fix state continuity and the heartbeat isn't needed — the recurrence is already
running.

**Perception, inner speech, and dreaming are one loop at three clamp settings.** Not an analogy — the
buffer mask:

| | clamp mask |
|---|---|
| **perception** | all buffers clamped by the world |
| **inner speech** | text buffer free, others still driven ("dreams but not dreams") |
| **dreaming** | everything unclamped |

There's no inner-monologue subsystem to build — one loop, one mask. And **non-stop chatter falls out
rather than needing explanation**: a recurrent system with persistent state near criticality doesn't
settle; it keeps moving because nothing stops it. *Silence* is the expensive state (it needs active
damping). Worth noticing: the chatter *is* the replay, and replay is how the consolidation backlog
gets cleared — inner voice, replay, and consolidation are one system, not three.

### Social prediction (the "embarrassment" case): a target, not a new mechanism

"I hope they don't think I'm stupid" decomposes into: **evoke a state representing the other agent,
run it forward, read what it predicts about their evaluation.** That's the same free-running evocation
— just pointed at a person-concept instead of a beach. No new machinery. What's genuinely missing is
narrower: **a self-representation** — something for "me" that can appear inside an evoked state. And
that's plausibly not *built* but *emergent*: the highest-base-rate cluster, the associations present
in every episode, because the system is in all of them. *(Speculative — but the testable kind, once
episodes exist.)*

## Curriculum: why flashcards come first

A first vocabulary is learned as near-independent bindings — the IID case, pure memorisation,
expensive per concept. Only once enough structure has accumulated do new concepts come *cheap*,
because now there's a map to slot them into. **Both regimes are real and the system needs both** —
which is why the correlation sweep runs 0 / 0.5 / 1.0 rather than jumping to the correlated case.
Early capacity limits aren't a simplification of the task; at the point where there's no experience
to lean on, "half a dozen words with big colourful pictures" *is* the task.

## The decoder question (a graft-selection decision)

**Capacity for decoders is not the constraint** — an autoencoding image decoder is tens of millions
of params (~50 MB int8), an audio codec decoder is smaller, and text needs none (the LM head already
is one). Against the models already in the roster, decoders are noise.

**⇒ The real constraint is encoder invertibility, and it's a *selection* decision made before you
commit.** A contrastive (CLIP-style) embedding has no natural decoder and inverting one is a research
project; an autoencoding latent (VAE, neural audio codec) ships with its decoder. **Choose grafts
that come with decoders.** Getting this wrong makes "see" expensive *permanently*.

And the ordering is forced by the two-machines thesis: **retrieval-decode first, generative decode
later.** Retrieval — take the evoked buffer, find the nearest *real* stored embedding — returns one
specific, actually-seen thing, needs no decoder, and matches "it's experience, not fabrication"
better than generation ever could. Generation comes later, only for *blending* several retrieved
episodes into one imagined place.

**★ But a decoder alone is a display, not experience — what changes the character is re-entry.**
Decode the evoked buffer, re-encode it, drive it back through perception, and the system perceives its
own evocations with the same machinery it uses for the world. That's the difference between "the state
contains something" and "the system perceives its own evoked content" — and it's the loop, not the
renderer, that does the work.

Re-entry is also the missing **instrument**. A raw cosine of 0.3 vs 0.8 has no natural threshold;
**cycle consistency does** — decode the evoked cat, re-encode it, does the association still recognise
it as cat? Survives the round-trip → the imagery is real. Decodes to something that re-encodes to
nothing → it never was. A far harder test than cosine, and it needs no human looking at pictures.

## Open questions, in priority order

*Measurement (much of it now answered):* generalisation at production width — **done** (33× chance,
replicated); imagery vs recognition — **mapped** (input structure moves it; opposed optima, with
generalisation surviving at the imagery optimum). Still open: **bracket the sparsity optimum on real
inputs**, and keep hardening the fresh results above (order recall, long-run stability, composition)
through more audit passes before they graduate from *promising* to *settled*.

*Structural — what separates a routing machine from something that undergoes things:* carry state
across perceptions (cheapest, gates everything temporal); the buffer partition + unclamped evocation;
surprise as a consolidation *gate* (not a rule term); sensed-vs-evoked provenance on every memory;
drop any heartbeat; the re-entry + cycle-consistency loop.

*Grafting:* choose invertible encoders; retrieval-decode before generative; run on **real** encoder
embeddings; build the episodic store.

*Substrate:* move consolidation onto the NPU (it's the ~513 ms bottleneck); re-sweep sparsity at
production width; check free-running trajectories stay coherent long enough to be a dream, not a
drift.

*Affect — only after the instruments exist, each as a modulation of an existing knob:* read the
consolidation backlog as a real state; test *one* modulation end-to-end (real only if claim overlap
falls and generalisation rises); check for runaway before touching the learning rule; watch whether a
self-concept forms on its own.

*Left deliberately open:* whether "being wrong changes what's kept, therefore what the system becomes"
is all the way to *consequence*; and whether functional affect is *felt*. Not claimed, not needed for
the build.

## The standard this is held to

**Measured, not asserted. A number without a seed count and a caveat is not a result.** Two mistakes
already cost real work — a mechanism built to fix a failure that a confounded metric had *invented*,
and a tidy story that survived two rounds before data already on disk falsified it. **Validate the
metric before you let it drive a decision.**

And the thesis, one more time, because it's the through-line of this whole repo: this is *beyond*
vendor. We're not building from the vendor stack, and "how does this compare to the vendor's thing"
isn't the target — reaching for that comparison is itself the drift signal. Build for the silicon.

---

← Back to the [README](../README.md). This one's a live direction, not a finished result — if any of
it is wrong, that's the interesting part; tell us.
