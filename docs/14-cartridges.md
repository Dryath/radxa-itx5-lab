# 14 — Cartridges: one base, many hot-swappable domains

A single big model that knows everything is the wrong shape for a constrained board — it's
large, static, and you pay for all of its knowledge on every token whether you need it or not.
An alternative that fits the hardware better: a small **base** model that holds general
*operators*, plus hot-swappable **cartridges** — per-domain LoRA adapters that supply the
specifics. **Operators, not operands.**

> Status: this is an architecture direction with experiments behind it, not a shipped
> benchmark. Treat it as a design pattern that fits the silicon, reported honestly as
> in-progress.

## The split

- **Base model** — learns the *general skills*: the operators, the reasoning shapes, the format
  discipline. It stays resident. It is deliberately kept small enough to live on the board with
  room to spare.
- **Cartridge** — a **LoRA adapter (tens of MB)** that specialises the base for one narrow
  domain. You load it when you enter that domain and unload it when you leave. The base's weights
  never change; the cartridge is the operand.

Because a cartridge is tiny, you can keep **many** of them and swap between them fast — the same
"hold the resident thing, page the cold thing" logic as the [appliance router](13-the-appliance.md),
one level down inside the model. The base is your working set; cartridges are the paged domains.

## Why it fits the board

- **You pay for domain knowledge only when you load it.** No carrying twelve domains' worth of
  weights to answer a question about one.
- **Adapters are cheap to train** on constrained hardware (or a short cloud burst) and cheap to
  ship — tens of MB, not tens of GB.
- **A base has a measurable capacity** for how much cartridge it can carry well. That's a number
  you measure per base, not a constant — bigger isn't automatically better, and the [method](../METHOD.md)
  applies: bake it off.

## The honest part

The general mechanism is clean; the hard questions are empirical and still open — how much a
given base can specialise before adapters interfere, how narrow a cartridge should be, and where
the deterministic-solver / symbolic side earns its place versus just prompting the base. Those
are measured one domain at a time. What's settled is the *shape*: on a small board, **a resident
base + hot-swappable adapters** beats one monolith you can't afford to keep loaded.

**Reference scaffold:** [`reference/cartridges/`](../reference/cartridges/) has a clean-room
`train_cartridge.py` (train one domain LoRA adapter, base untouched) and `hotswap.py` (hold the base
resident, swap adapters per request). It needs `peft`/`transformers`; it's the correct shape to
start from, and it ships **no domain data** — you bring your own corpus.

---

← Back to the [README](../README.md).
