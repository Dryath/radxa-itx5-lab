# 04 — The bandwidth wall (the boss fight)

If you take one thing from this repo, take this: **on the RK3588, LLM decode is bound by
LPDDR5 memory bandwidth, and that bandwidth is shared across every compute master.** TOPS
are a red herring. Bytes-per-token is the metric.

## The law

> **decode tok/s ≈ (~25 GB/s effective DDR ceiling) ÷ (active bytes read per token)**

That's it. Every result in this repo is downstream of this one relationship. A dense model
re-reads all its weights every token → slow. A sparse MoE reads only its active experts →
fast (if they fit in the IOMMU — [00](00-the-hardware.md), Wall #2). Want a decode-speed
estimate? Divide 25 GB/s by the bytes you have to move. You'll be close.

## The evidence

- **Effective wall-time bandwidth is 19–25 GB/s across every production shape** — about
  half of LPDDR5's ~50 GB/s theoretical peak. Peak measured **~25.2 GB/s at K=4096
  (39.7 ns/MB)**.
- **Not all K are equal.** K=6144 (the FFN down-projection) is the worst per-byte at
  **49.1 ns/MB, +24%** over the K=4096 sweet spot — which is exactly why it's the dimension
  that caps M-batching.
- **There's a cliff.** Empirically: `K ≤ 6144` is comfortable (M up to 32); `K = 7168–10240`
  is constrained (M drops 16 → 4); **`K ≥ 12288` is a dead zone that hangs even at M=1.**

## "Shared" is the cruel word

The ~25 GB/s is **per-master and shared**, not per-master and additive:

- One A76 core nearly saturates the bus by itself (a copy benchmark hits ~24.7 GB/s on one
  core).
- Running two masters at once doesn't stack: **CPU+NPU concurrent additivity ≈ 0.76×**,
  **CPU+GPU ≈ 0.54×**.
- Piling on more CPU cores *backfires*: **all-8-core decode is 2.4× slower than 4×A76**
  (the A55s add sync/bandwidth contention for no throughput).
- Two separate NPU contexts don't parallelize either — the NPU is a **single serial
  device** (2 contexts → 1.05×). You add throughput by batching (M), never by running
  things side by side.

## It's a starvation problem, not a compute problem

NPU silicon utilization sits at **0.17–0.75%** during real inference. The chip isn't
working hard; it's waiting — on memory, and on the CPU's blocking completion poll
([01](01-the-interface.md)). "Add more compute" is the wrong instinct on this platform
almost every time.

## The one nuance (so you don't over-rotate)

Decode isn't *100%* bandwidth-bound — there's real dequantization compute in the mix. A raw
memory-bandwidth microbench peaks at 2 threads (23.4 GB/s) and *drops* at 4 (19.9 GB/s), yet
decode throughput still rises with more threads (e.g. 6.76 → 7.86 → 8.08 tok/s at 2/3/4
threads on one MoE). Best read: **decode is ~60–70% bandwidth-bound**, with the rest being
per-byte dequant work. Bandwidth is the ceiling; it isn't the *only* thing.

## The conclusion nobody wanted

If one board's bandwidth is the wall, the only way to add *aggregate* bandwidth is **another
memory controller — i.e. a second physical board.** That's not a cop-out; it's what the
measurements keep pointing at, and for a while the lab ran a two-board cluster on exactly that
logic. Sometimes the honest optimization is "buy the second Rock 5."

**Epilogue: we went back to one board.** The cluster is the right answer *only* if you genuinely
need aggregate bandwidth across boards — and running two of them has its own costs (more hardware
to look after; in our case one board's run ended when **I dropped it** — operator error, full
stop, no fault of the board). For a single well-scoped workload, one tuned board is what you
actually want to run. The physics stands; the cluster just didn't earn its keep here.

---

Next: [05 — Kernel & tuning](05-kernel-and-tuning.md) — the config that unlocks the levers,
and the traps.
