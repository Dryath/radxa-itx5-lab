# THE METHOD — how this lab works

> **We fit the model to the hardware. We do not force the hardware to fit the model.**

This is the governing rule, and everything in this repo follows from it. It outranks
cleverness and it outranks any number in a datasheet.

Off-the-shelf models are a Kmart dress — cut for the median datacentre GPU, then taken in at
the seams to fit a $150 board. The alternative isn't a better sewing machine. It's to read
the silicon's actual preferences *off the silicon* and build *for* those.

## The rules

### 1. The hardware's constraints are the specification, not the obstacle
Choose the shapes, precisions, and access patterns the chip already wants. **If the silicon
says "not that axis / not that shape / not that precision," the answer is to stop asking — not
to engineer around it.** Nearly every result in this repo is an instance of this: MoE over
dense because decode is bytes-per-token ([04](docs/04-the-bandwidth-wall.md)); vision on the
NPU because a small dense tower fits the IOMMU ([08](docs/08-vision-and-heterogeneous.md));
`M=144` because that's where the CBUF budget peaks ([03](docs/03-going-fast.md)). We didn't
pick those; we measured them.

### 2. The goal is fixed. The route is the silicon's to determine.
The target doesn't move — *how you reach it is decided by measurement, not by design
preference.* This closes a tempting misreading of Rule 1: "let the silicon tell you what it
is" does **not** mean "whatever emerges is fine, move the goal to match." The goal stays put;
the substrate dictates the path.

**⇒ The test, when a mechanism isn't working:** *have I ruled out everything the measurement
can rule out — or am I reaching for a new mechanism because I prefer it?*

- **Substrate-led:** an output channel generalised zero. We didn't tune harder — we measured
  across a 50× sparsity range and two learning rates until *everything tunable was ruled
  out*, and only then concluded the parameterisation was wrong. The measurement diagnosed it;
  the fix was the consequence, not the hypothesis.
- **Forced (same day):** a mechanism was *wanted* for movement in a closed loop, so
  spike-frequency adaptation was imposed and swept. It made the loop monotonically worse. The
  push-back was information.

### 3. You can only read the silicon if you can tell its answer from your own noise
This is the dependency that makes Rules 1 and 2 executable. In one two-day stretch, **five
times a mechanism returned a clean null that was the *harness*, not the system.** Taken at
face value, every one of those would have read as "the silicon says no" when the silicon had
not spoken.

So: **prefer the measurement that could *remove* a candidate over the one that confirms a
favourite.** And watch the drift pattern — *the pull is toward building, and it's strongest
exactly when a result is disappointing.* **Three of those five harness bugs were a fix built
before establishing what needed fixing.** Measure first. Validate the metric before you let
it drive a decision.

---

The rest of this repo is what happens when you actually follow that. The
[graveyard](docs/06-the-graveyard.md) is where you can check that we did — every dead end has
the measurement that killed it.
