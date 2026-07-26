# 07 — So you actually want to run a model

You came for "how fast can this board run an LLM," not register dumps. Here's the short,
opinionated, measured answer so you can start from a good place instead of our worst ones.

## Rule zero: run a MoE, not a dense model

This falls straight out of the [bandwidth law](04-the-bandwidth-wall.md) — decode speed is
set by **bytes read per token**, and a sparse MoE only reads its *active* experts. The data
is stark (all Q4, 4×A76, tuned):

| model | type | total B | **active B** | prefill t/s | **decode t/s** |
|---|---|---:|---:|---:|---:|
| LFM2-8B-A1B | MoE | 8.3 | 1.0 | 40 | **19.5** |
| Qwen3-8B | dense | 8.2 | 8.2 | 11 | **4.7** |
| LFM2-24B-A2B | MoE | 24 | 2.0 | 26 | **14.6** |
| Qwen3.5-2B | dense | 2.0 | 2.0 | — | 14.2 |

Read those rows twice:

- **Same-size, MoE vs dense:** LFM2-8B-A1B decodes **4.2× faster** than Qwen3-8B dense —
  identical total size, but the MoE moves 1 GB/token instead of 8.
- **The MoE magic trick:** LFM2-24B-A2B (2 B active) decodes at **14.6 t/s — the same as a
  2 B *dense* model — while carrying 12× the parameters.** You get the capability of a 24 B
  model at the decode cost of a 2 B one. That's the whole reason to be here.

Decode speed depends on **active bytes per token and nothing else** — not the architecture
family. Sort your candidate models by active params; that's your decode-speed ranking.

## Rule one: uniform Q4, never "UD"/dynamic quant

Same Qwen3.6-35B-A3B: unsloth's **UD**-Q4_K_M decoded at **6.45 t/s**; the plain **standard
Q4_K_M** hit **8.0 t/s (+25%)** — because the dynamic quant bumps the hot tensors to more
bits, i.e. more bytes/token on a bandwidth-bound board. On this hardware, **always pick
uniform `Q4_0` or `Q4_K_M`.** And between those two, **`Q4_0` wins both axes** on the A76
(the K-quant unpack costs CPU cycles you don't have).

## The deploy recipe (what actually gets you the numbers)

```bash
# 1. Governors → performance (make it persistent; a boot service is ideal —
#    otherwise they revert to ondemand on reboot and you silently lose ~throughput)
for p in /sys/devices/system/cpu/cpufreq/policy*; do echo performance | sudo tee $p/scaling_governor; done
echo performance | sudo tee /sys/class/devfreq/dmc/governor   # DDR — the important one

# 2. Pin to the 4 A76 cores, 4 threads, Flash-Attention on, Q4_0
taskset -c 4-7 ./llama-cli -m LFM2.5-8B-A1B-Q4_0.gguf -t 4 -fa 1  [your prompt/opts]
```

Build the runtime with `-mcpu=cortex-a76+dotprod`. Do **not** let it spread to all 8 cores
(the A55s make it *slower* — [04](04-the-bandwidth-wall.md)) and do **not** reach for
`isolcpus` without reading the landmine in [05](05-kernel-and-tuning.md).

## What the ceiling looks like

A tuned **LFM2.5-8B-A1B Q4_0** on 4×A76 lands around **~60 prefill / ~20 decode t/s** — and
that's essentially the silicon's ceiling for this class of model. Bigger MoEs scale by
active bytes (24B-A2B ≈ 14–15 decode; a 35B-A3B ≈ 8). A 35B-A3B will even hold its **full
262K-token context in 32 GB** (KV is only ~20 KiB/token on hybrid attention) — memory isn't
the limit there, *prefill time* is, so keep interactive context in the 16–32K range.

## "But the NPU?"

For these MoE LLMs, the tuned **CPU path is the ceiling** — the experts are too big to stay
resident in the NPU's ~2.75 GB IOMMU window ([00](00-the-hardware.md)), so putting them on
the NPU means cold-staging every token and losing. The NPU earns its keep on a *different*
workload — dense vision — and the two can run at the same time. That's [08](08-vision-and-heterogeneous.md).
