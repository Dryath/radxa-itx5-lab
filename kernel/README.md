# kernel — config that matters + a missing arm64 option

Full rationale is in [`docs/05`](../../docs/05-kernel-and-tuning.md). The short version:

## The options that make or break NPU work

| Option | Why |
|---|---|
| `CONFIG_ROCKCHIP_RKNPU_DRM_GEM=y` | **Mandatory.** DMA-heap mode makes every NPU allocation fail with `EINVAL`. |
| `CONFIG_ROCKCHIP_RKNPU_FENCE=y` (+ `SYNC_FILE`) | Enables a working `FENCE_OUT` fd → busy-poll instead of blocking IRQ wait. The main reason to run a custom kernel. |
| `PROC_FS` / `DEBUG_FS` | NPU load/monitoring visibility. |

Always `grep` the built `.config` afterwards — `FENCE`/`PROC_FS` like to silently revert to
`# not set` when a dependency is unmet.

## The patch

`patches/0002-arm64-Kconfig-add-CMDLINE_EXTEND-option.patch` re-adds the `CMDLINE_EXTEND`
Kconfig entry, which exists in the arm64 C implementation (`fdt.c` / `kaslr_early.c`) but
was never given a Kconfig knob. Apply against your kernel tree:

```bash
cd <kernel-source>
git apply /path/to/0002-arm64-Kconfig-add-CMDLINE_EXTEND-option.patch
```

> Kernel patches are, by nature, **GPLv2** — they modify GPL'd Linux. That's separate from
> the MIT license on the rest of this repo (see [`LICENSE`](../LICENSE)).
