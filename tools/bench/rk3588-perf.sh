#!/bin/bash
# rk3588-perf.sh — pin RK3588 to max performance for LFM2 inference.
# Installed as a systemd oneshot (rk3588-perf.service) so it persists across reboots.
# Most important line for our bandwidth-bound decode: DMC -> performance (else DDR
# idles back to 534 MHz; we want it pinned at 2400). CPU/NPU/GPU performance help
# first-token latency + consistent benchmarks.
set -u

# CPU: every cpufreq policy -> performance (A55 cluster + A76 cluster)
for g in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
  [ -w "$g" ] && echo performance > "$g"
done

# DDR controller (the key one for decode bandwidth)
[ -w /sys/class/devfreq/dmc/governor ] && echo performance > /sys/class/devfreq/dmc/governor

# NPU (matters for prefill / NPU work)
for n in /sys/class/devfreq/*npu*/governor; do [ -w "$n" ] && echo performance > "$n"; done

# GPU (Mali) — harmless; helps if we ever use Vulkan
for gpu in /sys/class/devfreq/*gpu*/governor; do [ -w "$gpu" ] && echo performance > "$gpu"; done

# Transparent huge pages (helps the large weight mmaps)
[ -w /sys/kernel/mm/transparent_hugepage/enabled ] && echo always > /sys/kernel/mm/transparent_hugepage/enabled

# Report
echo "rk3588-perf applied:"
echo "  cpu(policy0)=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor 2>/dev/null) cpu(policy4)=$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor 2>/dev/null)"
echo "  dmc=$(cat /sys/class/devfreq/dmc/governor 2>/dev/null)@$(cat /sys/class/devfreq/dmc/cur_freq 2>/dev/null)"
echo "  npu=$(cat /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null) thp=$(grep -oE '\[[a-z]+\]' /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
