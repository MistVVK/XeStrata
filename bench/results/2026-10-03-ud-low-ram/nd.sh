#!/bin/bash
# nd.sh: how often --mmap-experts in a 62 GiB cgroup gives other logits than cfg-1, with the adaptive VRAM tier on
# (the config's default) and off (--adapt-every 0); the off runs are compared with nd-off-1
H=$(cd "$(dirname "$0")" && pwd)
S=(systemd-run --user --scope -q -p MemoryMax=62G)
for i in 1 2 3 4 5 6; do
  "${S[@]}" python3 "$H/cfgrun.py" nd-on-$i --drop --resident-budget-gib=1 -- --mmap-experts
  "${S[@]}" python3 "$H/cfgrun.py" nd-off-$i --drop --resident-budget-gib=1 -- --mmap-experts --adapt-every 0
done
for i in 1 2 3 4 5 6; do
  echo "on-$i $(cmp -s nd-on-$i.logits cfg-1.logits && echo same || echo DIFFERS)  off-$i $(cmp -s nd-off-$i.logits nd-off-1.logits && echo same || echo DIFFERS)"
done
