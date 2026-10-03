#!/bin/bash
# ab64.sh: one more pair at full RAM, then setup's budget for a 64 GB PC (36 GiB with the 128K context's KV in RAM)
# against --mmap-experts, alternating, inside a 62 GiB memory cgroup (the engine and its page cache), cold OS cache
H=$(cd "$(dirname "$0")" && pwd); M=${MODELS:?the folder of the UD-Q4_K_XL shards}
S=(systemd-run --user --scope -q -p MemoryMax=62G)
python3 "$H/evict.py" "$M"/*.gguf; python3 "$H/cfgrun.py" cfg-3
python3 "$H/evict.py" "$M"/*.gguf; python3 "$H/cfgrun.py" mm-3 --drop --resident-budget-gib=1 -- --mmap-experts
for i in 1 2; do
  python3 "$H/evict.py" "$M"/*.gguf
  "${S[@]}" python3 "$H/cfgrun.py" c64-$i --drop --resident-budget-gib=1 -- --resident-budget-gib 36
  python3 "$H/evict.py" "$M"/*.gguf
  "${S[@]}" python3 "$H/cfgrun.py" m64-$i --drop --resident-budget-gib=1 -- --mmap-experts
done
