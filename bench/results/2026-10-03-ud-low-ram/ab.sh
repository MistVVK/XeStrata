#!/bin/bash
# ab.sh: setup's UD config (resident budget) against the same config reading every expert from the files
# (--mmap-experts), alternating, each from a cold OS cache
H=$(cd "$(dirname "$0")" && pwd); M=${MODELS:?the folder of the UD-Q4_K_XL shards}
for i in 1 2; do
  python3 "$H/evict.py" "$M"/*.gguf; python3 "$H/cfgrun.py" cfg-$i
  python3 "$H/evict.py" "$M"/*.gguf; python3 "$H/cfgrun.py" mm-$i --drop --resident-budget-gib=1 -- --mmap-experts
done
