#!/bin/bash
# starts.sh: engine starts back to back, the arena as one registered mapping (default) against host USM per layer
# (STRATA_ARENA_HOST_USM=1), alternating, 40 each: IQ2_XS, 1 generated token of the 20-token prompt.  Counts the
# starts refused at the output head's upload, with the kernel's xe messages around each.
H=$(cd "$(dirname "$0")" && pwd)
export BUILD=xe IDS=port_ids.txt NEW=1 NODUMP=1 CACHE=10000
for r in $(seq 1 40); do
  for mode in mmap usm; do
    t0=$(date '+%Y-%m-%d %H:%M:%S')
    if [ $mode = usm ]; then STRATA_ARENA_HOST_USM=1 "$H/run.sh" "st-$mode-$r" iq2_xs > /dev/null
    else "$H/run.sh" "st-$mode-$r" iq2_xs > /dev/null; fi
    refused=$(grep -c 'bytes refused' "$H/st-$mode-$r.txt")
    vm=$(journalctl -k --since "$t0" --no-pager 2>/dev/null | grep -c 'VM worker error')
    echo "$mode $r refused=$refused vm_worker=$vm"
  done
done
