#!/bin/bash
# h1.sh: the expert arena as one mapping (default) against one host USM block per layer read by the GPU's kernels at
# PCIe share 0.15 (the best setting of records/2026-10-02-new-machine), on the current build: IQ2_XS and IQ3_S, MTP with
# suffix drafts off, fixed cache, 128 tokens, three rounds alternating
H=$(cd "$(dirname "$0")" && pwd)
export BUILD=xe IDS=port_ids.txt NEW=128 NODUMP=1 CACHE=10000
M=(--mtp "<data>/mtp/rt" --suffix-draft 0)
for r in 1 2 3; do
  for m in iq2_xs iq3_s; do
    "$H/run.sh" h1-$m-mmap-$r $m "${M[@]}"
    STRATA_ARENA_HOST_USM=1 "$H/run.sh" h1-$m-usm-0.15-$r $m "${M[@]}" --pcie-mode kernel --pcie-frac 0.15
  done
done
