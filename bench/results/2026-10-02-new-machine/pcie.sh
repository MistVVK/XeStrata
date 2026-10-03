#!/bin/bash
# pcie.sh: expert arena mmap (default) vs host USM per layer (STRATA_ARENA_HOST_USM=1) at several PCIe shares and modes,
# IQ2_XS and IQ3_S, MTP with suffix drafts off, fixed cache, 128 tokens, two rounds alternating (the conditions of
# records/2026-09-30-pcie-compare, on the Gen5 x16 link)
H=$(cd "$(dirname "$0")" && pwd)
export BUILD=xe-portable IDS=port_ids.txt NEW=128 NODUMP=1 CACHE=10000
M=(--mtp "<data>/mtp/rt" --suffix-draft 0)
for r in 1 2; do
  for m in iq2_xs iq3_s; do
    "$H/run.sh" pc-$m-mmap-$r $m "${M[@]}"
    for f in 0 0.15 0.30 0.55; do
      STRATA_ARENA_HOST_USM=1 "$H/run.sh" pc-$m-usm-kernel-$f-$r $m "${M[@]}" --pcie-mode kernel --pcie-frac $f
    done
    STRATA_ARENA_HOST_USM=1 "$H/run.sh" pc-$m-usm-direct-0.30-$r $m "${M[@]}" --pcie-mode direct --pcie-frac 0.30
    STRATA_ARENA_HOST_USM=1 "$H/run.sh" pc-$m-usm-dma-0.30-$r $m "${M[@]}" --pcie-mode dma --pcie-frac 0.30
  done
done
