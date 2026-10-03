#!/bin/bash
# pf_ab.sh: the AVX-2 kernels' prefetch of the expert rows (upstream fa6310b) where the CPU's experts set the pace: the
# B70 made to look like a small card (small_check.sh's limits), IQ2_XS, a 96-token MTP answer; on and off
# (STRATA_IQ_PREFETCH=0), three alternating rounds.
H=$(cd "$(dirname "$0")" && pwd)
export STRATA_VRAM_LIMIT_MIB=8192 STRATA_MAX_ALLOC_MIB=4096 STRATA_NO_XMX=1
for r in 1 2 3; do for arm in on off; do
  case $arm in on) E=();; off) E=(STRATA_IQ_PREFETCH=0);; esac
  env "${E[@]}" NODUMP=1 EXE="$H/bin/strata-i" "$H/run.sh" "pf-$arm-$r" iq2_xs --suffix-draft 0 > /dev/null
  echo "$r $arm $(grep -E '^decode  ' "$H/pf-$arm-$r.txt" | grep -o '[0-9.]* tok/s') $(grep -o 'pool [0-9.]*' "$H/pf-$arm-$r.txt" | head -1)"
done; done
