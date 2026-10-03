#!/bin/bash
# hc_ab.sh: the multi-token hyper-connection read's norm one group per token and stream (bin/strata-q) against one
# per token (STRATA_HC_SPLIT=0): IQ3_S and IQ2_XS, MTP, 128 tokens on port_ids; two alternating rounds after a warm-up.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
NODUMP=1 IDS=port_ids.txt NEW=8 EXE="$B/strata-q" "$H/run.sh" hc-jit iq3_s --suffix-draft 0 --mtp "$H/../../mtp/rt" > /dev/null
for r in 1 2; do for m in iq3_s iq2_xs; do for arm in split one; do
  case $arm in split) E=();; one) E=(STRATA_HC_SPLIT=0);; esac
  env "${E[@]}" NODUMP=1 IDS=port_ids.txt NEW=128 EXE="$B/strata-q" "$H/run.sh" "hc-$m-$arm-$r" "$m" --suffix-draft 0 --mtp "$H/../../mtp/rt" > /dev/null
  echo "$r $m $arm $(grep -E '^decode  ' "$H/hc-$m-$arm-$r.txt" | grep -o '[0-9.]* tok/s')"
done; done; done
