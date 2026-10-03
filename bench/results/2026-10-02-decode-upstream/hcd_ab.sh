#!/bin/bash
# hcd_ab.sh: the multi-token hyper-connection read's down projection, one sub-group per row for every token
# (bin/strata-v), against one per (row, token) (STRATA_HC_DOWN_ROWS=0): IQ3_S and IQ2_XS, MTP, 128 tokens on
# port_ids; three alternating rounds after a warm-up.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
NODUMP=1 IDS=port_ids.txt NEW=8 EXE="$B/strata-v" "$H/run.sh" hcd-jit iq3_s --suffix-draft 0 --mtp "$H/../../mtp/rt" > /dev/null
for r in 1 2 3; do for m in iq3_s iq2_xs; do for arm in rows pairs; do
  case $arm in rows) E=();; pairs) E=(STRATA_HC_DOWN_ROWS=0);; esac
  env "${E[@]}" NODUMP=1 IDS=port_ids.txt NEW=128 EXE="$B/strata-v" "$H/run.sh" "hcd-$m-$arm-$r" "$m" --suffix-draft 0 --mtp "$H/../../mtp/rt" > /dev/null
  echo "$r $m $arm $(grep -E '^decode  ' "$H/hcd-$m-$arm-$r.txt" | grep -o '[0-9.]* tok/s')"
done; done; done
