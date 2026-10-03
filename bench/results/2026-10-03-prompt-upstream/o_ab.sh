#!/bin/bash
# o_ab.sh: the QSA block scores one block a work-item (bin/strata-o) against the sub-group per block
# (STRATA_SCORES_WARP=1): the 26,292-token prompt, IQ3_S, int8 KV, 24 tokens, logits; two alternating rounds.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
NODUMP=1 IDS=port_ids.txt NEW=4 EXE="$B/strata-o" "$H/run.sh" o-jit iq3_s --suffix-draft 0 > /dev/null
for r in 1 2; do for arm in new warp; do
  case $arm in new) E=();; warp) E=(STRATA_SCORES_WARP=1);; esac
  env "${E[@]}" IDS=long_ids.txt NEW=24 PREFILL=auto EXE="$B/strata-o" "$H/run.sh" "o-$arm-$r" iq3_s --max-context 32768 --kv int8 --suffix-draft 0 > /dev/null
  echo "$r $arm $(grep -E '^prefill  ' "$H/o-$arm-$r.txt" | grep -o '[0-9.]* tok/s') $(md5sum < "$H/o-$arm-$r.logits" | cut -c1-8)"
done; done
