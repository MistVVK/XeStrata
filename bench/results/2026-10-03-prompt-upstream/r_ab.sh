#!/bin/bash
# r_ab.sh: --prefill auto up to 32768 (bin/strata-n) against the old ceiling (STRATA_PREFILL_AUTO_MAX=8192): the
# 26,292-token prompt and its first 12,000 tokens, IQ3_S, int8 KV, 24 tokens; two alternating rounds after a warm-up.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
pre() { grep -E '^prefill  ' "$H/$1.txt" | grep -o '[0-9.]* tok/s'; }
NODUMP=1 IDS=port_ids.txt NEW=4 EXE="$B/strata-n" "$H/run.sh" r-jit iq3_s --suffix-draft 0 > /dev/null
for r in 1 2; do for p in long l12k; do for arm in new old; do
  case $arm in new) E=();; old) E=(STRATA_PREFILL_AUTO_MAX=8192);; esac
  env "${E[@]}" IDS="${p}_ids.txt" NEW=24 PREFILL=auto EXE="$B/strata-n" "$H/run.sh" "r-$p-$arm-$r" iq3_s --max-context 32768 --kv int8 --suffix-draft 0 > /dev/null
  f=$H/r-$p-$arm-$r.txt
  echo "$r $p $arm $(pre "r-$p-$arm-$r") $(grep -o 'chunk auto: [0-9]*' "$f") $(grep -o 'experts streamed [0-9]*' "$f") $(grep -o 'refilled in [0-9.]* ms' "$f") $(md5sum < "$H/r-$p-$arm-$r.logits" | cut -c1-8)"
done; done; done
