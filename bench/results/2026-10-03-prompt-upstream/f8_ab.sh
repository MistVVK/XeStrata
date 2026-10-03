#!/bin/bash
# f8_ab.sh: the resident experts of at most 256 rows multiplied straight from their GGUF blocks (bin/strata-s)
# against every expert dequantized (STRATA_PREFILL_FUSED=0): the first 1,500 tokens of the long prompt, its first
# 6,133 (mid_ids), and all 26,292; IQ3_S, int8 KV, 24 tokens; two alternating rounds after a warm-up.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
NODUMP=1 IDS=short_ids.txt NEW=4 PREFILL=auto EXE="$B/strata-s" "$H/run.sh" f8-jit iq3_s --max-context 32768 --kv int8 --suffix-draft 0 > /dev/null
for r in 1 2; do for p in short mid long; do for arm in fused dq; do
  case $arm in fused) E=();; dq) E=(STRATA_PREFILL_FUSED=0);; esac
  env "${E[@]}" IDS="${p}_ids.txt" NEW=24 PREFILL=auto EXE="$B/strata-s" "$H/run.sh" "f8-$p-$arm-$r" iq3_s --max-context 32768 --kv int8 --suffix-draft 0 > /dev/null
  echo "$r $p $arm $(grep -E '^prefill  ' "$H/f8-$p-$arm-$r.txt" | grep -o '[0-9.]* tok/s') $(md5sum < "$H/f8-$p-$arm-$r.logits" | cut -c1-8)"
done; done; done
