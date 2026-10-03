#!/bin/bash
# f2_ab.sh: the long prompt (26,292 tokens: chunks 8192 x 3 + 1716) with streaming from 2048-token chunks as in C,
# to separate the stream-1024 change from the others: c, f with STRATA_PREFILL_STREAM_MIN=2048, and that with
# STRATA_PREFILL_FUSED=0.  Two alternating rounds (an earlier run warmed the kernels).
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
pre() { grep -E '^prefill  ' "$H/$1.txt" | grep -o '[0-9.]* tok/s'; }
for r in 1 2; do for arm in c f2048 f2048nofused; do
  case $arm in c) X=c; E=();; f2048) X=f; E=(STRATA_PREFILL_STREAM_MIN=2048);;
    f2048nofused) X=f; E=(STRATA_PREFILL_STREAM_MIN=2048 STRATA_PREFILL_FUSED=0);; esac
  env "${E[@]}" IDS=long_ids.txt NEW=24 PREFILL=auto EXE="$B/strata-$X" "$H/run.sh" f2-long-$arm-$r iq3_s --max-context 32768 --kv int8 --suffix-draft 0 > /dev/null
  echo "$r $arm $(pre f2-long-$arm-$r) $(grep -o 'experts streamed [0-9]*' "$H/f2-long-$arm-$r.txt")"
done; done
