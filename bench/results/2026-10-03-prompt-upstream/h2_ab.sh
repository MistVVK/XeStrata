#!/bin/bash
# h2_ab.sh: the first 1,500 tokens of the long prompt (one chunk) on bin/strata-h, streaming from 1024-token chunks
# with and without the fused products, against streaming from 2048.  Two alternating rounds (h_ab.sh warmed up).
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
pre() { grep -E '^prefill  ' "$H/$1.txt" | grep -o '[0-9.]* tok/s'; }
run() { local name=$1; shift
  env "$@" IDS=short_ids.txt NEW=24 PREFILL=auto EXE="$B/strata-h" "$H/run.sh" "$name" iq3_s --max-context 32768 --kv int8 --suffix-draft 0 > /dev/null; }
for r in 1 2; do for arm in s1024 s1024nofused s2048; do
  case $arm in s1024) E=(STRATA_PREFILL_STREAM_MIN=1024);; s1024nofused) E=(STRATA_PREFILL_STREAM_MIN=1024 STRATA_PREFILL_FUSED=0);;
    s2048) E=(STRATA_PREFILL_STREAM_MIN=2048);; esac
  run "h2-$arm-$r" STRATA_PREFILL_TIMING=1 "${E[@]}"
  echo "$r $arm $(pre "h2-$arm-$r")"
done; done
