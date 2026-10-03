#!/bin/bash
# h_ab.sh: bin/strata-h (strata-g plus the GDN recurrence's pipelined loads) on the 26,292-token prompt, streaming from
# 2048-token chunks: c (stage C), h, h with STRATA_GDN_PIPELINE=0, with STRATA_PREFILL_FUSED=0, and that with
# STRATA_GROUP_COPY=1.  Then the first 1,500 tokens (one chunk) streaming from 1024 against 2048.  Two alternating
# rounds after a warm-up.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
pre() { grep -E '^prefill  ' "$H/$1.txt" | grep -o '[0-9.]* tok/s'; }
run() { local name=$1 x=$2 ids=$3; shift 3
  env "$@" IDS="$ids" NEW=24 PREFILL=auto EXE="$B/strata-$x" "$H/run.sh" "$name" iq3_s --max-context 32768 --kv int8 --suffix-draft 0 > /dev/null; }
NODUMP=1 run h-jit h port_ids.txt
for r in 1 2; do for arm in c h hnopipe hnofused hcopy; do
  S=STRATA_PREFILL_STREAM_MIN=2048
  case $arm in c) X=c; E=();; h) X=h; E=("$S");; hnopipe) X=h; E=("$S" STRATA_GDN_PIPELINE=0);;
    hnofused) X=h; E=("$S" STRATA_PREFILL_FUSED=0);; hcopy) X=h; E=("$S" STRATA_PREFILL_FUSED=0 STRATA_GROUP_COPY=1);; esac
  run "h-long-$arm-$r" "$X" long_ids.txt "${E[@]}"
  echo "$r $arm $(pre "h-long-$arm-$r")"
done; done
for r in 1 2; do for s in 1024 2048; do
  run "h-short-$s-$r" h short_ids.txt STRATA_PREFILL_STREAM_MIN=$s
  echo "$r short $s $(pre "h-short-$s-$r")"
done; done
