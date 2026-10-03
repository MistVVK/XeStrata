#!/bin/bash
# i_ab.sh: what is kept of the prompt path's changes (bin/strata-i: F-1, F-2 and the GDN recurrence's pipelined loads)
# against stage C on the 26,292-token prompt; i also with STRATA_GDN_PIPELINE=0 (F-1 and F-2 alone).  Two alternating
# rounds after a warm-up.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
pre() { grep -E '^prefill  ' "$H/$1.txt" | grep -o '[0-9.]* tok/s'; }
run() { local name=$1 x=$2; shift 2
  env "$@" IDS=long_ids.txt NEW=24 PREFILL=auto EXE="$B/strata-$x" "$H/run.sh" "$name" iq3_s --max-context 32768 --kv int8 --suffix-draft 0 > /dev/null; }
NODUMP=1 IDS=port_ids.txt NEW=4 EXE="$B/strata-i" "$H/run.sh" i-jit iq3_s --suffix-draft 0 > /dev/null
for r in 1 2; do for arm in c i inopipe; do
  case $arm in c) X=c; E=();; i) X=i; E=();; inopipe) X=i; E=(STRATA_GDN_PIPELINE=0);; esac
  run "i-long-$arm-$r" "$X" "${E[@]}"
  echo "$r $arm $(pre "i-long-$arm-$r")"
done; done
