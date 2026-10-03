#!/bin/bash
# free_ab.sh: three builds of the same source, two alternating rounds:
#   icpx  - icpx (STRATA_NONFREE=ON, bin/strata-icpx), the XMX products
#   noxmx - the same binary with STRATA_NO_XMX=1 (the products on the vector engines)
#   free  - intel/llvm's DPC++ 6.2 (dpclang++, bin/strata-free), run without oneAPI's environment; its runtime
#           reports no XMX for the B70, so it takes the products without XMX
# IQ3_S with MTP (20-token prompt, 128 tokens), and the 26,292-token prompt (prefill auto, int8 KV, 24 tokens, logits).
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
run() { local v=$1; shift; case $v in free) NOONEAPI=1 "$H/run.sh" "$@" ;; noxmx) STRATA_NO_XMX=1 "$H/run.sh" "$@" ;; *) "$H/run.sh" "$@" ;; esac; }
exe() { case $1 in noxmx) echo "$B/strata-icpx" ;; *) echo "$B/strata-$1" ;; esac; }
for v in icpx free; do NODUMP=1 IDS=port_ids.txt NEW=4 EXE=$(exe $v) run $v fr-jit-$v iq3_s --suffix-draft 0 > /dev/null; done
for r in 1 2; do for v in icpx noxmx free; do
  for _ in 1 2 3; do NODUMP=1 IDS=port_ids.txt NEW=128 EXE=$(exe $v) run $v fr-mtp-$v-$r iq3_s --mtp "$H/../../mtp/rt" | grep -q "rc=0" && break; done
  for _ in 1 2 3; do IDS=long_ids.txt NEW=24 PREFILL=auto EXE=$(exe $v) run $v fr-long-$v-$r iq3_s --max-context 32768 --kv int8 --suffix-draft 0 | grep -q "rc=0" && break; done
  echo "$v $r decode $(grep -E '^decode  ' "$H/fr-mtp-$v-$r.txt" | grep -o '[0-9.]* tok/s') prefill $(grep -E '^prefill  ' "$H/fr-long-$v-$r.txt" | grep -o '[0-9.]* tok/s')"
done; done
