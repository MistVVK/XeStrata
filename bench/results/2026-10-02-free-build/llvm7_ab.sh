#!/bin/bash
# llvm7_ab.sh: the engine built with icpx (bin/strata-icpx) against the same source built with intel/llvm 7.1.1 built
# from source (bin/strata-llvm7, run with its libsycl and without oneAPI's environment), two alternating rounds:
# IQ3_S with MTP (20-token prompt, 128 tokens) and the 26,292-token prompt (prefill auto, int8 KV, 24 tokens, logits).
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin; I="<repo>/.tools/intel-llvm/install"   # the repository
export CACHE=10000
run() { local v=$1; shift; if [ "$v" = llvm7 ]; then NOONEAPI=1 LD_LIBRARY_PATH=$I/lib "$H/run.sh" "$@"; else "$H/run.sh" "$@"; fi; }
for v in icpx llvm7; do NODUMP=1 IDS=port_ids.txt NEW=4 EXE=$B/strata-$v run $v l7-jit-$v iq3_s --suffix-draft 0 > /dev/null; done
for r in 1 2; do for v in icpx llvm7; do
  for _ in 1 2 3; do NODUMP=1 IDS=port_ids.txt NEW=128 EXE=$B/strata-$v run $v l7-mtp-$v-$r iq3_s --mtp "$H/../../mtp/rt" | grep -q "rc=0" && break; done
  for _ in 1 2 3; do IDS=long_ids.txt NEW=24 PREFILL=auto EXE=$B/strata-$v run $v l7-long-$v-$r iq3_s --max-context 32768 --kv int8 --suffix-draft 0 | grep -q "rc=0" && break; done
  echo "$v $r decode $(grep -E '^decode  ' "$H/l7-mtp-$v-$r.txt" | grep -o '[0-9.]* tok/s') prefill $(grep -E '^prefill  ' "$H/l7-long-$v-$r.txt" | grep -o '[0-9.]* tok/s')"
done; done
