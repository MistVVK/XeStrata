#!/bin/bash
# xmx_ab.sh: oneMKL (bin/strata-mkl) against the XMX kernels (bin/strata-xmx), the long prompt (IQ3_S, prefill auto,
# int8 KV, 4 tokens, logits dumped), two alternating rounds after a short warm-up run (JIT).  It ran in <record> next
# to ../2026-10-02-new-machine/run.sh and long_ids.txt (the 26,292-token prompt).
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
IDS=port_ids.txt NEW=4 NODUMP=1 EXE=$B/strata-xmx "$H/run.sh" xmx-jit iq3_s --suffix-draft 0 > /dev/null
for r in 1 2; do for v in mkl xmx; do
  for _ in 1 2 3; do IDS=long_ids.txt NEW=4 PREFILL=auto EXE=$B/strata-$v "$H/run.sh" xmx-long-$v-$r iq3_s --max-context 32768 --kv int8 --suffix-draft 0 "$@" | grep -q "rc=0" && break; done
  echo "$v $r $(grep -E '^prefill  ' "$H/xmx-long-$v-$r.txt" | grep -o '[0-9.]* tok/s')"
done; done
