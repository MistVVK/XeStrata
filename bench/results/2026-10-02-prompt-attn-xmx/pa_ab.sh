#!/bin/bash
# pa_ab.sh: the engine with the FP32 prompt attention (bin/strata-icpx) against the XMX one (bin/strata-pa), IQ3_S,
# the 26,292-token prompt (prefill auto, int8 KV, 24 tokens, logits), two alternating rounds after a warm-up each.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
for v in icpx pa; do NODUMP=1 IDS=port_ids.txt NEW=4 EXE=$B/strata-$v "$H/run.sh" pa-jit-$v iq3_s --suffix-draft 0 > /dev/null; done
for r in 1 2; do for v in icpx pa; do
  for _ in 1 2 3; do IDS=long_ids.txt NEW=24 PREFILL=auto EXE=$B/strata-$v "$H/run.sh" pa-long-$v-$r iq3_s --max-context 32768 --kv int8 --suffix-draft 0 | grep -q "rc=0" && break; done
  echo "$v $r prefill $(grep -E '^prefill  ' "$H/pa-long-$v-$r.txt" | grep -o '[0-9.]* tok/s')"
done; done
