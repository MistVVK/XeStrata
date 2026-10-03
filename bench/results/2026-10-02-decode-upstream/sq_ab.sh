#!/bin/bash
# sq_ab.sh: the grouped experts' SwiGLU and q8_1 quantization in one launch (bin/strata-w) against two (bin/strata-u):
# IQ3_S and IQ2_XS, MTP, 128 tokens on port_ids; three alternating rounds after a warm-up.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
for v in u w; do NODUMP=1 IDS=port_ids.txt NEW=8 EXE="$B/strata-$v" "$H/run.sh" sq-jit-$v iq3_s --suffix-draft 0 --mtp "$H/../../mtp/rt" > /dev/null; done
for r in 1 2 3; do for m in iq3_s iq2_xs; do for v in w u; do
  NODUMP=1 IDS=port_ids.txt NEW=128 EXE="$B/strata-$v" "$H/run.sh" "sq-$m-$v-$r" "$m" --suffix-draft 0 --mtp "$H/../../mtp/rt" > /dev/null
  echo "$r $m $v $(grep -E '^decode  ' "$H/sq-$m-$v-$r.txt" | grep -o '[0-9.]* tok/s')"
done; done; done
