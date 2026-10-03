#!/bin/bash
# d_ab.sh OLD NEW [MODEL]: MTP decode (128 tokens on port_ids, --mtp) and no drafts (the --spec 4 window, suffix off,
# 96 tokens on chat_ids), one warm-up each, then two alternating rounds; logits dumped for the bit comparison.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin; o=$1; n=$2; m=${3:-iq3_s}
export CACHE=10000
for v in "$o" "$n"; do NODUMP=1 NEW=8 EXE="$B/strata-$v" "$H/run.sh" "d-jit-$v" "$m" --suffix-draft 0 --mtp "$H/../../mtp/rt" > /dev/null; done
for r in 1 2; do for v in "$o" "$n"; do
  IDS=port_ids.txt NEW=128 EXE="$B/strata-$v" "$H/run.sh" "d-mtp-$v-$m-$r" "$m" --suffix-draft 0 --mtp "$H/../../mtp/rt" > /dev/null
  EXE="$B/strata-$v" "$H/run.sh" "d-none-$v-$m-$r" "$m" --suffix-draft 0 > /dev/null
  echo "$v $r mtp $(grep -E '^decode  ' "$H/d-mtp-$v-$m-$r.txt" | grep -o '[0-9.]* tok/s') none $(grep -E '^decode  ' "$H/d-none-$v-$m-$r.txt" | grep -o '[0-9.]* tok/s')"
done; done
