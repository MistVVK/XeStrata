#!/bin/bash
# long_base.sh: the old records' "long" resident-MTP run (26,292-token prompt, 24 tokens, int8 KV, prefill auto) per
# model, then IQ2_XS and IQ3_S again with STRATA_PREFILL_TIMING=1 for the breakdown (its own overhead: not for speed)
H=$(cd "$(dirname "$0")" && pwd)
export BUILD=xe-portable NODUMP=1 IDS=long_ids.txt NEW=24 PREFILL=auto
L=(--max-context 32768 --kv int8 --mtp "<data>/mtp/rt")
for m in iq2_xs iq3_xxs iq3_s q2_0 coder-iq1_m swift-iq2_xs swift-iq3_xxs swift-q2_0; do
  "$H/run.sh" long-$m $m "${L[@]}"
  grep -E '^prefill  ' "$H/long-$m.txt"
done
for m in iq2_xs iq3_s; do
  STRATA_PREFILL_TIMING=1 "$H/run.sh" long-timing-$m $m "${L[@]}"
done
