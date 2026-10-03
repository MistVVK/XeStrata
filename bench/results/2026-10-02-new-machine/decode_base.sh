#!/bin/bash
# decode_base.sh: every model, no drafts and MTP, 128 tokens, two rounds alternating; portable build, no logits dump
H=$(cd "$(dirname "$0")" && pwd)
export BUILD=xe-portable IDS=port_ids.txt NEW=128 NODUMP=1
for r in 1 2; do
  for m in iq2_xs iq3_xxs iq3_s q2_0 coder-iq1_m swift-iq2_xs swift-iq3_xxs swift-q2_0; do
    "$H/run.sh" base-$m-nodraft-$r $m --suffix-draft 0
    "$H/run.sh" base-$m-mtp-$r $m --mtp "<data>/mtp/rt"
  done
done
