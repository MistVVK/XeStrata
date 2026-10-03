#!/bin/bash
# chat_base.sh: the old records' "spec" step (chat prompt, 96 tokens, no drafts and MTP), two rounds; portable build
H=$(cd "$(dirname "$0")" && pwd)
export BUILD=xe-portable NODUMP=1
for r in 1 2; do
  for m in iq2_xs iq3_xxs iq3_s q2_0 coder-iq1_m swift-iq2_xs swift-iq3_xxs swift-q2_0; do
    "$H/run.sh" chat-$m-nodraft-$r $m --suffix-draft 0
    "$H/run.sh" chat-$m-mtp-$r $m --mtp "<data>/mtp/rt"
  done
done
