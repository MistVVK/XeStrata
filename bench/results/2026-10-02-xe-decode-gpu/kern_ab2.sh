#!/bin/bash
# kern_ab2.sh: the engine before the dense-kernel work (strata-base, f9a9cfc) against the current one (strata-cur, 27cbc31),
# IQ3_S and IQ2_XS, no drafts and MTP, 128 tokens, --expert-cache 10000, two rounds alternating
H=$(cd "$(dirname "$0")" && pwd)
export IDS=port_ids.txt NEW=128 NODUMP=1 CACHE=10000
for r in 1 2; do for m in iq3_s iq2_xs; do for v in base cur; do
  for mode in nd mtp; do
    if [ $mode = mtp ]; then X=(--mtp "<data>/mtp/rt"); else X=(--suffix-draft 0); fi
    for _ in 1 2 3; do EXE=$H/bin/strata-$v "$H/run.sh" ab2-$m-$v-$mode-$r $m "${X[@]}" | grep -q "rc=0" && break; done
  done
done; done; done
for m in iq3_s iq2_xs; do for mode in nd mtp; do for v in base cur; do
  printf "%-7s %-4s %-5s" $m $mode $v
  for r in 1 2; do printf " %s" "$(grep -E '^decode  ' "$H/ab2-$m-$v-$mode-$r.txt" | grep -o '[0-9.]* tok/s')"; done; echo
done; done; done
