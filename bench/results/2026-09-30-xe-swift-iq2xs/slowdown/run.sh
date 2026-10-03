#!/bin/bash
# the llama.cpp reference, then the engine without drafts (as the Swift IQ2_XS record did), then the engine again
W=<scratch>/swap; S=<scratch>
cd <repo>
swapon --show > $W/swap-before.txt; free -m >> $W/swap-before.txt
$W/watch.sh $W/watch.txt & wp=$!
M=$S/models/swift-iq2_xs; mkdir -p $W/keep; cp $M/spec-*.txt $M/ref.txt $M/ref.log $W/keep/ 2>/dev/null
echo "ref start $(date +%T)" >> $W/marks.txt; $S/model_check.sh $S/menv/swift-iq2_xs.env ref > $W/ref-step.txt 2>&1; echo "ref end $(date +%T)" >> $W/marks.txt
for i in 1 2; do
  echo "spec$i start $(date +%T)" >> $W/marks.txt; $S/model_check.sh $S/menv/swift-iq2_xs.env spec > $W/spec$i-step.txt 2>&1
  cp $M/spec-nodraft.txt $W/spec$i-nodraft.txt; cp $M/spec-suffix.txt $W/spec$i-suffix.txt; echo "spec$i end $(date +%T)" >> $W/marks.txt
done
kill $wp; pkill -P $wp; cp $W/keep/* $M/
