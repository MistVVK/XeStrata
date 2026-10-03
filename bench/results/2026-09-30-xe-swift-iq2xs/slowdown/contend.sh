#!/bin/bash
# the engine without drafts on Swift IQ2_XS while sha256sum reads a 40 GB model file, as fetch-all.sh did after each download
W=<scratch>/swap; S=<scratch>; cd <repo>
$W/watch.sh $W/watch-contend.txt & wp=$!
sha256sum <data>/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf > $W/sha.txt 2>&1 & sp=$!
echo "start $(date +%T)" > $W/marks-contend.txt
$S/model_check.sh $S/menv/swift-iq2_xs.env spec > $W/contend-step.txt 2>&1
cp $S/models/swift-iq2_xs/spec-nodraft.txt $W/contend-nodraft.txt; cp $S/models/swift-iq2_xs/spec-suffix.txt $W/contend-suffix.txt
echo "end $(date +%T); sha256sum still running: $(kill -0 $sp 2>/dev/null && echo yes || echo no)" >> $W/marks-contend.txt
kill $sp 2>/dev/null; kill $wp; pkill -P $wp; cp $W/keep/spec-*.txt $S/models/swift-iq2_xs/
