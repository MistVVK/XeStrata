#!/bin/bash
# e_ab.sh: bin/strata-d2 (D1-D3, the AVX-2 prefetch, the draft head's reserve), IQ3_S, MTP, 128 tokens.
#  1. each change switched off by its own variable against all on: STRATA_COMMIT_SYNC=1 (D2), STRATA_SCORES_MULTI=0
#     (D3), STRATA_IQ_PREFETCH=0 (the prefetch); port_ids, two alternating rounds after a warm-up
#  2. the draft subset: mtp/rt (English and code) against mtp/rt-cjk (with CJK), on a Japanese (ja_ids) and an English
#     (chat_ids) prompt, two alternating rounds
H=$(cd "$(dirname "$0")" && pwd); EXE="$H/bin/strata-d2"; M=$H/../../mtp
export CACHE=10000
NODUMP=1 NEW=8 EXE="$EXE" "$H/run.sh" "e-jit" iq3_s --suffix-draft 0 --mtp "$M/rt" > /dev/null
NODUMP=1 NEW=8 EXE="$EXE" "$H/run.sh" "e-jit2" iq3_s --suffix-draft 0 --mtp "$M/rt-cjk" > /dev/null
dec() { grep -E '^decode  ' "$H/$1.txt" | grep -o '[0-9.]* tok/s'; }
acc() { grep -E '^speculation' "$H/$1.txt" | grep -o 'accepted [0-9]* of [0-9]*'; }
for r in 1 2; do
  for v in on sync noscores nopf; do
    case $v in on) E=();; sync) E=(STRATA_COMMIT_SYNC=1);; noscores) E=(STRATA_SCORES_MULTI=0);; nopf) E=(STRATA_IQ_PREFETCH=0);; esac
    env "${E[@]}" IDS=port_ids.txt NEW=128 EXE="$EXE" "$H/run.sh" "e-$v-$r" iq3_s --suffix-draft 0 --mtp "$M/rt" > /dev/null
    echo "$r $v $(dec "e-$v-$r")"
  done
done
for r in 1 2; do for p in ja chat; do for rt in rt rt-cjk; do
  IDS="${p}_ids.txt" NEW=128 EXE="$EXE" "$H/run.sh" "e-$p-$rt-$r" iq3_s --suffix-draft 0 --mtp "$M/$rt" > /dev/null
  echo "$r $p $rt $(dec "e-$p-$rt-$r") $(acc "e-$p-$rt-$r")"
done; done; done
