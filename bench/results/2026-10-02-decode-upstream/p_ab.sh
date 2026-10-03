#!/bin/bash
# p_ab.sh: the sampled path split over the GPU (bin/strata-p) against the one-group kernel (STRATA_OLD_SAMPLER=1):
# IQ3_S, MTP, 128 tokens on port_ids, temperature 0.7, top_k 40, top_p 0.95, seed 7; greedy as the ceiling; two
# alternating rounds after a warm-up.
H=$(cd "$(dirname "$0")" && pwd); B=$H/bin
export CACHE=10000
S=(--seed 7 --temperature 0.7 --top-k 40 --top-p 0.95)
NODUMP=1 IDS=port_ids.txt NEW=8 EXE="$B/strata-p" "$H/run.sh" p-jit iq3_s --suffix-draft 0 --mtp "$H/../../mtp/rt" "${S[@]}" > /dev/null
for r in 1 2; do for arm in split old greedy; do
  case $arm in split) E=(); X=("${S[@]}");; old) E=(STRATA_OLD_SAMPLER=1); X=("${S[@]}");; greedy) E=(); X=();; esac
  env "${E[@]}" NODUMP=1 IDS=port_ids.txt NEW=128 EXE="$B/strata-p" "$H/run.sh" "p-$arm-$r" iq3_s --suffix-draft 0 --mtp "$H/../../mtp/rt" "${X[@]}" > /dev/null
  f=$H/p-$arm-$r.txt
  echo "$r $arm $(grep -E '^decode  ' "$f" | grep -o '[0-9.]* tok/s') $(grep -o 'drafts accepted [0-9]* of [0-9]*' "$f") out $(grep '^output  :' "$f" | md5sum | cut -c1-8)"
done; done
