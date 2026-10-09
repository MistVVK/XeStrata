#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# .claude/skills/release/scripts/bench.sh - a release's speed table on one GPU: the release and upstream's latest
# release, IQ3_XXS, the six tiers of bench/results/2026-10-04-speed-matrix (1K to 262K) with what setup writes for
# each context (images off), 256 greedy tokens, each run a new process.
#
#   XE="CMD..." UP="CMD..." DATA=DIR IDS=DIR .claude/skills/release/scripts/bench.sh OUT GPU
#
# XE and UP: the command that starts each engine (its strata binary, or a wrapper that runs it in its container);
# the generate options are appended.  XE_PROFILE and UP_PROFILE: each engine's shipped expert profile (its
# data/expert-profile.bin).  DATA: the data folder (packs/qwen-iq3_xxs, models/...IQ3_XXS-0000N-of-00002.gguf,
# mtp/rt).  IDS: the prompts of the speed-matrix record's prompts.py.  ROUNDS (default 2): runs per arm and tier.
# Another model file than IQ3_XXS: PACK, NATIVE (its first shard), PLE (the shard with per_layer_token_embd) and EXTRA
# (options both engines take besides, e.g. UD-Q4_K_XL's --resident-budget-gib).  TIERS (default all six): a subset.
# OUT/runs/GPU-ARM-TIER-ROUND.txt gets each log; one discarded warm-up per arm first (JIT); the arms alternate,
# the first arm changing from round to round.  bench-collect.py OUT reads them.
set -u
[ $# -eq 2 ] || { echo "usage: $0 OUT GPU" >&2; exit 2; }
out=$1; gpu=$2
: "${XE:?}" "${UP:?}" "${DATA:?}" "${IDS:?}" "${XE_PROFILE:?}" "${UP_PROFILE:?}"
rounds=${ROUNDS:-2}
G=$DATA/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS
P=${PACK:-$DATA/packs/qwen-iq3_xxs}
N=${NATIVE:-$G-00001-of-00002.gguf}
L=${PLE:-$G-00002-of-00002.gguf}
read -r -a X <<< "${EXTRA:-}"
mkdir -p "$out/runs"

# ARM TIER FILE: one run
run() {
  local cmd prof C
  case $1 in
    xe) read -r -a cmd <<< "$XE"; prof=$XE_PROFILE ;;
    up) read -r -a cmd <<< "$UP"; prof=$UP_PROFILE ;;
  esac
  # setup's context for the tier (its smallest is 8K), int8 KV above 8K, KV streaming from 64K
  case $2 in
    1k|4k) C=(--max-context 8192) ;;
    32k)   C=(--max-context 32768 --kv int8) ;;
    64k)   C=(--max-context 65536 --kv int8 --kv-resident 32768) ;;
    128k)  C=(--max-context 131072 --kv int8 --kv-resident 32768) ;;
    262k)  C=(--max-context 262144 --kv int8 --kv-resident 32768) ;;
  esac
  timeout 2400 "${cmd[@]}" --pack "$P" --native "$N" --ple-gguf "$L" "${X[@]}" \
    --expert-profile "$prof" --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp "$DATA/mtp/rt" \
    "${C[@]}" --tokens-file "$IDS/$2.ids" --max-new 256 > "$3" 2>&1
  echo "$gpu $1 $2 rc=$? $(grep -E '^(prefill|decode)  ' "$3" | tr -s ' ' | tr '\n' ' ')"
}

for a in xe up; do run "$a" 1k "$out/runs/$gpu-$a-warmup.txt"; done
for t in ${TIERS:-1k 4k 32k 64k 128k 262k}; do
  for r in $(seq 1 "$rounds"); do
    if [ $((r % 2)) -eq 1 ]; then arms="xe up"; else arms="up xe"; fi
    for a in $arms; do run "$a" "$t" "$out/runs/$gpu-$a-$t-$r.txt"; done
  done
done
