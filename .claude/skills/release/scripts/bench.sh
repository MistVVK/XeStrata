#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# .claude/skills/release/scripts/bench.sh - a release's speed table on one GPU: the release and upstream's latest
# release, IQ3_XXS, the six tiers of the 2026-10-04-speed-matrix record (1K to 262K) with what setup writes for
# each context on this machine (images off; KV streaming only where the RAM holds it, from 2026-10-12), 256 greedy
# tokens, each run a new process.  Both engines follow the same continuation of each prompt (--spec-follow
# IDS/TIER.follow, follow.sh beside this script, from 2026-10-16): each engine's own greedy text splits from the
# other's within tens of tokens, and the drafts' acceptance on the text it splits into moved decode by up to 20%.
#
#   XE="CMD..." UP="CMD..." DATA=DIR IDS=DIR .claude/skills/release/scripts/bench.sh OUT GPU
#
# XE and UP: the command that starts each engine (XeStrata's xestrata-engine, upstream's strata, or a wrapper that runs
# it in its container); the generate options are appended.  XE_PROFILE and UP_PROFILE: each engine's shipped expert profile (its
# data/expert-profile.bin).  DATA: the data folder (packs/qwen-iq3_xxs, models/...IQ3_XXS-0000N-of-00002.gguf,
# mtp/rt).  IDS: the prompts of prompts.py and the continuations of follow.sh beside this script.  ROUNDS (default
# 2): runs per arm and tier.  TIERS (default the six): fewer, for a check of the scripts (TIERS=1k ROUNDS=1), not a
# release's table.
# OUT/runs/GPU-ARM-TIER-ROUND.txt gets each log; one discarded warm-up per arm first (JIT); the arms alternate,
# the first arm changing from round to round.  bench-collect.py OUT reads them.
# upstream plainly at SLOW tok/s or less (default 10) is stopped mid-run and not run again in that tier: its prompt
# when the PP progress line says so after 30 s, its output when 256 tokens at SLOW tok/s and a minute have passed
# since the prompt was read.  The log ends with "bench: stopped, upstream's prompt|output ...".  SLOW=0: never.
set -u
[ $# -eq 2 ] || { echo "usage: $0 OUT GPU" >&2; exit 2; }
out=$1; gpu=$2
: "${XE:?}" "${UP:?}" "${DATA:?}" "${IDS:?}" "${XE_PROFILE:?}" "${UP_PROFILE:?}"
rounds=${ROUNDS:-2}
tiers=${TIERS:-1k 4k 32k 64k 128k 262k}
slow=${SLOW:-10}
P=$DATA/packs/qwen-iq3_xxs
G=$DATA/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS
for t in $tiers; do
  [ -s "$IDS/$t.follow" ] || { echo "no $IDS/$t.follow: write the continuations with follow.sh first" >&2; exit 2; }
done
mkdir -p "$out/runs"
# KV streaming where setup writes it for the model on this machine's RAM (setup.kv_streaming_wanted, through
# tools/bench_core.py; upstream's setup decides the same, its sycl/ port again in up-sycl.sh)
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
declare -A kvr
for c in 65536 131072 262144; do
  kvr[$c]=$(python3 "$repo/tools/bench_core.py" kv-resident "$G-00001-of-00002.gguf" "$c") || exit 1
done

# PID LOG: writes LOG.stop and stops PID once the run is plainly at $slow tok/s or less
watch_slow() {
  local pp p_done p_total ms rate t_done=
  while kill -0 "$1" 2>/dev/null; do
    sleep 5
    pp=$(grep '^PP ' "$2" | tail -n 1)
    [ -n "$pp" ] || continue
    read -r _ p_done p_total ms rate <<< "$pp"
    if [ "$p_done" -lt "$p_total" ]; then
      if [ "$ms" -ge 30000 ] && awk -v r="$rate" -v s="$slow" 'BEGIN { exit !(r <= s) }'; then
        echo "bench: stopped, upstream's prompt at $rate tok/s after $((ms / 1000)) s" > "$2.stop"; kill "$1"; return
      fi
    elif [ -z "$t_done" ]; then
      t_done=$SECONDS
    elif [ $((SECONDS - t_done)) -gt $((256 / slow + 60)) ]; then
      echo "bench: stopped, upstream's output under $slow tok/s ($((SECONDS - t_done)) s after the prompt)" > "$2.stop"
      kill "$1"; return
    fi
  done
}

# ARM TIER FILE: one run
run() {
  local cmd prof C
  case $1 in
    xe) read -r -a cmd <<< "$XE"; prof=$XE_PROFILE ;;
    up) read -r -a cmd <<< "$UP"; prof=$UP_PROFILE ;;
  esac
  # setup's context for the tier (its smallest is 8K), int8 KV above 8K, KV streaming from 64K where it fits the RAM
  case $2 in
    1k|4k) C=(--max-context 8192) ;;
    32k)   C=(--max-context 32768 --kv int8) ;;
    64k)   C=(--max-context 65536 --kv int8) ;;
    128k)  C=(--max-context 131072 --kv int8) ;;
    262k)  C=(--max-context 262144 --kv int8) ;;
  esac
  case $2 in 64k|128k|262k) read -r -a k <<< "${kvr[${C[1]}]}"; C+=("${k[@]}") ;; esac
  timeout 2400 "${cmd[@]}" --pack "$P" --native "$G-00001-of-00002.gguf" --ple-gguf "$G-00002-of-00002.gguf" \
    --expert-profile "$prof" --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp "$DATA/mtp/rt" \
    "${C[@]}" --tokens-file "$IDS/$2.ids" --spec-follow "$IDS/$2.follow" --max-new 256 > "$3" 2>&1 &
  local pid=$! rc w=
  if [ "$1" = up ] && [ "$slow" -gt 0 ]; then watch_slow "$pid" "$3" & w=$!; fi
  wait "$pid"; rc=$?
  [ -z "$w" ] || wait "$w"
  if [ -f "$3.stop" ]; then cat "$3.stop" >> "$3"; rm -f "$3.stop"; fi
  echo "$gpu $1 $2 rc=$rc $(grep -E '^(prefill|decode)  |^bench: stopped' "$3" | tr -s ' ' | tr '\n' ' ')"
}

for a in xe up; do run "$a" 1k "$out/runs/$gpu-$a-warmup.txt"; done
for t in $tiers; do
  up_slow=
  for r in $(seq 1 "$rounds"); do
    if [ $((r % 2)) -eq 1 ]; then arms="xe up"; else arms="up xe"; fi
    for a in $arms; do
      [ "$a" = up ] && [ -n "$up_slow" ] && continue
      run "$a" "$t" "$out/runs/$gpu-$a-$t-$r.txt"
      [ "$a" = up ] && grep -q '^bench: stopped' "$out/runs/$gpu-$a-$t-$r.txt" && up_slow=1
    done
  done
done

exit 0
