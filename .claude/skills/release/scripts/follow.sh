#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# .claude/skills/release/scripts/follow.sh - the continuations the speed table's engines follow (--spec-follow):
# for each tier, the 256 greedy tokens one engine writes after the prompt, as IDS/TIER.follow beside IDS/TIER.ids.
# Made once and kept with the prompts, so every release and upstream are timed on the same text: a greedy text
# splits from another engine's, build's or setting's within tens of tokens (the rounding differs), and the drafts'
# acceptance on the text it splits into moved decode by up to 20% (2026-10-16, 16 prompts on the RTX 4070 and the
# RX 9060 XT).
#
#   XE="CMD..." DATA=DIR IDS=DIR XE_PROFILE=FILE .claude/skills/release/scripts/follow.sh [TIER...]
#
# XE, DATA, IDS and XE_PROFILE as for bench.sh; TIER defaults to the six.  Each run takes bench.sh's settings with
# the BF16 hyper-connection read (XESTRATA_HC_Q8=0, the model's own weights).  An existing file is kept.
set -u
: "${XE:?}" "${DATA:?}" "${IDS:?}" "${XE_PROFILE:?}"
P=$DATA/packs/qwen-iq3_xxs
G=$DATA/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
read -r -a cmd <<< "$XE"
[ $# -gt 0 ] || set -- 1k 4k 32k 64k 128k 262k
for t in "$@"; do
  f=$IDS/$t.follow
  [ -s "$f" ] && { echo "$t: $f is there"; continue; }
  case $t in
    1k|4k) C=(--max-context 8192) ;;
    32k)   C=(--max-context 32768 --kv int8) ;;
    64k)   C=(--max-context 65536 --kv int8) ;;
    128k)  C=(--max-context 131072 --kv int8) ;;
    262k)  C=(--max-context 262144 --kv int8) ;;
    *) echo "unknown tier $t" >&2; exit 2 ;;
  esac
  case $t in 64k|128k|262k)
    read -r -a k <<< "$(python3 "$repo/tools/bench_core.py" kv-resident "$G-00001-of-00002.gguf" "${C[1]}")" || exit 1
    C+=("${k[@]}") ;;
  esac
  log=$(mktemp) || exit 1
  XESTRATA_HC_Q8=0 timeout 2400 "${cmd[@]}" --pack "$P" --native "$G-00001-of-00002.gguf" \
    --ple-gguf "$G-00002-of-00002.gguf" --expert-profile "$XE_PROFILE" --expert-cache auto --prefill auto --spec 4 \
    --spec-min-p 0.5 --mtp "$DATA/mtp/rt" "${C[@]}" --tokens-file "$IDS/$t.ids" --max-new 256 > "$log" 2>&1
  rc=$?
  ids=$(sed -n 's/^output *: *//p' "$log" | tr -s ' ' ',' | sed 's/,$//')
  if [ $rc -ne 0 ] || [ -z "$ids" ]; then
    echo "$t: the run failed (rc=$rc), its log: $log" >&2; exit 1
  fi
  printf '%s' "$ids" > "$f"
  rm -f "$log"
  echo "$t: $(tr , "\n" < "$f" | grep -c .) tokens into $f"
done
