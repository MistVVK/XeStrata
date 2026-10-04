#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# run.sh MODEL TIER: one cell of the speed table with what setup writes for that context (images off), 256 greedy
# tokens, output to runs/MODEL-TIER.txt. IDS: the folder of prompts.py's ids; BUILD: build folder (default llvm7)
set -u
R="<repo>"; D="<data>"; H=$(cd "$(dirname "$0")" && pwd)
m=$1; t=$2
PROF=$R/data/expert-profile.bin
case $m in
  q2_0)    P=$D/packs/qwen-q2_0;     G=$D/models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0 ;;
  iq2_xs)  P=$D/packs/iq2_xs;        G=$D/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS ;;
  iq3_xxs) P=$D/packs/qwen-iq3_xxs;  G=$D/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS ;;
  iq3_s)   P=$D/packs/qwen-iq3_s;    G=$D/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S ;;
  coder)   P=$D/packs/coder-iq1_m;   G=$D/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M
           PROF=$R/data/expert-profile-coder.bin ;;
  *) echo "unknown model $m"; exit 2 ;;
esac
# setup's context for the tier (its smallest is 8K), int8 KV above 8K, KV streaming from 64K
case $t in
  1k|4k) C=(--max-context 8192) ;;
  32k)   C=(--max-context 32768 --kv int8) ;;
  64k)   C=(--max-context 65536 --kv int8 --kv-resident 32768) ;;
  128k)  C=(--max-context 131072 --kv int8 --kv-resident 32768) ;;
  262k)  C=(--max-context 262144 --kv int8 --kv-resident 32768) ;;
  *) echo "unknown tier $t"; exit 2 ;;
esac
export LD_LIBRARY_PATH="$R/.tools/intel-llvm/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ZES_ENABLE_SYSMAN=1
cd "$R" || exit 1
mkdir -p "$H/runs"
timeout 2400 "./build/${BUILD:-llvm7}/strata" --pack "$P" --native "$G-00001-of-00002.gguf" --ple-gguf "$G-00002-of-00002.gguf" \
  --expert-profile "$PROF" --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp "$D/mtp/rt" "${C[@]}" \
  --tokens-file "${IDS:-$H/ids}/$t.ids" --max-new 256 > "$H/runs/$m-$t.txt" 2>&1
echo "$m $t rc=$? $(grep -E '^(prefill|decode)  ' "$H/runs/$m-$t.txt" | tr -s ' ' | tr '\n' ' ')"
