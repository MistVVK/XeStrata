#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# ab.sh NAME MODEL TOKENS MAXNEW [VAR=VAL ...] [-- EXTRA...]: one run of the engine for the record's tables.
# MODEL: iq2 (IQ2_XS, the expert cache), ud (Unsloth UD-Q4_K_XL, a 36 GiB RAM budget) or udm (UD-Q4_K_XL from the
# files through the OS cache, a 1 GiB budget).  STRATA is the engine, DATA the data folder (models/, packs/, mtp/).
# Run from the repository root; the line with "decode" and "prefill" is the measurement.
set -u
STRATA=${STRATA:-build/xe/strata}
D=${DATA:?set DATA to the data folder}
n=$1 model=$2 toks=$3 maxnew=$4; shift 4
envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; [ $# -gt 0 ] && shift
I=$D/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS U=$D/models/unsloth/Qwen3.8-Flash-Next-UD-Q4_K_XL
case $model in
  iq2) M=(--pack "$D/packs/iq2_xs" --native "$I-00001-of-00002.gguf" --ple-gguf "$I-00002-of-00002.gguf") ;;
  ud)  M=(--pack "$D/packs/unsloth-ud-q4_k_xl" --native "$U-00001-of-00004.gguf" --ple-gguf "$U-00002-of-00004.gguf"
          --resident-budget-gib 36) ;;
  udm) M=(--pack "$D/packs/unsloth-ud-q4_k_xl" --native "$U-00001-of-00004.gguf" --ple-gguf "$U-00002-of-00004.gguf"
          --resident-budget-gib 1 --mmap-experts) ;;
  *) echo "ab.sh: MODEL is iq2, ud or udm" >&2; exit 2 ;;
esac
env "${envs[@]}" "$STRATA" "${M[@]}" --max-context 8192 --expert-profile data/expert-profile.bin --expert-cache auto \
    --prefill 4096 --spec 4 --spec-min-p 0.5 --mtp "$D/mtp/rt" --tokens-file "$toks" --max-new "$maxnew" "$@" \
    > "$n.log" 2>&1
printf "%-12s rc=%d  " "$n" $?
grep -oE "[0-9]+ expert-pool workers" "$n.log" | head -1 | tr '\n' ' '
grep -E "^(decode|prefill) " "$n.log" | sed -E 's/ +/ /g' | tr '\n' ' '; echo
