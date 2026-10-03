#!/bin/bash
# run.sh NAME MODEL EXTRA...: one engine run (greedy, fixed residency), output to NAME.txt (and NAME.logits).
# NODUMP=1: no logits dump; IDS=file of token ids (default chat_ids.txt); NEW=tokens to generate (default 96);
# BUILD=build folder under build/ (default xe); PREFILL=prefill chunk (default 512)
set -u
R="<repo>"; D="<data>"; H=$(cd "$(dirname "$0")" && pwd)
name=$1; m=$2; shift 2
PROF=$R/data/expert-profile.bin; PLE=2
case $m in
  iq2_xs)        P=$D/packs/iq2_xs; G=Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS ;;
  iq3_s)         P=$D/packs/qwen-iq3_s; G=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S ;;
  iq3_xxs)       P=$D/packs/qwen-iq3_xxs; G=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS ;;
  q2_0)          P=$D/packs/qwen-q2_0; G=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0 ;;
  coder-iq1_m)   P=$D/packs/coder-iq1_m; G=Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M; PROF=$R/data/expert-profile-coder.bin ;;
  swift-iq2_xs)  P=$D/packs/swift-iq2_xs; G=Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS; PLE=1 ;;
  swift-iq3_xxs) P=$D/packs/swift-iq3_xxs; G=Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS; PLE=1 ;;
  swift-q2_0)    P=$D/packs/swift-q2_0; G=Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0; PLE=1 ;;
  *) echo "unknown model $m"; exit 2 ;;
esac
set +u
# shellcheck source=/dev/null
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u
export ZES_ENABLE_SYSMAN=1
cd "$R" || exit 1
DUMP=(--dump-logits "$H/$name.logits"); [ -n "${NODUMP:-}" ] && DUMP=()
timeout 1500 "./build/${BUILD:-xe}/strata" --pack "$P" --native "$D/models/$G-00001-of-00002.gguf" --ple-gguf "$D/models/$G-0000$PLE-of-00002.gguf" \
  --expert-profile "$PROF" --expert-cache "${CACHE:-auto}" --prefill "${PREFILL:-512}" --spec 4 --adapt-every 0 \
  --tokens-file "$H/${IDS:-chat_ids.txt}" --max-new "${NEW:-96}" "${DUMP[@]}" "$@" > "$H/$name.txt" 2>&1
echo "$name rc=$? $(grep -E '^decode  ' "$H/$name.txt")"
