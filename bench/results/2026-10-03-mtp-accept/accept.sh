#!/bin/bash
# accept.sh: MTP draft acceptance per model and prompt.  setup's decode settings (--spec 4 --spec-min-p 0.5, the MTP
# draft layer with the CJK draft vocabulary), greedy, up to 512 tokens ending at the first end of turn (--stop-eos),
# the adaptive VRAM tier off (--adapt-every 0) so a run does not depend on the previous one; one run per (model, prompt).
H=$(cd "$(dirname "$0")" && pwd); R=${XESTRATA:?the XeStrata folder}; D=${DATA:?the data folder}
cd "$R" || exit 1
for m in iq2_xs:iq2_xs:Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS:2 q2_0:qwen-q2_0:Qwen3.8-Flash-Next-GSQ-RCO-Q2_0:2 \
         iq3_xxs:qwen-iq3_xxs:Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS:2 iq3_s:qwen-iq3_s:Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S:2 \
         coder:coder-iq1_m:Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M:2 swift-iq2_xs:swift-iq2_xs:Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS:1 \
         swift-q2_0:swift-q2_0:Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0:1 swift-iq3_xxs:swift-iq3_xxs:Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS:1; do
  IFS=: read -r name pack gguf ple <<< "$m"
  prof=$R/data/expert-profile.bin; [ "$name" = coder ] && prof=$R/data/expert-profile-coder.bin
  for p in chat ja code math; do
    out=$H/$name-$p.txt
    timeout 1500 ./build/xe/strata --pack "$D/packs/$pack" --native "$D/models/$gguf-00001-of-00002.gguf" \
      --ple-gguf "$D/models/$gguf-0000$ple-of-00002.gguf" --expert-profile "$prof" --expert-cache auto --prefill auto \
      --spec 4 --spec-min-p 0.5 --mtp "$D/mtp/rt" --adapt-every 0 --tokens-file "$H/${p}_ids.txt" --max-new 512 --stop-eos \
      > "$out" 2>&1
    echo "$name $p rc=$? $(grep -E '^decode ' "$out" | grep -o '[0-9]* tokens') $(grep -E '^speculation ' "$out" | sed 's/  */ /g') $(grep -E '^decode ' "$out" | grep -o '[0-9.]* tok/s')"
  done
done
