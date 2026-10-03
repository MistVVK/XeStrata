#!/bin/bash
# nodraft (T=1 windows) and oracle (T=4 windows, every draft accepted) on IQ3_XXS, with the CPU on ggml's dot and by default
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export ZES_ENABLE_SYSMAN=1
cd <repo>
source <scratch>/menv/qwen-iq3_xxs.env
BASE=(--pack "$PACK" --native "$S1" --ple-gguf "$PLE" --expert-profile "$PROF" --expert-cache auto --prefill 512 --spec 4 --adapt-every 0 --suffix-draft 0)
ids=$(cat <scratch>/models/qwen-iq3_xxs/chat_ids.txt)
for cpu in ggml default; do
  if [ $cpu = ggml ]; then E=(env STRATA_NO_IQ256=1 STRATA_NO_IQ512=1 STRATA_NO_IQ4NL=1); else E=(env); fi
  "${E[@]}" ./build/xe/strata "${BASE[@]}" --tokens "$ids" --max-new 96 --dump-logits <scratch>/iq3/$cpu-nodraft.bin > <scratch>/iq3/$cpu-nodraft.txt 2>&1; echo "$cpu nodraft rc=$?"
  grep "^output" <scratch>/iq3/$cpu-nodraft.txt | sed 's/^output *: *//; s/ /,/g' > <scratch>/iq3/$cpu-oracle.txt
  "${E[@]}" ./build/xe/strata "${BASE[@]}" --spec-oracle <scratch>/iq3/$cpu-oracle.txt --tokens "$ids" --max-new 96 --dump-logits <scratch>/iq3/$cpu-oracle.bin > <scratch>/iq3/$cpu-oracle.txt.log 2>&1; echo "$cpu oracle rc=$?"
done
