#!/bin/bash
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export ZES_ENABLE_SYSMAN=1
cd <repo>
source <scratch>/menv/qwen-iq3_xxs.env
ids=$(cat <scratch>/models/qwen-iq3_xxs/chat_ids.txt)
env STRATA_NO_IQ256=1 STRATA_NO_IQ512=1 STRATA_NO_IQ4NL=1 ./build/xe/strata --pack "$PACK" --native "$S1" --ple-gguf "$PLE" --expert-profile "$PROF" --expert-cache auto --prefill 512 --spec 4 --adapt-every 0 --tokens "$ids" --max-new 96 --dump-logits <scratch>/iq3/ggml-suffix.bin > <scratch>/iq3/ggml-suffix.txt 2>&1; echo "rc=$?"
