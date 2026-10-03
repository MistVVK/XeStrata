#!/bin/bash
# One model through the IQ2_XS checks (bench/results/2026-09-30-xe-iq2xs), with the same tools and conditions.
#   model_check.sh ENVFILE STEP...
# ENVFILE sets TAG, S1 (shard 1), S2 (shard 2), PLE (the shard with per_layer_token_embd), PROF (expert profile),
# NEP ("FILE:L L L;FILE:L L" - native_expert_parity runs), PACK (pack dir).  Steps: pack nep dq paris ref spec
# server long.  Outputs go to $OUT (scratch), one file per run.
set -u
source "$1"; shift
R=<repo>
S=<scratch>
OUT=$S/models/$TAG; mkdir -p "$OUT"
PY=<data>/venv/bin/python
MTP=<data>/mtp/rt
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; set -u
export ZES_ENABLE_SYSMAN=1
cd "$R"
BASE=(--pack "$PACK" --native "$S1" --ple-gguf "$PLE" --expert-profile "$PROF" --expert-cache auto)
chat_ids() {   # the chat prompt through this model's own tokenizer
  $PY - "$PACK" <<'EOF'
import json, pathlib, sys
sys.path.insert(0, "tools")
import strata_tokenizer as ST
p = pathlib.Path(sys.argv[1]) / "tokenizer"
vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items(): tokens[i] = t
tok = ST.Tokenizer(tokens, (p / "merges.txt").read_text(encoding="utf-8").split("\n"), json.loads((p / "token_type.json").read_text()))
text = "<|im_start|>user\nExplain in three sentences why the sky is blue.<|im_end|>\n<|im_start|>assistant\n"
print(",".join(map(str, tok.encode(text, parse_special=True))))
EOF
}
for step in "$@"; do
  echo "=== $TAG $step $(date -Iseconds)"
  case $step in
  pack)
    [ -e "$PACK/native_experts.txt" ] && [ -e "$PACK/tokenizer/vocab.json" ] && { echo "pack present"; continue; }
    STRATA_GGUF_PY=$R/third_party/llama.cpp/gguf-py timeout 3000 $PY tools/iq_pack.py --gguf "$S1" --out "$PACK" $PACKARGS \
      > "$OUT/pack.txt" 2>&1; echo "rc=$?"; tail -5 "$OUT/pack.txt" ;;
  nep)
    i=0; IFS=';' read -ra runs <<< "$NEP"
    for r in "${runs[@]}"; do
      f=${r%%:*}; layers=${r#*:}; i=$((i + 1))
      timeout 1800 ./build/xe/native_expert_parity "$f" $layers > "$OUT/nep$i.txt" 2>&1; echo "rc=$? ($f: $layers)"
      grep -iE "fail|layer [0-9]+:|rel" "$OUT/nep$i.txt" | tail -12
    done ;;
  dq)
    timeout 1800 ./build/xe/dequant_bf16_test "$S1" "$S2" > "$OUT/dq.txt" 2>&1; echo "rc=$?"; cat "$OUT/dq.txt" ;;
  paris)
    ids=$($PY - "$PACK" <<'EOF'
import json, pathlib, sys
sys.path.insert(0, "tools")
import strata_tokenizer as ST
p = pathlib.Path(sys.argv[1]) / "tokenizer"
vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items(): tokens[i] = t
tok = ST.Tokenizer(tokens, (p / "merges.txt").read_text(encoding="utf-8").split("\n"), json.loads((p / "token_type.json").read_text()))
print(",".join(map(str, tok.encode("The capital of France is", parse_special=True))))
EOF
)
    timeout 1500 ./build/xe/strata "${BASE[@]}" --prefill 512 --spec 4 --tokens "$ids" --max-new 12 > "$OUT/paris.txt" 2>&1
    echo "rc=$?"; grep -E "^output|^decode  |^prefill  " "$OUT/paris.txt" ;;
  ref)
    ids=$(chat_ids); echo "$ids" > "$OUT/chat_ids.txt"
    ( cd "$OUT" && timeout 7200 "$S/ref_logits" "$S1" "$ids" 96 "$OUT/ref_logits.txt" > "$OUT/ref.txt" 2> "$OUT/ref.log" ); echo "rc=$?"
    cut -c1-200 "$OUT/ref.txt" ;;
  spec)
    ids=$(cat "$OUT/chat_ids.txt" 2>/dev/null || chat_ids)
    timeout 1500 ./build/xe/strata "${BASE[@]}" --prefill 512 --spec 4 --adapt-every 0 --suffix-draft 0 --tokens "$ids" --max-new 96 > "$OUT/spec-nodraft.txt" 2>&1; echo "nodraft rc=$?"
    timeout 1500 ./build/xe/strata "${BASE[@]}" --prefill 512 --spec 4 --adapt-every 0 --tokens "$ids" --max-new 96 > "$OUT/spec-suffix.txt" 2>&1; echo "suffix rc=$?"
    timeout 1500 ./build/xe/strata "${BASE[@]}" --prefill 512 --spec 4 --adapt-every 0 --mtp "$MTP" --tokens "$ids" --max-new 96 > "$OUT/spec-mtp.txt" 2>&1; echo "mtp rc=$?"
    for m in nodraft suffix mtp; do printf "%-8s " $m; grep -E "^decode  " "$OUT/spec-$m.txt"; grep -E "^speculation" "$OUT/spec-$m.txt"; done
    for m in nodraft suffix mtp; do grep "^output" "$OUT/spec-$m.txt" | md5sum; done ;;
  server)
    $PY - "$OUT/server.json" <<EOF
import json, sys
json.dump({"exe": "$R/build/xe/strata", "args": ["--pack", "$PACK", "--native", "$S1", "--ple-gguf", "$PLE",
  "--expert-profile", "$PROF", "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
  "--max-context", "8192", "--mtp", "$MTP"], "cwd": "$R", "tokenizer": "$PACK/tokenizer", "model_name": "$TAG",
  "log": "$OUT/server-engine.log", "lib_dirs": [], "port": 8095}, open(sys.argv[1], "w"), indent=1)
EOF
    : > "$OUT/server-engine.log"
    setsid $PY serve/server.py --engine strata --config "$OUT/server.json" --port 8095 > "$OUT/server.log" 2>&1 < /dev/null &
    spid=$!
    for i in $(seq 1 120); do curl -sf http://127.0.0.1:8095/v1/models > /dev/null && break; sleep 5; done
    timeout 1200 $PY "$S/server_check.py" > "$OUT/server_check.txt" 2>&1; echo "check rc=$?"
    kill -- -"$spid" 2>/dev/null; sleep 5; kill -9 -- -"$spid" 2>/dev/null
    cat "$OUT/server_check.txt"; grep "strata serve: prompt" "$OUT/server-engine.log" | cut -c1-200 ;;
  long)
    [ -e "$OUT/long_ids.txt" ] || $PY bench/results/2026-09-30-xe-iq2xs/long_prompt.py "$PACK" "$OUT/long_ids.txt"
    L=(--tokens-file "$OUT/long_ids.txt" --max-new 24 --max-context 32768 --kv int8 --spec 4 --prefill auto)
    timeout 1500 ./build/xe/strata "${BASE[@]}" "${L[@]}" --kv-resident 20480 > "$OUT/long-kvstream-suffix.txt" 2>&1; echo "suffix rc=$?"
    timeout 1500 ./build/xe/strata "${BASE[@]}" "${L[@]}" --kv-resident 20480 --mtp "$MTP" > "$OUT/long-kvstream-mtp.txt" 2>&1; echo "mtp rc=$?"
    timeout 1500 ./build/xe/strata "${BASE[@]}" "${L[@]}" --kv-resident 20480 --mtp "$MTP" --mtp-window 8192 > "$OUT/long-kvstream-mtp-ring.txt" 2>&1; echo "ring rc=$?"
    timeout 1500 ./build/xe/strata "${BASE[@]}" "${L[@]}" --mtp "$MTP" > "$OUT/long-resident-mtp.txt" 2>&1; echo "resident rc=$?"
    for m in kvstream-suffix kvstream-mtp kvstream-mtp-ring resident-mtp; do printf "%-18s " $m; grep -E "^output" "$OUT/long-$m.txt" | cut -c1-80; grep -E "^decode  |^prefill  " "$OUT/long-$m.txt"; done ;;
  esac
done
