#!/bin/bash
# vision_model_check.sh ENVFILE MMPROJ: one model's image requests through the server (CPU encoder), then the
# landscape and portrait prompts through llama-mtmd-cli on the CPU for comparison.
set -u
source "$1"; MMPROJ=$2
R=<repo>; S=<scratch>; V=$S/vision
OUT=$S/models/$TAG/vision; mkdir -p "$OUT"; PY=<data>/venv/bin/python
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; set -u; export ZES_ENABLE_SYSMAN=1
cd "$R"
$PY - "$OUT/server.json" <<PYEOF
import json, sys
json.dump({"exe": "$R/build/xe/strata", "args": ["--pack", "$PACK", "--native", "$S1", "--ple-gguf", "$PLE",
  "--expert-profile", "$PROF", "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
  "--max-context", "8192", "--mtp", "<data>/mtp/rt", "--vision", "--vram-reserve-mib", "700"],
  "cwd": "$R", "tokenizer": "$PACK/tokenizer", "model_name": "$TAG", "log": "$OUT/server-engine.log", "lib_dirs": [],
  "port": 8095, "vision": {"exe": "$R/build-vision-cpu/bin/strata-vision", "mmproj": "$MMPROJ", "model": "$S1",
  "gpu": False, "max_tokens": 300, "threads": 8}}, open(sys.argv[1], "w"), indent=1)
PYEOF
: > "$OUT/server-engine.log"
setsid $PY serve/server.py --engine strata --config "$OUT/server.json" --port 8095 > "$OUT/server.log" 2>&1 < /dev/null &
for i in $(seq 1 150); do curl -sf http://127.0.0.1:8095/v1/models > /dev/null && break; sleep 5; done
post() { curl -s -m 900 http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' -d @$1 > "$OUT/$2.json"
  python3 -c "import json; d=json.load(open('$OUT/$2.json')); m=d['choices'][0]['message']; u=d.get('usage',{}); print('$2', u.get('prompt_tokens'), (u.get('prompt_tokens_details') or {}).get('cached_tokens'), repr((m.get('reasoning_content') or '')[:110]), repr((m.get('content') or '')[:110]))"; }
post $V/req-landscape.json landscape
post $V/req-portrait.json portrait
post $V/req-two.json two
post $V/req-turn1.json turn1
python3 - "$OUT" "$V" <<PYEOF
import json, sys
out, v = sys.argv[1], sys.argv[2]
t1 = json.load(open(v + "/req-turn1.json")); a = json.load(open(out + "/turn1.json"))["choices"][0]["message"]
t1["messages"] += [{"role": "assistant", "content": a.get("content") or ""},
                   {"role": "user", "content": "What colour is the circle? Answer briefly."}]
json.dump(t1, open(out + "/req-turn2.json", "w"))
PYEOF
post "$OUT/req-turn2.json" turn2
srv=$(pgrep -f "^<data>/venv/bin/python serve/server.py"); kill -- -$(ps -o pgid= -p $srv | tr -d ' ')
for i in $(seq 1 30); do pgrep -f "^<repo>/build/xe/strata" > /dev/null || break; sleep 1; done
Q="What text is written in the image, and what shapes and colors are in it?"
$PY $V/render.py "$PACK" "$Q" > "$OUT/prompt.txt"
for n in landscape portrait; do
  timeout 1800 $S/llama-ref/build-mtmd/bin/llama-mtmd-cli -m "$S1" --mmproj "$MMPROJ" --image $V/$n.png --image-max-tokens 300 -n 64 --temp 0 \
    --jinja --chat-template "{%- for m in messages -%}{{ m['content'] }}{%- endfor -%}" -p "$(cat $OUT/prompt.txt)" -t 8 > "$OUT/ref-$n.txt" 2> "$OUT/ref-$n.log"
  python3 -c "
import json
x = json.load(open('$OUT/$n.json'))['choices'][0]['message'].get('reasoning_content') or ''
r = open('$OUT/ref-$n.txt').read().strip()
k = next((i for i in range(min(len(x), len(r))) if x[i] != r[i]), min(len(x), len(r)))
print('$n vs llama-mtmd-cli: common prefix', k, 'of', len(x), len(r), '|', repr(x[max(0,k-20):k+30]), '|', repr(r[max(0,k-20):k+30]))"
done
