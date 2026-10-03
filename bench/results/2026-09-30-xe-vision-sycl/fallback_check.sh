#!/bin/bash
# run.sh CONFIG TAG [server args...]: start, print /v1/status vision + the encoder's oneDNN env, one image request, stop
F=$(dirname "$0"); cfg=$1; tag=$2; shift 2
cd <repo>
setsid <data>/venv/bin/python serve/server.py --engine strata --config $cfg --port 8096 "$@" > $F/server-$tag.log 2>&1 < /dev/null &
for i in $(seq 1 150); do curl -sf http://127.0.0.1:8096/v1/models > /dev/null && break; sleep 5; done
curl -s http://127.0.0.1:8096/v1/status | python3 -c "import json,sys; print('$tag status vision:', json.load(sys.stdin)['vision'])"
for p in $(pgrep -f "^<repo>/engine/strata-vision"); do echo "$tag encoder $(tr '\0' '\n' < /proc/$p/cmdline | head -1): $(tr '\0' '\n' < /proc/$p/environ | grep -E '^GGML_SYCL_(ENABLE_DNN|FA_ONEDNN)=' | tr '\n' ' ')"; done
S=<scratch>
curl -s -m 600 http://127.0.0.1:8096/v1/chat/completions -H 'Content-Type: application/json' -d @$S/vision/req-landscape.json > $F/resp-$tag.json
python3 -c "import json; d=json.load(open('$F/resp-$tag.json')); m=d['choices'][0]['message']; print('$tag reply:', repr((m.get('reasoning_content') or '')[:90]))"
grep "\[strata\] images" $F/server-$tag.log
srv=$(pgrep -f "^<data>/venv/bin/python serve/server.py"); kill -- -$(ps -o pgid= -p $srv | tr -d ' ')
for i in $(seq 1 30); do pgrep -f "^<repo>/build/xe/strata" > /dev/null || break; sleep 1; done
