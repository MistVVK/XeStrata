#!/bin/bash
# The encoder killed while the server runs: an image request fails with the reason, a text request still works.
S=<scratch>; V=$S/vision
cd <repo>
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export ZES_ENABLE_SYSMAN=1
ask() { curl -s -m 600 http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' -d @$1 | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get('error') or {'content': d['choices'][0]['message'].get('content'), 'reasoning': (d['choices'][0]['message'].get('reasoning_content') or '')[:80], 'usage': d.get('usage')})[:300])"; }
setsid <data>/venv/bin/python serve/server.py --engine strata --config $S/visfail/cfg-kill.json --port 8095 > $S/visfail/server-kill.log 2>&1 < /dev/null &
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8095/v1/models > /dev/null && break; sleep 5; done
echo -n "image before the kill: "; ask $V/req-portrait.json
enc=$(pgrep -f "^<repo>/build-vision-cpu/bin/strata-vision"); echo "killing the encoder (pid $enc)"; kill -9 $enc; sleep 1
echo -n "image after the kill:  "; ask $V/req-landscape.json
echo -n "text after the kill:   "; ask $S/visfail/text.json
echo -n "status: "; curl -s http://127.0.0.1:8095/v1/status | python3 -c "import json,sys; print(json.load(sys.stdin)['vision'])"
srv=$(pgrep -f "^<data>/venv/bin/python serve/server.py"); kill -- -$(ps -o pgid= -p $srv | tr -d ' ')
for i in $(seq 1 30); do pgrep -f "^<repo>/build/xe/strata" > /dev/null || break; sleep 1; done
echo "left: $(pgrep -f '^<repo>/build/xe/strata|^<data>/venv/bin/python serve/server.py' | wc -l)"
