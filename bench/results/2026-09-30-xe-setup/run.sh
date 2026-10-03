#!/bin/bash
# The config setup.py wrote, started as run-iq2_xs.sh does (without --open), in a clean environment
S=<scratch>; V=$S/vision
cd <repo>
setsid env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 .venv/bin/python serve/server.py --engine strata --config strata-iq2_xs.json --port 8095 > $S/setup-run/server.log 2>&1 < /dev/null &
for i in $(seq 1 150); do curl -sf http://127.0.0.1:8095/v1/models > /dev/null && break; sleep 5; done
tail -3 $S/setup-run/server.log
for req in $V/req-landscape.json $S/visfail/text.json; do
  curl -s -m 900 http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' -d @$req | python3 -c "import json,sys; d=json.load(sys.stdin); m=d['choices'][0]['message']; print(d.get('usage', {}).get('prompt_tokens'), repr((m.get('reasoning_content') or '')[:150]))"
done
srv=$(pgrep -f "^.venv/bin/python serve/server.py|^<repo>/.venv/bin/python serve/server.py"); kill -- -$(ps -o pgid= -p $srv | tr -d ' ')
for i in $(seq 1 30); do pgrep -f "^<repo>/engine/strata" > /dev/null || break; sleep 1; done
echo "left: $(pgrep -f '^<repo>/engine/strata' | wc -l)"
