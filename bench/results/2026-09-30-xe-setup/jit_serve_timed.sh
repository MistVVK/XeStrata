#!/bin/bash
# serve_timed.sh TAG: start the server with strata-iq2_xs.json, time readiness and three requests, stop
J=$(dirname "$0"); S=$(dirname "$J"); tag=$1; cd <repo>
t0=$(date +%s.%N)
setsid env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 .venv/bin/python serve/server.py --engine strata --config strata-iq2_xs.json --port 8097 > $J/server-$tag.log 2>&1 < /dev/null &
for i in $(seq 1 300); do curl -sf http://127.0.0.1:8097/v1/models > /dev/null && break; sleep 1; done
t1=$(date +%s.%N); echo "$tag ready after $(awk "BEGIN{printf \"%.1f\", $t1-$t0}") s"
for req in $J/req-text.json $J/req-text2.json $S/vision/req-landscape.json; do
  a=$(date +%s.%N); curl -s -m 900 http://127.0.0.1:8097/v1/chat/completions -H 'Content-Type: application/json' -d @$req > $J/resp-$tag-$(basename $req); b=$(date +%s.%N)
  echo "$tag $(basename $req .json) $(awk "BEGIN{printf \"%.2f\", $b-$a}") s, prompt $(python3 -c "import json;print(json.load(open('$J/resp-$tag-$(basename $req)')).get('usage',{}).get('prompt_tokens'))")"
done
srv=$(pgrep -f "^.venv/bin/python serve/server.py --engine strata --config strata-iq2_xs.json --port 8097"); kill -- -$(ps -o pgid= -p $srv | tr -d ' ')
for i in $(seq 1 30); do pgrep -f "^<repo>/engine/strata" > /dev/null || break; sleep 1; done
