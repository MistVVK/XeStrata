#!/bin/bash
# img_ab.sh (run inside the container): the server with setup's config (Vulkan image encoder) and with the CPU
# encoder, each image once, temperature 0, 64 tokens; replies in /data/img-<enc>-<image>.json
cd /xs || exit 1
python3 - <<'PY'
import json
c = json.load(open('xestrata-iq2_xs.json'))
json.dump(c, open('/data/cfg-vulkan.json', 'w'), indent=1)
d = dict(c); d['vision'] = dict(c['vision']['fallback'], mmproj=c['vision']['mmproj'], model=c['vision']['model'])
json.dump(d, open('/data/cfg-cpu.json', 'w'), indent=1)
PY
for enc in vulkan cpu; do
  .venv/bin/python serve/server.py --engine strata --config /data/cfg-$enc.json --port 8095 > /data/serve-$enc.log 2>&1 &
  for _ in $(seq 1 120); do curl -s -m 5 http://127.0.0.1:8095/v1/models | grep -q . && break; sleep 5; done
  for img in landscape portrait; do
    b64=$(base64 -w0 /data/$img.png)
    printf '{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},{"type":"text","text":"What text is written in the image, and what shapes and colors are in it?"}]}],"max_tokens":64,"temperature":0}' "$b64" > /tmp/req.json
    curl -s -m 600 http://127.0.0.1:8095/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/req.json > /data/img-$enc-$img.json
  done
  pid=$(ss -ltnp | grep 8095 | grep -o "pid=[0-9]*" | head -1 | cut -d= -f2); kill "$pid"; wait
done
