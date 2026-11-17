#!/usr/bin/env python3
"""prompt_check.py PORT: one real question to the running server; exit 0 when the answer names Canberra.  Standard library only."""
import json
import sys
import time
import urllib.request

port = sys.argv[1]
body = {"messages": [{"role": "user", "content": "What is the capital city of Australia? Answer with the city name only."}], "max_tokens": 400, "temperature": 0}
request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
started = time.time()
with urllib.request.urlopen(request, timeout=900) as response:
    data = json.load(response)
message = data["choices"][0]["message"]
text = (message.get("content") or "") + " " + (message.get("reasoning_content") or "")
print(f"answer ({time.time() - started:.1f}s, {data.get('usage', {}).get('completion_tokens', '?')} tokens): {text.strip()[-160:]!r}")
sys.exit(0 if "canberra" in text.lower() else 1)
