"""Real-model server checks: a streamed reply, a stream cancelled after a few chunks followed by a new request on
the same engine, and a three-turn conversation (the prompt cache).  Prints what each step saw."""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8095/v1/chat/completions"


def post(body, stream=False):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def chat(messages, max_tokens=64):
    t0 = time.time()
    with post({"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0}) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"], out.get("usage", {}), time.time() - t0


# 1. streaming
t0 = time.time()
chunks, text = 0, ""
with post({"model": "x", "stream": True, "max_tokens": 64, "temperature": 0,
           "messages": [{"role": "user", "content": "Name three primary colors."}]}) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        delta = d["choices"][0].get("delta", {})
        piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
        if piece:
            chunks += 1
            text += piece
print(f"1. stream: {chunks} chunks in {time.time() - t0:.1f} s: {text[-160:]!r}")

# 2. cancel mid-stream, then a new request on the same engine
with post({"model": "x", "stream": True, "max_tokens": 400, "temperature": 0,
           "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper."}]}) as r:
    n = 0
    for line in r:
        if line.decode().startswith("data:"):
            n += 1
        if n >= 8:
            break   # closing the connection cancels the request
print(f"2. cancelled after {n} chunks")
time.sleep(1)
reply, usage, dt = chat([{"role": "user", "content": "What is 17 + 25? Answer with the number only."}], 256)
print(f"2. after the cancel: {dt:.1f} s, usage {usage}: {reply[-120:]!r}")

# 3. three turns (the conversation cache)
msgs = [{"role": "user", "content": "My name is Aiko. Remember it."}]
for turn, follow in enumerate(["What is my name?", "Spell my name backwards."], 1):
    reply, usage, dt = chat(msgs, 256)
    print(f"3. turn {turn}: {dt:.1f} s, usage {usage}: {reply[-120:]!r}")
    msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": follow}]
reply, usage, dt = chat(msgs, 256)
print(f"3. turn 3: {dt:.1f} s, usage {usage}: {reply[-160:]!r}")
