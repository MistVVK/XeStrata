# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""perf.py PORT OUT [TOKENS]: prompt-read and decode speed against a running server.  Three requests, each a fresh
document of about TOKENS tokens (default 8000; a different first line each, so nothing is reused) and 256 generated
tokens, greedy, no thinking; then three short chats (decode only).  Writes the timings as JSON and prints the
medians."""
import json
import statistics
import sys
import time
import urllib.request

PORT, OUT = int(sys.argv[1]), sys.argv[2]
TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 8000


def ask(content, max_tokens):
    body = {'model': 'x', 'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_tokens,
            'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


items = max(1, TOKENS // 27)
long_runs, short_runs = [], []
for k in range(3):
    doc = f'Run {time.time_ns()} #{k}.\n' + ' '.join(
        f'Item {i}: the warehouse holds {(i * 7 + k) % 101} crates of type {chr(65 + i % 26)}.' for i in range(items))
    t = ask(doc + '\n\nSummarize the inventory in detail, item by item.', 256)['timings']
    long_runs.append(t)
for q in ('Write a short story about a lighthouse keeper.', 'Explain how a refrigerator works, step by step.',
          'Describe the water cycle in detail.'):
    short_runs.append(ask(q, 256)['timings'])
res = {'long': long_runs, 'short': short_runs}
json.dump(res, open(OUT, 'w'), indent=1)
pp = statistics.median(t['prompt_per_second'] for t in long_runs)
dl = statistics.median(t['predicted_per_second'] for t in long_runs)
ds = statistics.median(t['predicted_per_second'] for t in short_runs)
print(f'prompt {long_runs[0]["prompt_n"]} tok: {pp:.1f} tok/s; decode after it {dl:.2f}; short decode {ds:.2f}')
