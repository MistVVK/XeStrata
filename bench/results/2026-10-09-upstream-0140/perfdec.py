# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""perfdec.py PORT OUT: decode speed only - one warm-up chat, then six chats of 512 generated tokens each (greedy, no
thinking); prints the total generated tokens over the total decode time, and the per-chat rates"""
import json
import sys
import urllib.request

PORT, OUT = int(sys.argv[1]), sys.argv[2]


def ask(content, max_tokens):
    body = {'model': 'x', 'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_tokens,
            'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


ask('Say hello.', 32)
qs = ('Write a short story about a lighthouse keeper.', 'Explain how a refrigerator works, step by step.',
      'Describe the water cycle in detail.', 'Write a Python function that merges two sorted lists, and explain it.',
      'List the planets of the solar system with one fact about each.', 'Explain the causes of the French Revolution.')
runs = [ask(q, 512)['timings'] for q in qs]
json.dump(runs, open(OUT, 'w'), indent=1)
n = sum(t['predicted_n'] for t in runs)
ms = sum(t['predicted_ms'] for t in runs)
print(f'prompt -; decode {n / ms * 1000:.2f} tok/s over {n} tokens; per chat '
      + ' '.join(f'{t["predicted_per_second"]:.1f}' for t in runs))
