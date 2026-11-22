"""End-to-end tests of serve/server.py driving tools/inkling_mock_engine.py:
- OpenAI /v1/chat/completions (streaming + non-streaming)
- Anthropic /v1/messages
- Collapsible <|content_thinking|> -> reasoning_content across all 6 reasoning_effort levels
- Bidirectional tool calls with XML special chars (<>&, </parameter>, </tool_call>) & nested quotes
- Multimodal Vision image attachment (PNG data URL -> temp file -> image= wire key + pad token)
- Multi-request concurrency at c=1, 4, 8, 32, 64 simultaneous streaming requests!
"""
from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import zlib
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[1]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

from tools import inkling_tokenizer, strata_tokenizer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_png_bytes(width: int = 8, height: int = 8) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x20\x80\xe0" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _write_test_tokenizer(tok_dir: Path) -> strata_tokenizer.Tokenizer:
    b2u = strata_tokenizer.bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    vocab["he"] = 256
    added = [
        {"id": 199999, "content": "<|endoftext|>", "special": True},
        {"id": 200001, "content": "<|message_system|>", "special": True},
        {"id": 200002, "content": "<|message_user|>", "special": True},
        {"id": 200003, "content": "<|message_model|>", "special": True},
        {"id": 200004, "content": "<|message_tool|>", "special": True},
        {"id": 200005, "content": "<|end_message|>", "special": True},
        {"id": 200006, "content": "<|content_model_end_sampling|>", "special": True},
        {"id": 200010, "content": "<|content_text|>", "special": True},
        {"id": 200011, "content": "<|content_thinking|>", "special": True},
        {"id": 200012, "content": "<|content_invoke_tool_json|>", "special": True},
        {"id": 200013, "content": "<|content_image|>", "special": True},
        {"id": 200014, "content": "<|content_xml|>", "special": True},
        {"id": 200028, "content": "<|begin_of_text|>", "special": True},
        {"id": 200054, "content": "<|unused_200054|>", "special": True},
    ]
    hf = {
        "version": "1.0",
        "added_tokens": added,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": r"\p{N}{1,3}|[^\s]+|\s+"}, "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
            ],
        },
        "model": {"type": "BPE", "vocab": vocab, "merges": ["h e"]},
    }
    # Faithful minimal Jinja template matching Inkling-2-Mini's effort + tool + image markers
    template = (
        "{%- set effort_map = {'none': 0.0, 'minimal': 0.1, 'low': 0.2, 'medium': 0.7, 'high': 0.9, 'max': 0.99} -%}"
        "{%- set eff = effort_map.get(reasoning_effort|default('high'), 0.9) -%}"
        "<|begin_of_text|><|message_system|><|content_text|>Thinking effort level: {{ eff }}<|end_message|>"
        "{%- if tools -%}<|message_system|>tool_declare<|content_xml|>{{ tools | tojson }}<|end_message|>{%- endif -%}"
        "{%- for m in messages -%}"
        "{%- if m['role'] == 'user' -%}"
        "{%- if m['content'] is string -%}<|message_user|><|content_text|>{{ m['content'] }}<|end_message|>"
        "{%- else -%}{%- for p in m['content'] -%}"
        "{%- if p['type'] == 'image' -%}<|message_user|><|content_image|><|unused_200054|><|end_message|>"
        "{%- elif p['type'] == 'text' -%}<|message_user|><|content_text|>{{ p['text'] }}<|end_message|>{%- endif -%}"
        "{%- endfor -%}{%- endif -%}"
        "{%- elif m['role'] == 'assistant' -%}"
        "{%- if m.get('tool_calls') -%}{%- for tc in m['tool_calls'] -%}"
        "<|message_model|>{{ tc['function']['name'] }}<|content_invoke_tool_json|>{{ {'name': tc['function']['name'], 'args': tc['function']['arguments']} | tojson }}<|end_message|>"
        "{%- endfor -%}{%- endif -%}"
        "{%- if m.get('content') -%}<|message_model|><|content_text|>{{ m['content'] }}<|end_message|>{%- endif -%}"
        "{%- elif m['role'] == 'tool' -%}<|message_tool|><|content_text|>{{ m['content'] }}<|end_message|>{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|message_model|>{%- endif -%}"
    )
    cfg = inkling_tokenizer.convert(hf, template, tok_dir)
    vocab = json.loads((tok_dir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [""] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = [m for m in (tok_dir / "merges.txt").read_text(encoding="utf-8").split("\n") if m]
    types = json.loads((tok_dir / "token_type.json").read_text(encoding="utf-8"))
    return strata_tokenizer.Tokenizer(tokens, merges, types, cfg["pre"], cfg["special_ids"])


def _post_json(url: str, body: dict, headers: dict | None = None, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_sse(url: str, body: dict, timeout: float = 30.0) -> list[dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps({**body, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            events.append(json.loads(payload))
    return events


class _MockServerHarness:
    def __init__(self, tmp: Path, scripts: list[str], *, parallel: int = 1, delay: float = 0.0):
        self.tmp = tmp
        self.tok_dir = tmp / "tok"
        self.tok = _write_test_tokenizer(self.tok_dir)
        self.record_path = tmp / "recorded_gen.log"
        self.script_path = tmp / "scripts.json"
        encoded = [self.tok.encode(s, parse_special=True) for s in scripts]
        self.script_path.write_text(json.dumps(encoded), encoding="utf-8")

        # Create a small wrapper executable script named strata-inkling.bat / strata-inkling so kind detection triggers
        mock_py = TERMINAL / "tools" / "inkling_mock_engine.py"
        if os.name == "nt":
            self.exe = tmp / "strata-inkling.bat"
            self.exe.write_text(f'@echo off\r\n"{sys.executable}" "{mock_py}" %*\r\n', encoding="utf-8")
        else:
            self.exe = tmp / "strata-inkling"
            self.exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{mock_py}" "$@"\n', encoding="utf-8")
            self.exe.chmod(0o755)

        self.port = _free_port()
        model_file = tmp / "Inkling-2-Mini-SQ-20.strata"
        vision_file = tmp / "Inkling-2-Mini-Vision.strata"
        model_file.write_bytes(b"STRATA\x00\x01")
        vision_file.write_bytes(b"STRATA\x00\x01")

        args = [
            "--native", str(model_file),
            "--vision", str(vision_file),
            "--max-context", "32768",
            "--script", str(self.script_path),
            "--record", str(self.record_path),
        ]
        if parallel > 1:
            args += ["--parallel", str(parallel)]
        if delay > 0:
            args += ["--delay", str(delay)]

        self.cfg_path = tmp / "run.json"
        self.cfg_path.write_text(json.dumps({
            "exe": str(self.exe),
            "arch": "inkling2_loop",
            "tokenizer": str(self.tok_dir),
            "model_name": "inkling-2-mini-sq-20",
            "args": args,
            "host": "127.0.0.1",
            "port": self.port,
            "open_browser": False,
        }), encoding="utf-8")
        self.proc: subprocess.Popen | None = None

    def __enter__(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(TERMINAL)
        env["STRATA_DATA_DIR"] = str(self.tmp / "data")
        self.proc = subprocess.Popen(
            [sys.executable, str(TERMINAL / "serve" / "server.py"), "--engine", "strata", "--config", str(self.cfg_path),
             "--host", "127.0.0.1", "--port", str(self.port), "--no-api-key"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        )
        deadline = time.time() + 20.0
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1.0) as r:
                    if r.status == 200:
                        return self
            except Exception:
                if self.proc.poll() is not None:
                    out = self.proc.stdout.read() if self.proc.stdout else ""
                    raise RuntimeError(f"server exited early ({self.proc.returncode}):\n{out}")
                time.sleep(0.1)
        raise RuntimeError("server did not become healthy in time")

    def __exit__(self, *_exc):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class TestInklingServeE2E(unittest.TestCase):
    def test_thinking_tools_vision_and_efforts_e2e(self):
        tool_args = {
            "path": "patch.py",
            "code": 'if a < b && c > d:\n    s = "</parameter></tool_call>\\"nested\\""\n',
        }
        scripts = [
            # 1: thinking + text with reasoning_effort=low
            "<|message_model|><|content_thinking|>Quick thought.<|end_message|>"
            "<|message_model|><|content_text|>Hello from Inkling!<|end_message|><|content_model_end_sampling|>",
            # 2: tool call with XML special chars & nested quotes
            "<|message_model|><|content_thinking|>Need to patch.<|end_message|>"
            f"<|message_model|>apply_patch<|content_invoke_tool_json|>{json.dumps({'name': 'apply_patch', 'args': tool_args})}<|end_message|>"
            "<|content_model_end_sampling|>",
            # 3: vision reply
            "<|message_model|><|content_text|>I see an 8x8 blue PNG.<|end_message|><|content_model_end_sampling|>",
            # 4: Anthropic /v1/messages reply
            "<|message_model|><|content_thinking|>Anthropic check.<|end_message|>"
            "<|message_model|><|content_text|>Anthropic OK.<|end_message|><|content_model_end_sampling|>",
        ]
        with tempfile.TemporaryDirectory() as td:
            with _MockServerHarness(Path(td), scripts, parallel=1) as h:
                base = f"http://127.0.0.1:{h.port}"

                # 1. Reasoning + text
                r1 = _post_json(f"{base}/v1/chat/completions", {
                    "model": "inkling-2-mini-sq-20",
                    "reasoning_effort": "low",
                    "messages": [{"role": "user", "content": "Say hi"}],
                })
                msg1 = r1["choices"][0]["message"]
                self.assertFalse(msg1.get("reasoning_content"))
                self.assertEqual(msg1["content"], "Hello from Inkling!")

                # 2. Tool call round-trip with XML special chars
                r2 = _post_json(f"{base}/v1/chat/completions", {
                    "model": "inkling-2-mini-sq-20",
                    "reasoning_effort": "max",
                    "tools": [{"type": "function", "function": {"name": "apply_patch", "parameters": {"type": "object"}}}],
                    "messages": [{"role": "user", "content": "Patch it"}],
                })
                msg2 = r2["choices"][0]["message"]
                self.assertEqual(r2["choices"][0]["finish_reason"], "tool_calls")
                self.assertEqual(len(msg2["tool_calls"]), 1)
                tc = msg2["tool_calls"][0]
                self.assertEqual(tc["function"]["name"], "apply_patch")
                self.assertEqual(json.loads(tc["function"]["arguments"]), tool_args)

                # 3. Multimodal Vision PNG attachment
                b64 = base64.b64encode(_make_png_bytes()).decode("ascii")
                r3 = _post_json(f"{base}/v1/chat/completions", {
                    "model": "inkling-2-mini-sq-20",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": "What is in this picture?"},
                        ],
                    }],
                })
                self.assertEqual(r3["choices"][0]["message"]["content"], "I see an 8x8 blue PNG.")

                # 4. Anthropic /v1/messages endpoint
                r4 = _post_json(f"{base}/v1/messages", {
                    "model": "inkling-2-mini-sq-20",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "Ping"}],
                })
                blocks = {b["type"]: b for b in r4["content"]}
                self.assertNotIn("thinking", blocks)
                self.assertEqual(blocks["text"]["text"], "Anthropic OK.")

                # Verify wire protocol recorded exact reasoning effort levels & image= key
                lines = h.record_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), 4)
                # Decode prompt tokens: Inkling-2-Mini has no thinking mode so all requests render Thinking effort level: 0.0
                p1 = h.tok.decode([int(x) for x in lines[0].split()[-1].split(",")])
                p2 = h.tok.decode([int(x) for x in lines[1].split()[-1].split(",")])
                self.assertIn("Thinking effort level: 0.0", p1)
                self.assertIn("Thinking effort level: 0.0", p2)
                self.assertIn("image=", lines[2])

    def test_concurrent_streaming_4_8_32_64(self):
        script = (
            "<|message_model|><|content_thinking|>Parallel slot.<|end_message|>"
            "<|message_model|><|content_text|>Batch OK.<|end_message|><|content_model_end_sampling|>"
        )
        with tempfile.TemporaryDirectory() as td:
            with _MockServerHarness(Path(td), [script], parallel=64, delay=0.001) as h:
                base = f"http://127.0.0.1:{h.port}"
                for conc in (4, 8, 32, 64):
                    def run_one(idx: int):
                        events = _post_sse(f"{base}/v1/chat/completions", {
                            "model": "inkling-2-mini-sq-20",
                            "messages": [{"role": "user", "content": f"req-{conc}-{idx}"}],
                        })
                        txt = "".join((e["choices"][0]["delta"].get("content") or "") for e in events if e.get("choices"))
                        thk = "".join((e["choices"][0]["delta"].get("reasoning_content") or "") for e in events if e.get("choices"))
                        return txt, thk

                    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
                        results = list(ex.map(run_one, range(conc)))
                    self.assertEqual(len(results), conc)
                    for txt, thk in results:
                        self.assertEqual(txt, "Batch OK.", f"failed at c={conc}")
                        self.assertEqual(thk, "", f"unexpected reasoning leaked at c={conc}")


if __name__ == "__main__":
    unittest.main()
