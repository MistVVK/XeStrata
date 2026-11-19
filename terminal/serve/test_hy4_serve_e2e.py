"""`strata serve` on a Hy4-Flash run config, end to end, against tools/hy4_mock_engine.py (which speaks the engine line protocol the real strata-hy4 --serve must):
the real serve/server.py process, the real tokenizer code and the model's real chat template, over HTTP (OpenAI and Anthropic routes).  What the real runner adds is the
token ids; everything on this side of the pipe is the product.

    python -m pytest serve/test_hy4_serve_e2e.py
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import hy4_tokenizer  # noqa: E402
import strata_tokenizer as ST  # noqa: E402
from test_hy4_tokenizer import fake_hf_tokenizer  # noqa: E402

SERVER = ROOT / "serve" / "server.py"
MOCK = ROOT / "tools" / "hy4_mock_engine.py"
TEMPLATE = ROOT / "tests" / "data" / "hy4" / "chat_template.jinja"
HY = "｜"
EXTRA_TAGS = ["<tool_calls:opensource>", "</tool_calls:opensource>", "<tool_call:opensource>", "</tool_call:opensource>",
              "<arg_key:opensource>", "</arg_key:opensource>", "<arg_value:opensource>", "</arg_value:opensource>"]
END = f"<{HY}hy_end:opensource{HY}>"


def tokenizer_json() -> dict:
    hf = fake_hf_tokenizer()
    base = max(a["id"] for a in hf["added_tokens"]) + 1
    for i, t in enumerate(EXTRA_TAGS):
        hf["added_tokens"].append({"id": base + i, "content": t, "special": False})
    return hf


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PYTHONPATH", "PYTHONHOME")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["STRATA_NO_PRESET"] = "1"
    return env


class Hy4Serve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.dir = d
        tokdir = d / "tokenizer"
        hf = tokenizer_json()
        # the eos is the added END token (the fake vocabulary's ids differ from the real 120025)
        end_id = next(a["id"] for a in hf["added_tokens"] if a["content"] == END)
        hy4_tokenizer.convert(hf, TEMPLATE.read_text(encoding="utf-8"), tokdir, bos=0, eos=end_id, pad=0)
        vocab = json.loads((tokdir / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        cls.tk = ST.Tokenizer(tokens, (tokdir / "merges.txt").read_text(encoding="utf-8").split("\n"), json.loads((tokdir / "token_type.json").read_text()), "deepseek3",
                              {"eos": end_id})
        cls.end_id = end_id
        cls.script = d / "script.json"
        cls.record = d / "gens.txt"
        py = d / "mock_engine_launcher.py"
        py.write_text(f"import runpy, sys\nsys.argv[0] = {str(MOCK)!r}\nrunpy.run_path({str(MOCK)!r}, run_name='__main__')\n")
        if os.name == "nt":
            exe = d / "strata-hy4.cmd"
            exe.write_text(f'@"{sys.executable}" "{MOCK}" %*\r\n')
        else:
            exe = d / "strata-hy4"
            exe.write_text(f"#!{sys.executable}\nimport runpy, sys\nrunpy.run_path({str(MOCK)!r}, run_name='__main__')\n")
            exe.chmod(0o700)
        model = d / "Hy4-Flash-SQ-64.strata"
        model.write_bytes(b"x")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            cls.port = s.getsockname()[1]
        cfg = {"exe": str(exe), "arch": "hy4flash", "model_name": "hy4-flash-sq-64", "tokenizer": str(tokdir), "cwd": str(d), "log": str(d / "engine.log"), "port": cls.port,
               "args": ["--native", str(model), "--engram", str(d / "e.strata"), "--max-context", "8192", "--script", str(cls.script), "--record", str(cls.record)]}
        cls.cfg_path = d / "strata-hy4-flash-sq-64.json"
        cls.cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        cls.proc = None

    @classmethod
    def tearDownClass(cls):
        cls.stop()
        cls.tmp.cleanup()

    @classmethod
    def stop(cls):
        if cls.proc is not None and cls.proc.poll() is None:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
        cls.proc = None

    def ids_of(self, text: str) -> list[int]:
        return self.tk.encode(text, parse_special=True)

    def start(self, answers: list[list[int]]):
        self.stop()
        self.script.write_text(json.dumps(answers), encoding="utf-8")
        self.record.write_text("", encoding="utf-8")
        type(self).proc = subprocess.Popen([sys.executable, str(SERVER), "--engine", "strata", "--config", str(self.cfg_path), "--host", "127.0.0.1", "--port", str(self.port)],
                                           cwd=str(self.dir), env=clean_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        t0 = time.time()
        while time.time() - t0 < 90:
            if self.proc.poll() is not None:
                self.fail("serve exited with %s: %s" % (self.proc.returncode, (self.proc.stdout.read() or "")[-1500:]))
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
                c.request("GET", "/health")
                r = c.getresponse()
                r.read()
                c.close()
                if r.status == 200:
                    return
            except OSError:
                pass
            time.sleep(0.3)
        self.fail("serve did not answer /health")

    def call(self, path: str, body: dict, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        data = r.read().decode("utf-8")
        c.close()
        return r.status, data

    def gens(self) -> list[str]:
        return [x for x in self.record.read_text(encoding="utf-8").splitlines() if x.strip()]

    # ------------------------------------------------------------------------------------------------------------------------
    def test_a_chat_answer_with_reasoning(self):
        text = "need a greeting</think:opensource>Hello there." + END
        self.start([self.ids_of(text)])
        status, data = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 200})
        self.assertEqual(status, 200, data)
        msg = json.loads(data)["choices"][0]
        self.assertEqual(msg["message"]["content"].strip(), "Hello there.")
        self.assertEqual((msg["message"].get("reasoning_content") or msg["message"].get("reasoning") or "").strip(), "need a greeting")
        self.assertEqual(msg["finish_reason"], "stop")
        # what the engine was sent: Hy4's own template, ids, no BOS added by serve, the generation prompt opens the think block
        sent = self.gens()[-1].split()
        self.assertEqual(sent[0], "GEN")
        ids = [int(x) for x in sent[-1].split(",")]
        prompt = self.tk.decode(ids)
        self.assertTrue(prompt.startswith(f"<{HY}hy_start:opensource{HY}>system<{HY}hy_middle:opensource{HY}><{HY}reasoning_mode:opensource{HY}>reasoning_effort:high"), prompt[:120])
        self.assertTrue(prompt.endswith(f"<{HY}hy_start:opensource{HY}>assistant<{HY}hy_middle:opensource{HY}><think:opensource>"), prompt[-120:])
        self.assertIn("user<", prompt)

    def test_no_think_and_sampling_keys(self):
        self.start([self.ids_of("Hi." + END)])
        status, data = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50, "temperature": 0.7, "top_p": 0.9, "top_k": 40,
                                                          "reasoning_effort": "none", "seed": 5})
        self.assertEqual(status, 200, data)
        self.assertEqual(json.loads(data)["choices"][0]["message"]["content"].strip(), "Hi.")
        sent = self.gens()[-1].split()
        keys = dict(p.split("=") for p in sent[2:-1])
        self.assertEqual((keys["temperature"], keys["top_p"], keys["top_k"], keys["seed"]), ("0.7", "0.9", "40", "5"))
        prompt = self.tk.decode([int(x) for x in sent[-1].split(",")])
        self.assertTrue(prompt.endswith("<think:opensource></think:opensource>"), prompt[-80:])
        self.assertIn("reasoning_effort:no_think", prompt)

    def test_a_tool_call_in_hy4s_spelling(self):
        call = ("<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Tokyo</arg_value:opensource>"
                "<arg_key:opensource>days</arg_key:opensource><arg_value:opensource>3</arg_value:opensource></tool_call:opensource></tool_calls:opensource>")
        self.start([self.ids_of("I will look it up.</think:opensource>" + call + END)])
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "d", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}}}}}]
        status, data = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "weather in Tokyo"}], "tools": tools, "max_tokens": 300})
        self.assertEqual(status, 200, data)
        choice = json.loads(data)["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        fn = choice["message"]["tool_calls"][0]["function"]
        self.assertEqual(fn["name"], "get_weather")
        self.assertEqual(json.loads(fn["arguments"]), {"city": "Tokyo", "days": 3})
        # the prompt documents the tools the way Hy4 was trained
        prompt = self.tk.decode([int(x) for x in self.gens()[-1].split()[-1].split(",")])
        self.assertIn("<tools>\n{\"name\": \"get_weather\"", prompt)

    def test_a_tool_result_goes_back_in_hy4s_spelling(self):
        self.start([self.ids_of("It is sunny.</think:opensource>Sunny." + END)])
        msgs = [{"role": "user", "content": "weather?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny, 24C"}]
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
        status, data = self.call("/v1/chat/completions", {"messages": msgs, "tools": tools, "max_tokens": 100})
        self.assertEqual(status, 200, data)
        self.assertIn("Sunny.", json.loads(data)["choices"][0]["message"]["content"])
        prompt = self.tk.decode([int(x) for x in self.gens()[-1].split()[-1].split(",")])
        self.assertIn("<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Tokyo</arg_value:opensource>", prompt)
        self.assertIn("<tool_response:opensource>sunny, 24C</tool_response:opensource>", prompt.replace("\n", ""))

    def test_the_anthropic_route_and_streaming(self):
        self.start([self.ids_of("plan</think:opensource>Streamed answer." + END)])
        status, data = self.call("/v1/messages", {"model": "hy4-flash", "max_tokens": 100, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                                 {"anthropic-version": "2023-06-01"})
        self.assertEqual(status, 200, data)
        self.assertIn("event: message_start", data)
        deltas = [json.loads(x[6:])["delta"] for x in data.splitlines() if x.startswith("data: ") and '"content_block_delta"' in x]
        self.assertEqual("".join(d["text"] for d in deltas if d["type"] == "text_delta"), "Streamed answer.")
        self.assertEqual("".join(d["thinking"] for d in deltas if d["type"] == "thinking_delta"), "plan")
        self.assertIn("message_stop", data)
        self.assertNotIn("<think", data)

    def test_logprobs_are_refused_with_a_reason(self):
        self.start([self.ids_of("Hi." + END)])
        status, data = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "logprobs": True, "top_logprobs": 2})
        self.assertEqual(status, 400, data)
        self.assertIn("Hy4-Flash", data)

    def test_the_start_line_has_none_of_the_flash_next_flags(self):
        self.start([self.ids_of("Hi." + END)])
        log = (self.dir / "engine.log").read_text(encoding="utf-8", errors="replace") if (self.dir / "engine.log").exists() else ""
        line = next((x for x in log.splitlines() if "engine started" in x), "")
        self.assertIn("--serve --native", line)
        for flag in ("--vram-reserve-mib", "--pool-workers", "--resident-budget-gib", "--batch", "--mtp", "--draft"):
            self.assertNotIn(flag, line)


if __name__ == "__main__":
    unittest.main()
