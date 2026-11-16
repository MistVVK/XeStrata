"""serve/test_stream_terminal.py - a stream that fails after its 200 ends with the API's own terminal frame (#5633, #4829).

    python -m pytest serve/test_stream_terminal.py -q

Chat completions: the error object, then [DONE] (no chunk with a finish_reason outside OpenAI's list: the SDKs raise
on the error object).  Every failure class (the engine's death, its ERR line, a refusal
that comes after the headers, an unexpected error in the server) ends that way on all three routes, never as HTTP
text inside the body or as a silent end; the next request is served.
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, EngineDied, EngineStarting, EngineStuck, GpuBusy, MockEngine,  # noqa: E402
                          ModelBusy, Service, serve)

ROOT = Path(__file__).resolve().parents[1]
ANSWER = "</think>\n\n" + "ab" * 200


class FailingEngine(MockEngine):
    """Streams a few tokens of its first answer, then raises `fail()`; every later answer is whole."""

    def __init__(self, tok, fail):
        super().__init__(tok, ANSWER, max_context=4096)
        self.fail, self.armed = fail, True

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if self.armed and i == 20:
                self.armed = False
                raise self.fail()
            yield t


FAILURES = {
    "died": (lambda: EngineDied("the engine stopped unexpectedly (exit code -9)"), "stopped unexpectedly"),
    "err_line": (lambda: ValueError("the engine said: ERR prompt too long for this card"), "ERR prompt too long"),
    "gpu_busy": (lambda: GpuBusy("the engine's slots are not free"), "slots are not free"),
    "starting": (lambda: EngineStarting("the engine is starting"), "the engine is starting"),
    "stuck": (lambda: EngineStuck("the engine could not be stopped"), "could not be stopped"),
    "model_busy": (lambda: ModelBusy("a request is running or queued"), "running or queued"),
    "unexpected": (lambda: RuntimeError("boom in the server"), "the server's log has the details"),
}


class StreamTerminal(unittest.TestCase):
    def start(self, fail):
        tok = ByteTokenizer()
        self.engine = FailingEngine(tok, fail)
        svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()

    def chat_stream(self):
        return self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                                  "max_tokens": 100, "stream": True})

    def test_chat_stream_ends_with_the_error_object_then_done_for_every_failure(self):
        for name, (fail, words) in FAILURES.items():
            with self.subTest(name):
                self.start(fail)
                code, text = self.chat_stream()
                self.assertEqual(code, 200)
                self.assertNotIn("HTTP/1.", text)
                self.assertTrue(text.endswith("data: [DONE]\n\n"), text[-200:])
                frames = [f[len("data: "):] for f in text.split("\n\n") if f.startswith("data: ")]
                self.assertEqual(frames[-1], "[DONE]")
                err = json.loads(frames[-2])                            # the error object is the last data frame
                self.assertEqual(set(err), {"error"})
                self.assertIn(words, err["error"]["message"])
                for f in frames[:-2]:                                   # no made-up finish_reason (coordinator 11-16)
                    finish = json.loads(f)["choices"][0].get("finish_reason")
                    self.assertIn(finish, (None, "stop", "length", "tool_calls"), f)
                code, text = self.post("/v1/chat/completions",
                                       {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                        "max_tokens": 100})
                self.assertEqual(code, 200, text)                       # the next request is served
                self.assertEqual(json.loads(text)["choices"][0]["finish_reason"], "length")
                self.httpd.shutdown()
                self.httpd.server_close()

    def test_messages_stream_ends_with_an_error_event_for_every_failure(self):
        for name, (fail, words) in FAILURES.items():
            with self.subTest(name):
                self.start(fail)
                code, text = self.post("/v1/messages", {"model": "m", "max_tokens": 100, "stream": True,
                                                        "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(code, 200)
                self.assertNotIn("HTTP/1.", text)
                last = text.rstrip("\n").split("\n\n")[-1]
                self.assertTrue(last.startswith("event: error\ndata: "), last)
                err = json.loads(last.split("data: ", 1)[1])
                self.assertEqual(err["type"], "error")
                self.assertIn(words, err["error"]["message"])
                self.httpd.shutdown()
                self.httpd.server_close()

    def test_responses_stream_ends_with_response_failed_for_every_failure(self):
        for name, (fail, words) in FAILURES.items():
            with self.subTest(name):
                self.start(fail)
                code, text = self.post("/v1/responses", {"model": "m", "input": "hi", "max_output_tokens": 100,
                                                         "stream": True})
                self.assertEqual(code, 200)
                self.assertNotIn("HTTP/1.", text)
                last = text.rstrip("\n").split("\n\n")[-1]
                self.assertTrue(last.startswith("event: response.failed\ndata: "), last)
                ev = json.loads(last.split("data: ", 1)[1])
                self.assertIn(words, ev["response"]["error"]["message"])
                self.httpd.shutdown()
                self.httpd.server_close()


if __name__ == "__main__":
    unittest.main()
