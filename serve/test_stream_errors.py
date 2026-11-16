"""An error after a stream has started: each API's own last frame, never a second HTTP response, never silence.

    python -m unittest serve.test_stream_errors -v
"""
import contextlib
import http.client
import io
import json
import types
import unittest
from pathlib import Path

from serve import server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
MESSAGES = [{"role": "user", "content": "hi"}]
CHAT = ("/v1/chat/completions", {"messages": MESSAGES, "stream": True})
ANTHROPIC = ("/v1/messages", {"model": "m", "messages": MESSAGES, "max_tokens": 9, "stream": True})
RESPONSES = ("/v1/responses", {"input": "hi", "stream": True})


class FailsAfterOneToken(MockEngine):
    failure = None

    def generate(self, *args, **kwargs):
        tokens = list(super().generate(*args, **kwargs))
        yield tokens[0]
        if self.failure is not None:
            raise self.failure
        yield from tokens[1:]


def frames(body):
    return [f for f in body.decode().split("\n\n") if f]


def data_of(frame):
    return json.loads(frame.partition("data: ")[2])


class StreamErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = FailsAfterOneToken(tok, "partial answer", max_context=131072)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def stream(self, api, failure):
        self.engine.failure = failure
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=15)
        try:
            c.request("POST", api[0], json.dumps(api[1]), {"Content-Type": "application/json"})
            r = c.getresponse()
            body = r.read()
        finally:
            c.close()
        self.assertEqual(r.status, 200, body)
        self.assertNotIn(b"HTTP/1.", body)
        return frames(body)

    def chat_error(self, failure):
        got = self.stream(CHAT, failure)
        self.assertEqual(got[-1], "data: [DONE]")
        return data_of(got[-2])["error"]

    def anthropic_error(self, failure):
        got = self.stream(ANTHROPIC, failure)
        self.assertTrue(got[-1].startswith("event: error\n"), got[-1])
        event = data_of(got[-1])
        self.assertEqual(event["type"], "error")
        return event["error"]

    def responses_error(self, failure):
        got = self.stream(RESPONSES, failure)
        self.assertTrue(got[-1].startswith("event: response.failed\n"), got[-1])
        response = data_of(got[-1])["response"]
        self.assertEqual(response["status"], "failed")
        return response["error"]

    def test_a_refusal_after_the_start_is_the_apis_own_frame(self):
        for failure in (server.GpuBusy("busy now", retry_after=5), server.EngineStuck("stuck now"),
                        server.EngineStarting("starting now"), server.ModelBusy("switching now")):
            with self.subTest(type(failure).__name__):
                self.assertEqual(self.chat_error(failure), {"type": "server_error", "message": str(failure)})
                self.assertEqual(self.responses_error(failure), {"code": "server_error", "message": str(failure)})
                self.assertEqual(self.anthropic_error(failure),
                                 {"type": "overloaded_error" if isinstance(failure, server.GpuBusy) else "api_error",
                                  "message": str(failure)})

    def test_any_other_failure_is_a_frame_and_keeps_its_details_in_the_log(self):
        failure = RuntimeError("a secret detail")
        for read_error in (self.chat_error, self.anthropic_error, self.responses_error):
            with self.subTest(read_error.__name__):
                log = io.StringIO()
                with contextlib.redirect_stdout(log):      # the handler's thread prints to this sys.stdout too
                    error = read_error(failure)
                self.assertIn("server's log", error["message"])
                self.assertNotIn("secret", error["message"])
                self.assertIn("[strata] error in a started stream", log.getvalue())
                self.assertIn("RuntimeError: a secret detail", log.getvalue())

    def test_a_client_that_left_gets_no_frame(self):
        for api in (CHAT, ANTHROPIC, RESPONSES):
            with self.subTest(api[0]):
                body = "\n\n".join(self.stream(api, server.ClientGone()))
                for frame in ("[DONE]", "event: error", "response.failed"):
                    self.assertNotIn(frame, body)

    def test_a_stream_that_does_not_fail_ends_as_before(self):
        self.assertEqual(self.stream(CHAT, None)[-1], "data: [DONE]")
        self.assertTrue(self.stream(ANTHROPIC, None)[-1].startswith("event: message_stop\n"))
        self.assertTrue(self.stream(RESPONSES, None)[-1].startswith("event: response.completed\n"))

    def test_a_frame_the_client_cannot_take_is_a_disconnect(self):
        def refuses(_data):
            raise BrokenPipeError()
        for send in (lambda h, s: h._chat_failed(s, "x"), lambda h, s: h._anthropic_failed(s, "x", "api_error")):
            noted = {}
            fake = types.SimpleNamespace(wfile=types.SimpleNamespace(write=refuses),
                                         _note=lambda **values: noted.update(values))
            send(self.httpd.RequestHandlerClass, fake)
            self.assertEqual(noted["outcome"], "disconnected")
            self.assertEqual(noted["error"]["message"], "x")     # still a server error: the monitor's state is "error"


if __name__ == "__main__":
    unittest.main()
