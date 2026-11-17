"""#6055: a request the server had accepted, cut off because the server stops the engine under it.

The stop path closes the engine (StrataEngine.close: STOP, then QUIT).  The engine ends the request that is still
reading its prompt with DONE ... cancel, and the server used to turn that into a successful reply with no tokens and
finish_reason "length": a client could not tell it from a reply that ran out of tokens, so it did not send it again.
It must fail instead (503, retry), alone and in a batch slot, streamed or not; a reply that finished before the stop
and a request the client cancelled itself are unchanged.

    python -m pytest serve/test_close_cutoff.py -q
"""
import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from serve import test_parallel as tp

LONG_PROMPT = "read this slowly " * 200          # ~3,400 byte tokens (the fake engine has 4,096): 32 a step, ~1 s


class CloseCutsARequestOff(tp.FakeEngineCase):
    def post(self, text, stream=False, max_tokens=64):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                "reasoning_effort": "none", "stream": stream}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

    def close_while_reading(self, stream):
        result = {}
        t = threading.Thread(target=lambda: result.update(zip(("status", "body"), self.post(LONG_PROMPT, stream))))
        t.start()
        e = self.engine
        end = time.monotonic() + 10
        while time.monotonic() < end and e.progress is None and not any(e.__dict__.get("slot_live") or []):
            time.sleep(0.02)
        time.sleep(0.1)                                  # inside the prompt read
        self.assertTrue(t.is_alive(), f"the request ended before the engine was closed: {result}")
        e.close()                                        # what the stop path does after begin_drain()
        t.join(60)
        self.assertFalse(t.is_alive())
        return result["status"], result["body"]

    def assert_failed(self, status, body, stream):
        self.assertNotIn('"finish_reason": "length"', body.replace('"finish_reason":"length"', '"finish_reason": "length"'))
        if stream:
            self.assertIn("error", body)
        else:
            self.assertEqual(status, 503, body)
            self.assertIn("draining", json.loads(body)["error"]["message"])

    def test_one_at_a_time(self):
        self.start(0)
        self.assert_failed(*self.close_while_reading(stream=False), stream=False)

    def test_one_at_a_time_streamed(self):
        self.start(0)
        self.assert_failed(*self.close_while_reading(stream=True), stream=True)

    def test_in_a_batch_slot(self):
        self.start(2)
        self.assert_failed(*self.close_while_reading(stream=False), stream=False)

    def test_a_reply_that_finished_before_the_stop_is_a_reply(self):
        self.start(0)
        status, body = self.post("hello")
        self.assertEqual(status, 200, body)
        self.engine.close()
        self.assertEqual(json.loads(body)["choices"][0]["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
