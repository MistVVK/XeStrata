"""#5938: `--source modelscope` stopped at "ModelScope does not answer; downloading it from https://huggingface.co" after ONE failed HEAD, on a
network where huggingface.co cannot be reached (mainland China). ModelScope answered the same request a moment later. setup.ms_answers asks up to three
times: a HEAD, a one-byte ranged GET, a HEAD again after a pause. A real local server stands in for ModelScope's CDN; each ask is bounded.
"""
from __future__ import annotations

import http.server
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402


class Cdn(http.server.BaseHTTPRequestHandler):
    plan: list = []          # one verdict per request: an HTTP status, or "drop"
    seen: list = []

    def log_message(self, *a):
        pass

    def answer(self):
        type(self).seen.append((self.command, self.headers.get("Range")))
        verdict = type(self).plan.pop(0) if type(self).plan else 200
        if verdict == "drop":
            self.connection.close()
            return
        self.send_response(verdict)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_HEAD = do_GET = answer


class ModelScopeProbe(unittest.TestCase):
    def serve(self, plan):
        Cdn.plan, Cdn.seen = list(plan), []
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Cdn)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}/file.gguf"

    def ask(self, plan):
        url = self.serve(plan)
        sleeps = []
        with mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": ""}), mock.patch.object(setup.acquisition, "require"):
            answer = setup.ms_answers(url, timeout=3.0, sleep=sleeps.append)
        return answer, list(Cdn.seen), sleeps

    def test_one_failed_head_does_not_drop_modelscope(self):
        answer, seen, _ = self.ask([503])                  # the HEAD is refused once, then everything answers
        self.assertTrue(answer)
        self.assertEqual([m for m, _ in seen], ["HEAD", "GET"])
        self.assertEqual(seen[1][1], "bytes=0-0")           # the second ask moves one byte, not the file

    def test_a_dropped_head_and_a_refused_range_still_get_a_third_ask(self):
        answer, seen, sleeps = self.ask(["drop", 503])
        self.assertTrue(answer)
        self.assertEqual([m for m, _ in seen], ["HEAD", "GET", "HEAD"])
        self.assertEqual(sleeps, [3])

    def test_a_host_that_never_answers_is_three_asks_not_a_hang(self):
        answer, seen, _ = self.ask([503, 503, 503])
        self.assertFalse(answer)
        self.assertEqual(len(seen), 3)

    def test_a_host_that_answers_is_asked_once(self):
        answer, seen, sleeps = self.ask([])
        self.assertTrue(answer)
        self.assertEqual(len(seen), 1)
        self.assertEqual(sleeps, [])

    def test_download_uses_it(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("if ms_answers(ms_url(*ms), timeout=netpolicy.FIRST_CONNECT_S):", src)
        self.assertNotIn("if reachable(ms_url(*ms)", src)


if __name__ == "__main__":
    unittest.main()
