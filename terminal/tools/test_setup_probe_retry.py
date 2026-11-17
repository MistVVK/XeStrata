"""The first range probe of a parallel download is tried again after a hiccup, and not after "no internet" (lane APP-LIN 0.2.14, from lane 8):
one failed probe used to make a 50-100 GB download a single stream."""
from __future__ import annotations

import errno
import http.client
import socket
import sys
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class FakeSource:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def resolve(self):
        self.calls += 1
        out = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(out, BaseException):
            raise out
        return "https://cdn/x"


def probe(src):
    waits = []
    return setup._probe_ranges(src, waits=(1.0, 3.0), sleep=waits.append), waits


class ProbeRetry(unittest.TestCase):
    def test_a_hiccup_is_tried_again_and_the_parallel_download_goes_on(self):
        for hiccup in (ConnectionResetError(), TimeoutError("timed out"), socket.timeout("timed out"), http.client.RemoteDisconnected("closed"),
                       urllib.error.URLError(ConnectionResetError()), urllib.error.HTTPError("u", 503, "busy", {}, None)):
            src = FakeSource(hiccup)
            ok, waits = probe(src)
            self.assertTrue(ok, repr(hiccup))
            self.assertEqual((src.calls, waits), (2, [1.0]))

    def test_three_tries_at_most_then_one_stream(self):
        src = FakeSource(ConnectionResetError(), ConnectionResetError(), ConnectionResetError(), None)
        ok, waits = probe(src)
        self.assertFalse(ok)
        self.assertEqual((src.calls, waits), (3, [1.0, 3.0]))

    def test_no_internet_fails_at_once_without_waiting(self):
        for offline in (urllib.error.URLError(socket.gaierror(-3, "Temporary failure in name resolution")), urllib.error.URLError(ConnectionRefusedError()),
                        urllib.error.URLError(OSError(errno.ENETUNREACH, "Network is unreachable")), OSError(errno.EHOSTUNREACH, "No route to host"),
                        urllib.error.URLError("unknown url type"), urllib.error.HTTPError("u", 404, "missing", {}, None)):
            src = FakeSource(offline)
            ok, waits = probe(src)
            self.assertFalse(ok, repr(offline))
            self.assertEqual((src.calls, waits), (1, []), repr(offline))

    def test_a_server_without_ranges_is_not_asked_again(self):
        src = FakeSource(setup._NoRange("HTTP 200"))
        ok, waits = probe(src)
        self.assertEqual((ok, src.calls, waits), (False, 1, []))

    def test_parallel_download_uses_the_probe(self):
        text = (ROOT / "setup.py").read_text(encoding="utf-8")
        at = text.index("def parallel_download(")
        self.assertIn("if not _probe_ranges(src):", text[at:at + 800])


if __name__ == "__main__":
    unittest.main()
