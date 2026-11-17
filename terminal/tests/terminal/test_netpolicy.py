"""strata_control/netpolicy.py against local servers: one that accepts and never answers (a black hole), a port that refuses, a server
that answers and then trickles bytes, a server that answers normally. Timings are scaled down (limits under a second); the rule's own
numbers are checked as constants. The slow real-time trickle test (bytes every 20-50 s for more than 5 minutes) runs with
STRATA_SLOW_NET_TESTS=1."""
import http.server
import os
import re
import socket
import socketserver
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from strata_control import netpolicy as N


class BlackHole:
    """Accepts connections and never sends a byte (and never reads one)."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.held, self.stop = [], False
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    def _accept(self):
        while not self.stop:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            self.held.append(c)

    def close(self):
        self.stop = True
        for c in self.held:
            c.close()
        self.sock.close()


#: "fails at once" for a refused connection. Windows itself reports a refused loopback port only after 2.03-2.04 s (measured on the owner PC: it tries the
#: connection again first), longer than the scaled test windows, so there the refused-port tests use windows of 3 / 6 s, a waiting-line delay of 3 s and
#: a limit of 4 s; on Linux and macOS the refusal is immediate (0.4 / 0.8 s windows, 0.5 s). Either way far under the rule's 12 s / 44 s windows.
AT_ONCE_S = 4.0 if os.name == "nt" else 0.5
REFUSED_WINDOW = (3.0, 6.0) if os.name == "nt" else (0.4, 0.8)
REFUSED_WAIT_LINE_S = 3.0 if os.name == "nt" else 0.5


def refused_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Server:
    """A small HTTP server; `behave(handler)` writes each answer."""

    def __init__(self, behave):
        outer = self
        self.requests = 0

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.requests += 1
                behave(self, outer.requests)

            do_HEAD = do_GET

        class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.httpd = TS(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self, path="/f"):
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def ok_body(body=b"hello"):
    def behave(h, n):
        h.send_response(200)
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        if h.command != "HEAD":
            h.wfile.write(body)
    return behave


def trickle(total: int, every: float):
    def behave(h, n):
        h.send_response(200)
        h.send_header("Content-Length", str(total))
        h.end_headers()
        for _ in range(total):
            time.sleep(every)
            try:
                h.wfile.write(b"x")
                h.wfile.flush()
            except OSError:
                return
    return behave


def headers_then_silence(h, n):
    h.send_response(200)
    h.send_header("Content-Length", "100")
    h.end_headers()
    h.wfile.flush()
    time.sleep(5)


def read_then_silence(h, n):
    time.sleep(5)


class Base(unittest.TestCase):
    def setUp(self):
        self.memory = N.HostMemory()
        self.opener = N.build_opener(urllib.request.ProxyHandler({}))
        self.lines = []
        p = mock.patch.object(N, "CONNECT_LIMITS", (0.4, 0.8))      # the rule's window, scaled: (0.4, 0.8) is the full window here
        p.start()
        self.addCleanup(p.stop)

    def open(self, url, limits=(0.4, 0.8), **kw):
        kw.setdefault("memory", self.memory)
        kw.setdefault("gap", 0.1)
        kw.setdefault("say", self.lines.append)
        return N.open_url(self.opener.open, urllib.request.Request(url), limits=limits, **kw)


class TheRuleNumbers(unittest.TestCase):
    def test_numbers_as_approved(self):
        self.assertEqual((N.FIRST_CONNECT_S, N.RETRY_CONNECT_S, N.RETRY_GAP_S), (12.0, 30.0, 2.0))
        self.assertEqual((N.STALL_S, N.GIVE_UP_S, N.HOST_MEMORY_S), (60.0, 300.0, 600.0))
        self.assertEqual((N.WAIT_LINE_AFTER_S, N.WAIT_LINE_EVERY_S), (5.0, 10.0))
        self.assertLessEqual(N.FIRST_CONNECT_S + N.RETRY_GAP_S + N.RETRY_CONNECT_S, 45)


class BlackHoleFailsFast(Base):
    def test_accepts_and_never_answers_fails_after_both_limits(self):
        hole = BlackHole()
        self.addCleanup(hole.close)
        t0 = time.monotonic()
        with self.assertRaises(N.NetFailure) as cm:
            self.open(f"http://127.0.0.1:{hole.port}/f")
        took = time.monotonic() - t0
        self.assertEqual(cm.exception.kind, "no_answer")
        self.assertGreaterEqual(took, 1.2)
        self.assertLess(took, 2.5)                          # 0.4 + 0.1 + 0.8, not two stall periods
        self.assertEqual(len(hole.held), 2)                  # one retry, no more
        self.assertIn("127.0.0.1 did not answer within", cm.exception.headline("the picture reader"))
        self.assertIn("so the picture reader could not be downloaded.", cm.exception.headline("the picture reader"))

    def test_a_tls_handshake_that_never_answers_is_bounded(self):
        hole = BlackHole()
        self.addCleanup(hole.close)
        t0 = time.monotonic()
        with self.assertRaises(N.NetFailure) as cm:
            self.open(f"https://127.0.0.1:{hole.port}/f")
        self.assertEqual(cm.exception.kind, "no_answer")
        self.assertLess(time.monotonic() - t0, 2.5)

    def test_the_wait_for_the_answer_counts_in_the_connection_phase(self):
        srv = Server(read_then_silence)                      # reads the request, never answers: not 60 s of stall
        self.addCleanup(srv.close)
        t0 = time.monotonic()
        with self.assertRaises(N.NetFailure):
            self.open(srv.url())
        self.assertLess(time.monotonic() - t0, 2.5)

    def test_the_host_is_not_tried_again_in_this_run(self):
        hole = BlackHole()
        self.addCleanup(hole.close)
        with self.assertRaises(N.NetFailure):
            self.open(f"http://127.0.0.1:{hole.port}/a")
        t0 = time.monotonic()
        with self.assertRaises(N.NetFailure) as cm:
            self.open(f"http://127.0.0.1:{hole.port}/b")
        self.assertLess(time.monotonic() - t0, 0.2)
        self.assertEqual(cm.exception.kind, "again")
        self.assertEqual(len(hole.held), 2)
        self.assertIn("did not answer a moment ago either", cm.exception.headline("the model files"))
        # a resume of a download that already moved does not ask the memory
        with self.assertRaises(N.NetFailure) as cm:
            self.open(f"http://127.0.0.1:{hole.port}/b", check_memory=False)
        self.assertEqual(cm.exception.kind, "no_answer")

    def test_a_shorter_call_reads_the_memory_but_never_writes_it(self):
        hole = BlackHole()
        self.addCleanup(hole.close)
        with self.assertRaises(N.NetFailure) as cm:
            self.open(f"http://127.0.0.1:{hole.port}/models", limits=(0.2, 0.2))      # an optional 5 s read, scaled
        self.assertEqual(cm.exception.kind, "no_answer")
        self.assertFalse(self.memory.failed_recently(f"127.0.0.1:{hole.port}"), "a short read does not mark the host")
        with self.assertRaises(N.NetFailure) as cm:                                  # the required download: its own full window
            self.open(f"http://127.0.0.1:{hole.port}/engine")
        self.assertEqual(cm.exception.kind, "no_answer")
        self.assertEqual(len(hole.held), 4)
        self.assertTrue(self.memory.failed_recently(f"127.0.0.1:{hole.port}"))
        t0 = time.monotonic()
        with self.assertRaises(N.NetFailure) as cm:                                  # and a later short read skips it at once
            self.open(f"http://127.0.0.1:{hole.port}/models", limits=(0.2, 0.2))
        self.assertEqual(cm.exception.kind, "again")
        self.assertLess(time.monotonic() - t0, 0.1)

    def test_the_waiting_line_names_the_host(self):
        hole = BlackHole()
        self.addCleanup(hole.close)
        with self.assertRaises(N.NetFailure):
            self.open(f"http://127.0.0.1:{hole.port}/f", limits=(0.8, 0.8), wait_after=0.2, wait_every=0.4)
        self.assertGreaterEqual(len(self.lines), 2)
        self.assertEqual(self.lines[0], "Waiting for 127.0.0.1 to answer (0 s so far)")
        for line in self.lines:
            self.assertRegex(line, N.WAITING_RE)

    def test_the_memory_is_in_memory_only_and_expires(self):
        now = [0.0]
        m = N.HostMemory(clock=lambda: now[0])
        m.remember("github.com")
        self.assertTrue(m.failed_recently("github.com"))
        now[0] = N.HOST_MEMORY_S - 1
        self.assertTrue(m.failed_recently("github.com"))
        now[0] = N.HOST_MEMORY_S
        self.assertFalse(m.failed_recently("github.com"))
        self.assertFalse(N.HostMemory().failed_recently("github.com"))   # a new run (a new process) starts empty


class ClearFailuresAreImmediate(Base):
    def test_a_refused_port_fails_at_once_without_retry_or_line(self):
        port = refused_port()
        t0 = time.monotonic()
        with self.assertRaises(N.NetFailure) as cm:
            self.open(f"http://127.0.0.1:{port}/f", limits=REFUSED_WINDOW, wait_after=REFUSED_WAIT_LINE_S)
        self.assertLess(time.monotonic() - t0, AT_ONCE_S)
        self.assertEqual(cm.exception.kind, "offline")
        self.assertFalse(self.memory.failed_recently("127.0.0.1"))
        self.assertIn("This computer could not connect to 127.0.0.1 (it looks offline)", cm.exception.headline("the model files"))

    def test_no_dns_answer_fails_at_once(self):
        with mock.patch.object(N.socket, "getaddrinfo", side_effect=socket.gaierror(-3, "Temporary failure in name resolution")):
            t0 = time.monotonic()
            with self.assertRaises(N.NetFailure) as cm:
                self.open("http://huggingface.co/f")
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual((cm.exception.kind, cm.exception.host), ("offline", "huggingface.co"))

    def test_a_name_lookup_that_hangs_is_bounded(self):
        def slow(*a, **k):
            time.sleep(3)
            raise socket.gaierror(-3, "late")
        with mock.patch.object(N.socket, "getaddrinfo", side_effect=slow):
            t0 = time.monotonic()
            with self.assertRaises(N.NetFailure) as cm:
                self.open("http://huggingface.co/f", limits=(0.3, 0.3))
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertEqual(cm.exception.kind, "no_answer")

    def test_404_is_not_tried_again_503_is_tried_once(self):
        def behave(h, n):
            h.send_response(404 if h.path == "/gone" else (503 if n == 1 else 200))
            h.send_header("Content-Length", "0")
            h.end_headers()
        srv = Server(behave)
        self.addCleanup(srv.close)
        with self.open(srv.url("/busy")) as r:              # request 1: 503, request 2: 200
            self.assertEqual(r.status, 200)
        self.assertEqual(srv.requests, 2)
        with self.assertRaises(urllib.error.HTTPError):
            self.open(srv.url("/gone"))
        self.assertEqual(srv.requests, 3)


class WorkingServers(Base):
    def test_a_normal_answer_is_one_request_and_no_line(self):
        srv = Server(ok_body(b"hello"))
        self.addCleanup(srv.close)
        with self.open(srv.url(), wait_after=0.5) as r:
            self.assertEqual(r.read(), b"hello")
        self.assertEqual(srv.requests, 1)
        self.assertEqual(self.lines, [])

    def test_a_silent_body_stalls_out_after_the_stall_time(self):
        srv = Server(headers_then_silence)
        self.addCleanup(srv.close)
        opener = N.build_opener(urllib.request.ProxyHandler({}), stall=0.5)
        r = opener.open(srv.url(), timeout=2)
        t0 = time.monotonic()
        with self.assertRaises(OSError):
            r.read()
        self.assertLess(time.monotonic() - t0, 1.5)

    def test_a_trickle_slower_than_the_connect_limit_keeps_going(self):
        srv = Server(trickle(8, 0.3))                        # a byte every 0.3 s for 2.4 s: longer than every connect limit
        self.addCleanup(srv.close)
        opener = N.build_opener(urllib.request.ProxyHandler({}), stall=1.0)
        with N.open_url(opener.open, srv.url(), limits=(0.2, 0.2), memory=self.memory) as r:
            self.assertEqual(r.read(), b"x" * 8)

    @unittest.skipUnless(os.environ.get("STRATA_SLOW_NET_TESTS") == "1", "real-time trickle: STRATA_SLOW_NET_TESTS=1")
    def test_real_time_trickle_for_more_than_five_minutes(self):
        """A few bytes every 20-50 s for about 6 minutes, with the rule's own STALL_S: the link is slow but alive and is never cut."""
        gaps = [20, 35, 50, 25, 45, 30, 50, 40, 20, 50]             # 365 s
        def behave(h, n):
            h.send_response(200)
            h.send_header("Content-Length", str(3 * len(gaps)))
            h.end_headers()
            for g in gaps:
                time.sleep(g)
                h.wfile.write(b"abc")
                h.wfile.flush()
        srv = Server(behave)
        self.addCleanup(srv.close)
        t0 = time.monotonic()
        with N.open_url(N.urlopen, srv.url(), memory=self.memory) as r:
            got = b""
            while len(got) < 3 * len(gaps):
                b = r.read(3)
                if not b:
                    break
                got += b
        self.assertEqual(len(got), 3 * len(gaps))
        self.assertGreater(time.monotonic() - t0, N.GIVE_UP_S)


class Kinds(unittest.TestCase):
    def test_kinds(self):
        U = urllib.error.URLError
        self.assertEqual(N.kind(U(socket.gaierror(-2, "x"))), "offline")
        self.assertEqual(N.kind(U(ConnectionRefusedError())), "offline")
        self.assertEqual(N.kind(U(TimeoutError("_ssl.c:993: The handshake operation timed out"))), "no_answer")
        self.assertEqual(N.kind(U("timed out")), "no_answer")
        self.assertEqual(N.kind(TimeoutError()), "no_answer")
        self.assertEqual(N.kind(U(ConnectionResetError())), "retry")
        self.assertEqual(N.kind(urllib.error.HTTPError("u", 503, "", {}, None)), "http_retry")
        self.assertEqual(N.kind(urllib.error.HTTPError("u", 404, "", {}, None)), "http")

    def test_stall_clock(self):
        now = [0.0]
        c = N.StallClock(clock=lambda: now[0])
        c.progress(0)
        now[0] = 299
        self.assertFalse(c.expired())
        c.progress(10)                                       # any new byte resets it
        now[0] = 500
        self.assertFalse(c.expired())
        c.progress(10)                                       # the same count is no progress
        now[0] = 599
        self.assertTrue(c.expired())
        self.assertEqual(c.words("the model files", 37, 100),
                         ("No data arrived for 5 minutes, so the download of the model files stopped at 37%.",
                          "Run the same command again: it continues where it stopped."))

    def test_what_words(self):
        self.assertEqual(N.what_words(None), "the model files")
        self.assertEqual(N.what_words("vision encoder"), "the picture reader")
        self.assertEqual(N.what_words("llama.cpp source"), "the tools that prepare models")
        self.assertEqual(N.what_words("Strata engine"), "the Strata engine")


if __name__ == "__main__":
    unittest.main()
