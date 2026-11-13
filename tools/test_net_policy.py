# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/download.py's one network rule and its disk words: a silent host costs the window and nothing more, the line
says what is waited for, a host that failed its window is not asked again in the run, an answered error passes through,
and a refused write is said in plain words.  Everything here is a local socket, no network."""
import errno
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from serve import download


class BlackHole:
    """A host that accepts the connection and never answers: the shape of a firewall that drops the answer."""

    def __init__(self, backlog: int = 16):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(backlog)
        self.sock.settimeout(0.1)
        self.port = self.sock.getsockname()[1]
        self.conns: list[socket.socket] = []
        self.stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                continue
            self.conns.append(conn)          # held open, never written to

    def url(self, name: str = "file.bin") -> str:
        return f"http://127.0.0.1:{self.port}/{name}"

    def close(self):
        self.stop = True
        for conn in self.conns:
            conn.close()
        self.sock.close()


class Quiet(BaseHTTPRequestHandler):
    """A host that answers: 200 with `BODY`, or 404 for a name ending in -missing."""
    BODY = b"x" * 64

    def do_GET(self):
        if self.path.endswith("-missing"):
            self.send_error(404, "no such file")
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.BODY)))
        self.end_headers()
        self.wfile.write(self.BODY)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.BODY)))
        self.end_headers()

    def log_message(self, *args):
        pass


def refused_port() -> int:
    """A port nothing listens on: bind, read the number, close."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Rule(unittest.TestCase):
    def setUp(self):
        self.saved = (download.FIRST_CONNECT_S, download.RETRY_CONNECT_S, download.RETRY_PAUSE_S,
                      download.WAIT_SAY_S, download.WAIT_EVERY_S, download.DEAD_HOST_S)
        # a test's window is a fraction of a second, in the same shape as the real one
        download.FIRST_CONNECT_S, download.RETRY_CONNECT_S, download.RETRY_PAUSE_S = 0.2, 0.3, 0.05
        download.WAIT_SAY_S, download.WAIT_EVERY_S = 0.1, 0.1
        download.forget_dead_hosts()

    def tearDown(self):
        (download.FIRST_CONNECT_S, download.RETRY_CONNECT_S, download.RETRY_PAUSE_S,
         download.WAIT_SAY_S, download.WAIT_EVERY_S, download.DEAD_HOST_S) = self.saved
        download.forget_dead_hosts()

    def test_silent_host_costs_the_window_and_says_what_it_waits_for(self):
        hole = BlackHole()
        said = []
        try:
            started = time.monotonic()
            with self.assertRaises(download.NetFailure) as caught:
                download.open_url(urllib.request.Request(hole.url()), say=said.append)
            waited = time.monotonic() - started
        finally:
            hole.close()
        self.assertGreaterEqual(waited, 0.4)         # the first try, the pause, the retry
        self.assertLess(waited, 2.0)                 # and not a moment more
        self.assertIn(f"127.0.0.1:{hole.port}", caught.exception.headline())
        self.assertIn("did not answer", caught.exception.headline())
        self.assertTrue(any("Waiting for" in line and "s so far" in line for line in said))

    def test_a_host_that_failed_is_not_asked_again_in_the_run(self):
        hole = BlackHole()
        try:
            with self.assertRaises(download.NetFailure):
                download.open_url(urllib.request.Request(hole.url()), say=lambda line: None)
            said = []
            started = time.monotonic()
            with self.assertRaises(download.NetFailure):
                download.open_url(urllib.request.Request(hole.url()), say=said.append)
            self.assertLess(time.monotonic() - started, 0.2)
            self.assertEqual(said, [])               # it is not even waited for
        finally:
            hole.close()

    def test_a_refused_port_fails_at_once_after_the_one_retry(self):
        url = f"http://127.0.0.1:{refused_port()}/file.bin"
        started = time.monotonic()
        with self.assertRaises(download.NetFailure):
            download.open_url(urllib.request.Request(url))
        self.assertLess(time.monotonic() - started, 3.0)

    def test_a_host_that_answers_is_read(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/file.bin"
            with download.open_url(urllib.request.Request(url)) as r:
                self.assertEqual(r.read(), Quiet.BODY)
        finally:
            server.shutdown()
            server.server_close()

    def test_an_answered_error_passes_through(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/gone-missing"
            started = time.monotonic()
            with self.assertRaises(urllib.error.HTTPError) as caught:
                download.open_url(urllib.request.Request(url))
            self.assertEqual(caught.exception.code, 404)
            caught.exception.close()
            self.assertLess(time.monotonic() - started, 2.0)   # not retried: the host answered
        finally:
            server.shutdown()
            server.server_close()

    def test_disk_words(self):
        self.assertEqual(download.disk_words(OSError(errno.ENOSPC, "No space left on device"))[0], "the drive is full")
        self.assertIn("quota", download.disk_words(OSError(errno.EDQUOT, "Disk quota exceeded"))[0])
        self.assertIn("read-only", download.disk_words(OSError(errno.EROFS, "Read-only file system"))[0])
        self.assertIn("may not write", download.disk_words(OSError(errno.EACCES, "Permission denied"))[0])
        self.assertIsNone(download.disk_words(ConnectionResetError(104, "Connection reset by peer")))
        self.assertIsNone(download.disk_words(OSError("no errno")))


if __name__ == "__main__":
    unittest.main()
