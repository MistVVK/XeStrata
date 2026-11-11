"""Tests for setup.py's parallel ranged downloads (0.2.0): a local threaded HTTP server that supports Range stands in
for Hugging Face's CDN.  Covers the parallel path, a resume after a kill and after a failed round, a server without Range,
a piece that fails once, a signed address that runs out, the SHA-256 pin, STRATA_DL_CONNECTIONS=1 and the progress line.

    python -m unittest tools.test_setup_download
"""
from __future__ import annotations

import contextlib
import hashlib
import http.server
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_mcp  # noqa: E402

CHUNK = 64 << 10                                       # the tests' piece size (setup uses 32 MiB)
DATA = bytes((i * 7 + (i >> 8)) & 0xFF for i in range(CHUNK * 40 + 12345))       # 40 whole pieces and a short last one
BIG = (bytes((i * 11 + (i >> 9)) & 0xFF for i in range(1 << 16)) * 170)[:(10 << 20) + 12345]     # 10 MiB and a bit: longer than the one-stream read
NAME = "t.gguf"


class Server:
    """A threaded HTTP server for one file.  `faults(path, start, end, nth)` may answer a request with an error."""

    def __init__(self, data=DATA, ranges=True, length=True, redirect=False, delay=0.002, faults=None, hold_from=None):
        self.data, self.ranges, self.length, self.redirect, self.delay = data, ranges, length, redirect, delay
        self.faults = faults
        # hold_from: a body byte at this file offset or later is not sent until the "release" event is set (the headers and every byte before it are):
        # a client is at a known place, with no timing in it, until the test lets go (a kill test; release.set() also frees the handlers)
        self.hold_from, self.release = hold_from, threading.Event()
        self.lock = threading.Lock()
        self.gets, self.heads, self.redirects = [], 0, 0           # gets: (start, end) of every ranged GET
        self.active = self.max_active = 0
        self.streams = self.max_streams = 0                         # the same without setup's one-byte probes (Range: bytes=0-0)
        self.nth = {}
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_HEAD(self):
                outer.handle(self, head=True)

            def do_GET(self):
                outer.handle(self, head=False)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/{NAME}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    def handle(self, h, head):
        path = h.path.split("?")[0]
        if self.redirect and path == "/" + NAME:
            with self.lock:
                self.redirects += 1
            h.send_response(302)
            h.send_header("Location", f"http://127.0.0.1:{self.port}/cdn/{NAME}?Expires={int(time.time()) + 3600}&Signature=x")
            h.send_header("Content-Length", "0")
            h.end_headers()
            return
        size = len(self.data)
        rng = h.headers.get("Range")
        start, end = 0, size - 1
        if rng and self.ranges:
            m = re.match(r"bytes=(\d+)-(\d*)$", rng)
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        if head:
            with self.lock:
                self.heads += 1
            h.send_response(200)
            if self.length:
                h.send_header("Content-Length", str(size))
            h.send_header("Content-Type", "application/octet-stream")
            h.end_headers()
            return
        with self.lock:
            self.gets.append((start, end) if rng and self.ranges else (0, size - 1))
            key = (start, end)
            self.nth[key] = n = self.nth.get(key, 0) + 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            probe = rng == "bytes=0-0"
            if not probe:
                self.streams += 1
                self.max_streams = max(self.max_streams, self.streams)
        try:
            fault = self.faults(path, start, end, n) if self.faults else None
            if fault and fault[0] == "status":
                h.send_response(fault[1])
                h.send_header("Content-Length", "0")
                h.end_headers()
                return
            body = self.data[start:end + 1]
            if rng and self.ranges:
                h.send_response(206)
                h.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                h.send_response(200)
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            cut = fault[1] if fault and fault[0] == "cut" else None
            try:
                for i in range(0, len(body), 16384):
                    if self.hold_from is not None and start + i >= self.hold_from:
                        self.release.wait(120)
                    if cut is not None and i >= cut:
                        h.connection.shutdown(2)           # the connection drops in the middle of the body
                        return
                    h.wfile.write(body[i:i + 16384])
                    if self.delay:
                        time.sleep(self.delay)
            except OSError:
                pass
        finally:
            with self.lock:
                self.active -= 1
                if not probe:
                    self.streams -= 1


def pin_of(data):
    return len(data), hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def small_pieces(**extra):
    """Setup's piece size, size floor and waits scaled down to the test file."""
    consts = dict(DL_CHUNK=CHUNK, DL_RUN=1, DL_MIN_PARALLEL=0, DL_BACKOFF=0.0, DL_SAVE_EVERY=0.05, DL_CONNECTIONS=4)
    consts.update(extra)
    with contextlib.ExitStack() as st:
        for k, v in consts.items():
            st.enter_context(mock.patch.object(setup, k, v))
        st.enter_context(mock.patch.object(setup.time, "sleep", lambda s: None))
        env = {k: v for k, v in os.environ.items() if k not in ("STRATA_DL_CONNECTIONS", "STRATA_ALLOW_UNPINNED")}
        env["STRATA_SOURCE"] = "huggingface"
        st.enter_context(mock.patch.dict(os.environ, env, clear=True))
        yield


def run(server, d, pin="auto", name=NAME):
    """setup.download() of the server's file into folder d; returns (exit code or None, stdout)."""
    dst = Path(d) / name
    want = pin_of(DATA) if pin == "auto" else pin
    out = io.StringIO()
    code = None
    with mock.patch.object(setup, "file_pin", lambda u: want), contextlib.redirect_stdout(out):
        try:
            setup.download(server.url, dst, "the model")
        except SystemExit as e:
            code = e.code if e.code is not None else 0
    return code, out.getvalue()


def asked(srv):
    """The pieces (indexes) the server was asked for, one entry per request; the one-byte probe is left out, and so is the
    plain whole-file GET that Python before 3.13 sends when it follows the HEAD's redirect (urllib turns the redirected HEAD into
    a GET and the download closes it at once); no piece is that big."""
    whole = (0, len(srv.data) - 1)
    return [s // CHUNK for s, e in srv.gets if (s, e) not in ((0, 0), whole)]


def files(d):
    return sorted(p.name for p in Path(d).iterdir())


class ParallelDownload(unittest.TestCase):
    def test_a_big_file_comes_over_several_connections(self):
        with small_pieces(), Server(redirect=True) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(files(d), [NAME, NAME + ".done"])             # no .part, no piece list left behind
            self.assertIn("sha256 " + hashlib.sha256(DATA).hexdigest(), (Path(d) / (NAME + ".done")).read_text())
            self.assertGreater(srv.max_active, 1)
            self.assertLessEqual(srv.max_active, 4)
            self.assertEqual(sorted(asked(srv)), list(range(41)))           # every piece once, none twice
            self.assertEqual(srv.redirects, 2)                              # HEAD and the one probe: the pieces use the CDN address

    def test_the_redirect_is_resolved_once(self):
        with small_pieces(), Server(redirect=True) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertLessEqual(srv.redirects, 2)

    def test_a_small_file_is_one_stream(self):
        with small_pieces(DL_MIN_PARALLEL=len(DATA) + 1), Server() as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(srv.max_streams, 1)   # (the one-byte probe can still be sending: not a stream)
            self.assertEqual(len(srv.gets), 1)

    def test_one_connection_variable_restores_one_stream(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"STRATA_DL_CONNECTIONS": "1"}):
                code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(srv.max_streams, 1)   # (the one-byte probe can still be sending: not a stream)
            self.assertEqual(srv.gets, [(0, len(DATA) - 1)])                # one request for the whole file

    def test_connection_count_variable(self):
        with small_pieces():
            self.assertEqual(setup.dl_connections(), 4)
            for raw, want in (("16", 16), ("0", 1), ("-3", 1), ("99", 32), ("x", 4), ("", 4)):
                with mock.patch.dict(os.environ, {"STRATA_DL_CONNECTIONS": raw}):
                    self.assertEqual(setup.dl_connections(), want, raw)


class NoRangeSupport(unittest.TestCase):
    def test_a_server_that_ignores_range_is_read_in_one_stream(self):
        with small_pieces(), Server(ranges=False) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(srv.max_streams, 1)   # (the one-byte probe can still be sending: not a stream)
            self.assertEqual(files(d), [NAME, NAME + ".done"])

    def test_a_server_without_any_length_is_read_in_one_stream(self):
        with small_pieces(), Server(length=False, ranges=False) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d, pin=None)                                # no length, so nothing to check a pin against first
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(srv.max_streams, 1)   # (the one-byte probe can still be sending: not a stream)

    def test_a_head_without_a_length_still_gets_the_connections(self):
        """ModelScope's HEAD has no Content-Length (#1915): the length comes from a one-byte ranged GET, not 'unknown'."""
        with small_pieces(), Server(length=False) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertGreater(srv.max_active, 1)
            self.assertEqual(sorted(asked(srv)), list(range(41)))
            self.assertEqual(files(d), [NAME, NAME + ".done"])

    def test_range_stopping_in_the_middle_falls_back_to_the_unbroken_start(self):
        def faults(path, start, end, nth):
            return None

        with small_pieces(), Server(faults=faults) as srv, tempfile.TemporaryDirectory() as d:
            real = setup._fetch_run
            calls = []

            def fetch(src, st, f, run):
                calls.append(run)
                if len(calls) == 6:
                    raise setup._NoRange("HTTP 200")
                return real(src, st, f, run)

            with mock.patch.object(setup, "_fetch_run", fetch):
                code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)           # the pin check passed: no gaps were kept
            self.assertEqual(files(d), [NAME, NAME + ".done"])


class Resume(unittest.TestCase):
    def test_a_failed_round_keeps_the_finished_pieces(self):
        broken = {"on": True}

        def faults(path, start, end, nth):
            if broken["on"] and start >= 20 * CHUNK:
                return ("status", 500)
            return None

        with small_pieces(DL_ROUNDS=1, DL_CHUNK_TRIES=1), Server(faults=faults) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertEqual(code, 1)                                       # could not finish: setup stops with a message
            self.assertIn("could not finish downloading", out)
            part = Path(d) / (NAME + ".part")
            side = Path(d) / (NAME + ".part.chunks")
            self.assertTrue(part.exists() and side.exists())
            saved = json.loads(side.read_text())
            done = {i for a, b in saved["done"] for i in range(a, b)}
            self.assertTrue(done and all(i < 20 for i in done), done)
            srv.gets.clear()
            broken["on"] = False
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            again = set(asked(srv))
            self.assertFalse(again & done, "pieces that were finished are fetched again")
            self.assertEqual(again | done, set(range(41)))
            self.assertEqual(files(d), [NAME, NAME + ".done"])

    def killed_download_continues(self, one_stream):
        """Kill setup.download in a subprocess once 12 pieces are on disk, then run it again: it continues and fetches none of them again.
        No timing decides this: the test server sends nothing from file offset HOLD on until the test has killed the runner (Server.hold_from), so the
        runner is always found part-way, however slow or fast the machine is. one_stream: the range probe fails in the runner, so setup reads the
        file in one stream (parallel_download returns False), whose .part is an unbroken run from the start."""
        # The parallel path saves every finished piece, so 13 whole pieces may arrive and 12 are on disk before the kill. The one-stream path writes
        # what it reads in blocks of 8 MiB (setup.download_single), so its file must be longer than that: 8 MiB land on disk, then the server holds.
        data = BIG if one_stream else DATA
        HOLD = 9 << 20 if one_stream else 13 * CHUNK
        pin = (len(data), hashlib.sha256(data).hexdigest())
        probe = "setup._probe_ranges = lambda *a, **k: False\n" if one_stream else ""
        runner = (
            "import sys; sys.path[:0] = [%r, %r]\n"
            "import setup\n"
            "setup.DL_CHUNK = %d; setup.DL_MIN_PARALLEL = 0; setup.DL_SAVE_EVERY = 0.05; setup.DL_CONNECTIONS = 4\n"
            "%s"
            "pin = (%d, %r)\n"
            "setup.file_pin = lambda u: pin\n"
            "setup.download(sys.argv[1], __import__('pathlib').Path(sys.argv[2]) / %r, 'the model')\n"
            % (str(ROOT), str(ROOT / "tools"), CHUNK, probe, pin[0], pin[1], NAME))
        with small_pieces(), Server(data=data, hold_from=HOLD) as srv, tempfile.TemporaryDirectory() as d:
            self.addCleanup(srv.release.set)                    # never leave a handler thread waiting
            env = dict(os.environ, STRATA_SOURCE="huggingface")
            err = Path(d) / "runner.err"
            with open(err, "wb") as err_file:
                proc = subprocess.Popen([sys.executable, "-c", runner, srv.url, d], env=env,
                                        stdout=subprocess.DEVNULL, stderr=err_file)
            side = Path(d) / (NAME + ".part.chunks")
            part = Path(d) / (NAME + ".part")
            started = time.time()
            deadline = started + 60
            try:
                while time.time() < deadline and proc.poll() is None:
                    try:
                        if side.is_file():
                            if sum(b - a for a, b in json.loads(side.read_text())["done"]) >= 12:
                                break
                        elif part.is_file() and part.stat().st_size >= 12 * CHUNK:   # the one-stream fallback (below)
                            break
                    except (OSError, ValueError, KeyError):
                        pass
                    threading.Event().wait(0.02)                # (not time.sleep: small_pieces() makes that a no-op)
            finally:
                code = proc.poll()
                proc.kill()
                proc.wait()
                srv.release.set()
            state = "the runner %s after %.1f s; its stderr: %s" % (
                "was still running" if code is None else "had already ended with exit code %s" % code, time.time() - started,
                err.read_text(errors="replace")[-2000:] or "(empty)")
            self.assertTrue(part.exists(), "the .part file is missing at the kill: " + state)
            if side.exists():
                saved = json.loads(side.read_text())
                done = {i for a, b in saved["done"] for i in range(a, b)}
            else:
                # One failed range probe makes setup read the file in one stream (parallel_download returns False): its .part is an
                # unbroken run from the start and has no piece list, which the next run reads as finished pieces (_load_pieces).
                # Seen on GitHub's Linux and macOS runners (run 42699343686); both ways must continue without fetching them again.
                done = set(range(part.stat().st_size // CHUNK))
            self.assertGreaterEqual(len(done), 12, "%d pieces at the kill; %s" % (len(done), state))
            srv.hold_from = None                                # the second run is not held
            srv.gets.clear()
            code, out = run(srv, d, pin=pin)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), data)
            self.assertFalse(set(asked(srv)) & done, "pieces that were finished before the kill are fetched again")

    def test_a_killed_download_continues(self):
        self.killed_download_continues(one_stream=False)

    def test_a_killed_one_stream_download_continues(self):
        self.killed_download_continues(one_stream=True)

    def test_a_part_from_one_stream_continues_after_its_last_whole_piece(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            have = CHUNK * 10 + 777                                         # an older setup's .part: bytes from the start
            (Path(d) / (NAME + ".part")).write_bytes(DATA[:have])
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(sorted(asked(srv)), list(range(10, 41)))       # pieces 0-9 were not asked for again

    def test_one_stream_continues_a_parallel_part_from_its_unbroken_start(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            part = Path(d) / (NAME + ".part")
            junk = bytearray(DATA)
            junk[8 * CHUNK:9 * CHUNK] = b"\0" * CHUNK                       # a gap behind the later pieces
            part.write_bytes(bytes(junk[:20 * CHUNK]))
            (Path(d) / (NAME + ".part.chunks")).write_text(json.dumps(
                {"v": 1, "total": len(DATA), "chunk": CHUNK, "done": [[0, 8], [9, 20]]}))
            with mock.patch.dict(os.environ, {"STRATA_DL_CONNECTIONS": "1"}):
                code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(srv.gets, [(8 * CHUNK, len(DATA) - 1)])        # resumed after piece 7, the gap refetched
            self.assertEqual(files(d), [NAME, NAME + ".done"])

    def test_a_piece_list_of_another_file_starts_over(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            (Path(d) / (NAME + ".part")).write_bytes(b"x" * (CHUNK * 3))
            (Path(d) / (NAME + ".part.chunks")).write_text(json.dumps(
                {"v": 1, "total": 999, "chunk": CHUNK, "done": [[0, 3]]}))
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)


class Failures(unittest.TestCase):
    def test_a_piece_that_fails_once_is_tried_again(self):
        def faults(path, start, end, nth):
            if start == 5 * CHUNK and nth == 1:
                return ("status", 500)
            if start == 9 * CHUNK and nth == 1:
                return ("cut", CHUNK // 2)                                  # drops half way: continues from where it was
            return None

        with small_pieces(), Server(faults=faults) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(sum(1 for s, _ in srv.gets if s == 5 * CHUNK), 2)
            self.assertEqual(sum(1 for s, _ in srv.gets if s == 9 * CHUNK), 1)
            resumed = [s for s, _ in srv.gets if 9 * CHUNK < s < 10 * CHUNK]
            self.assertEqual(len(resumed), 1)                               # the rest of the piece, not all of it again
            self.assertGreater(resumed[0], 9 * CHUNK)

    def test_a_signed_address_that_ran_out_is_asked_for_again(self):
        def faults(path, start, end, nth):
            if path.startswith("/cdn/") and nth == 1 and start in (4 * CHUNK, 11 * CHUNK):
                return ("status", 403)
            return None

        with small_pieces(), Server(redirect=True, faults=faults) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertGreater(srv.redirects, 2)                            # HEAD + first probe + a new address after the 403

    def test_a_pinned_hash_mismatch_keeps_the_file_as_bad_and_stops(self):
        wrong = bytes(reversed(DATA))
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d, pin=pin_of(wrong))
            self.assertEqual(code, 1)
            self.assertEqual(files(d), [NAME + ".bad"])                     # no file, no .done, no .part, no piece list; the bytes are kept
            self.assertEqual((Path(d) / (NAME + ".bad")).read_bytes(), DATA)

    def test_a_pinned_hash_mismatch_with_the_same_size(self):
        other = bytes((b ^ 0x55) for b in DATA)
        with small_pieces(), Server(data=other) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertEqual(code, 1)
            self.assertIn("does not match the pinned SHA-256", out)
            self.assertIn(NAME + ".bad", out)
            self.assertIn("the server sends exactly the bytes you have", out)      # the repair compared every block: the server's file is this one
            self.assertEqual(files(d), [NAME + ".bad"])

    def test_a_kept_bad_file_is_taken_back_when_it_matches_the_pin_now(self):
        """A false mismatch (a bad read, a pin that was wrong and is fixed) costs one hash, not the download: the .bad file is the file again."""
        wrong = bytes(reversed(DATA))
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            code, _ = run(srv, d, pin=pin_of(wrong))
            self.assertEqual(code, 1)
            srv.gets.clear()
            code, out = run(srv, d)                                        # the right pin now
            self.assertIsNone(code, out)
            self.assertEqual(srv.gets, [], "nothing was asked of the server")
            self.assertEqual(files(d), [NAME, NAME + ".done"])
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)

    def test_a_kept_bad_file_is_removed_by_a_good_download(self):
        other = bytes((b ^ 0x55) for b in DATA)
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            (Path(d) / (NAME + ".bad")).write_bytes(other)                  # kept by an earlier run; not the pinned bytes
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual(files(d), [NAME, NAME + ".done"])
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)

    def test_a_kept_bad_file_is_removed_when_there_is_no_room_for_the_new_copy(self):
        other = bytes((b ^ 0x55) for b in DATA)
        usage = type("U", (), {"free": 1000, "total": 10 ** 6, "used": 0})()
        seen = []

        def faults(path, start, end, nth):
            seen.append(files(self.d))
            return None

        with small_pieces(), Server(faults=faults) as srv, tempfile.TemporaryDirectory() as d,                 mock.patch.object(setup.shutil, "disk_usage", lambda p: usage):
            self.d = d
            (Path(d) / (NAME + ".bad")).write_bytes(other)
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertTrue(seen and NAME + ".bad" not in seen[0], seen[:1])   # gone before the first piece came down
            self.assertEqual(files(d), [NAME, NAME + ".done"])

    def test_a_kept_bad_file_stays_while_there_is_room(self):
        other = bytes((b ^ 0x55) for b in DATA)
        seen = []

        def faults(path, start, end, nth):
            seen.append(files(self.d))
            return None

        with small_pieces(), Server(faults=faults) as srv, tempfile.TemporaryDirectory() as d:
            self.d = d
            (Path(d) / (NAME + ".bad")).write_bytes(other)
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertTrue(seen and NAME + ".bad" in seen[0], "kept until the new copy is checked")
            self.assertNotIn(NAME + ".bad", files(d))

    def test_a_full_disk_stops_with_a_message(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            real = setup._piece_worker

            def worker(src, st, todo, part):
                st.fatal = setup._DiskError("No space left on device")
                st.stop.set()

            with mock.patch.object(setup, "_piece_worker", worker):
                code, out = run(srv, d)
            self.assertEqual(code, 1)
            self.assertIn("cannot write", out)

    def test_the_file_is_not_preallocated(self):
        """The app counts a .part's size as the bytes done, so it must not jump to the full size at the start."""
        seen = []

        def faults(path, start, end, nth):
            if start == 30 * CHUNK:
                p = Path(self.d) / (NAME + ".part")
                seen.append(p.stat().st_size if p.exists() else None)
            return None

        with small_pieces(DL_CONNECTIONS=2), Server(faults=faults, delay=0.01) as srv, tempfile.TemporaryDirectory() as d:
            self.d = d
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            # a preallocated file would already be the full size (len(DATA)) here; what the other connection has written ahead of piece 30 is
            # thread timing (Windows put it at 34 and 39 pieces), so the check is only that the file has not been made full size
            self.assertTrue(seen and seen[0] is not None and seen[0] < len(DATA), seen)


class Progress(unittest.TestCase):
    def test_the_progress_line_is_the_one_the_app_reads(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            lines = [x for x in out.replace("\r", "\n").split("\n") if x.strip()]
            bar = [x for x in lines if "GB (" in x]
            self.assertTrue(bar)
            for x in bar:
                self.assertRegex(x, r"^  the model: +[\d.]+ / [\d.]+ GB \(\d+%\) +$")
            parsed = strata_mcp.install_progress(bar)["download"]
            self.assertEqual(parsed["file"], "the model")
            self.assertEqual(parsed["percent"], 100)
            self.assertEqual(parsed["total_gb"], round(len(DATA) / 1e9, 2))

    def test_the_format_matches_the_one_stream_path(self):
        with small_pieces(), Server() as srv, tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            _, par = run(srv, d1)
            with mock.patch.dict(os.environ, {"STRATA_DL_CONNECTIONS": "1"}):
                _, one = run(srv, d2)
            def shape(out):
                return {re.sub(r"[\d.]+", "N", x) for x in out.replace("\r", "\n").split("\n") if "GB (" in x}
            self.assertEqual(shape(par), shape(one))


class SignedAddress(unittest.TestCase):
    def test_expiry_of_a_cdn_address(self):
        now = time.time()
        self.assertAlmostEqual(setup._url_ttl(f"https://cdn.example/f?Expires={int(now) + 600}&Signature=x"), 600, delta=3)
        self.assertIsNone(setup._url_ttl("https://cdn.example/f"))
        self.assertIsNone(setup._url_ttl(f"https://cdn.example/f?Expires={int(now) - 5}"))      # already out, or a wrong clock
        self.assertIsNone(setup._url_ttl("https://cdn.example/f?Expires=soon"))
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
        self.assertAlmostEqual(setup._url_ttl(f"https://cdn.example/f?X-Amz-Date={stamp}&X-Amz-Expires=900"), 900, delta=3)


if __name__ == "__main__":
    unittest.main()


class Runs(unittest.TestCase):
    """One request asks for several neighbouring pieces (DL_RUN): fewer connections to open, the same piece list."""

    def test_a_request_covers_several_pieces(self):
        with small_pieces(DL_RUN=4), Server() as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(files(d), [NAME, NAME + ".done"])
            spans = [(s, e) for s, e in srv.gets if (s, e) != (0, 0)]
            self.assertLess(len(spans), 41)                                 # fewer requests than pieces
            self.assertTrue(any(e - s + 1 > CHUNK for s, e in spans), spans)
            self.assertTrue(all(e - s + 1 <= 4 * CHUNK for s, e in spans), spans)
            covered = sorted(i for s, e in spans for i in range(s // CHUNK, e // CHUNK + 1))
            self.assertEqual(covered, list(range(41)))                       # every piece once, none twice

    def test_the_last_pieces_are_asked_for_one_by_one(self):
        st = setup._Pieces(CHUNK * 100, set(), conns=4)
        todo = list(range(100))
        self.assertEqual(setup._take_run(st, todo), [0, 1, 2, 3])
        with mock.patch.object(setup, "DL_RUN", 4):
            self.assertEqual(setup._take_run(st, list(range(60, 100))), [60, 61, 62, 63])
            self.assertEqual(setup._take_run(st, list(range(93, 100))), [93])      # fewer than two rounds left
            self.assertEqual(setup._take_run(st, [5, 9, 10]), [5])                  # only neighbours go together

    def test_a_broken_run_continues_from_the_byte_it_reached(self):
        cuts = []

        def faults(path, start, end, nth):
            if end - start + 1 > 2 * CHUNK and not cuts:
                cuts.append(start)
                return ("cut", int(2.5 * CHUNK))                           # drops inside the third piece of the run
            return None

        with small_pieces(DL_RUN=4, DL_CONNECTIONS=2), Server(faults=faults) as srv, tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(len(cuts), 1)
            first = cuts[0]
            again = [s for s, e in srv.gets if first < s < first + 4 * CHUNK]
            self.assertEqual(len(again), 1)                                  # one more request: for the rest
            self.assertGreaterEqual(again[0], first + 2 * CHUNK)              # the two whole pieces are not fetched again

    def test_a_connection_that_keeps_moving_is_not_given_up_on(self):
        """Seven breaks in a row (more than DL_CHUNK_TRIES) that each carried bytes forward do not end the request."""
        count = [0]

        def faults(path, start, end, nth):
            if start < 4 * CHUNK and end - start + 1 > CHUNK // 2 and count[0] < 7:
                count[0] += 1
                return ("cut", CHUNK // 3)
            return None

        with small_pieces(DL_RUN=4, DL_CONNECTIONS=2, DL_ROUNDS=1), Server(faults=faults) as srv,                 tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertIsNone(code, out)
            self.assertEqual((Path(d) / NAME).read_bytes(), DATA)
            self.assertEqual(count[0], 7)

    def test_a_request_without_progress_still_gives_up(self):
        def faults(path, start, end, nth):
            if start >= 4 * CHUNK:
                return ("status", 500)
            return None

        with small_pieces(DL_RUN=4, DL_ROUNDS=1, DL_CHUNK_TRIES=2), Server(faults=faults) as srv,                 tempfile.TemporaryDirectory() as d:
            code, out = run(srv, d)
            self.assertEqual(code, 1)
            self.assertIn("could not finish downloading", out)

    def test_the_default_is_many_connections_and_runs(self):
        self.assertGreaterEqual(setup.DL_CONNECTIONS, 8)
        self.assertGreaterEqual(setup.DL_RUN, 2)
        self.assertEqual(setup.DL_CHUNK, 32 << 20)                           # the piece list on disk names this size: resumes need it


class WholeMarkNeedsThePin(unittest.TestCase):
    """#4088: a full-length shard with no finish mark and a pinned SHA-256 must not be marked "whole" (the approach of PR #4318, written here).

    A run killed between the rename of a finished .part and the end of its hash leaves exactly that state; the mark the next run wrote made done()
    true, download() skipped the file, the pinned hash was never checked and a shard damaged by that crash stayed in place."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.repo = next(r for r, files in setup.REPO_PINS.items() if files)
        self.name = next(iter(setup.REPO_PINS[self.repo]))
        self.hf = f"https://huggingface.co/{self.repo}/resolve/main/"
        patch = mock.patch.object(setup, "whole_shard", return_value=True)       # the test is about the mark, not the GGUF header
        patch.start()
        self.addCleanup(patch.stop)

    def test_a_pinned_shard_is_left_for_the_download_check(self):
        shard = self.folder / self.name
        shard.write_bytes(b"x")
        self.assertIsNotNone(setup.file_pin(self.hf + self.name), "the file needs a pin for this test to mean anything")
        setup.mark_whole_unless_pinned([shard], self.hf, own=False)
        self.assertFalse(setup.done(shard), "no finish mark: download() hashes it and repairs it")
        self.assertFalse(shard.with_name(shard.name + ".done").exists())

    def test_a_file_without_a_pin_keeps_the_whole_mark(self):
        shard = self.folder / "hand-copied.gguf"
        shard.write_bytes(b"x")
        self.assertIsNone(setup.file_pin(self.hf + shard.name))
        setup.mark_whole_unless_pinned([shard], self.hf, own=False)
        self.assertTrue(setup.done(shard))
        self.assertIn("whole", shard.with_name(shard.name + ".done").read_text(encoding="utf-8"))

    def test_the_persons_own_file_keeps_the_mark(self):
        shard = self.folder / self.name
        shard.write_bytes(b"x")
        setup.mark_whole_unless_pinned([shard], self.hf, own=True)           # --gguf-dir: not a download
        self.assertTrue(setup.done(shard))

    def test_a_family_with_no_hf_folder_keeps_the_mark_and_does_not_raise(self):
        """The NVFP4 families have no "hf" key (they download a file list): the helper must not need one (it raised KeyError in main())."""
        shard = self.folder / self.name
        shard.write_bytes(b"x")
        setup.mark_whole_unless_pinned([shard], None, own=False)
        self.assertTrue(setup.done(shard))
        setup.mark_whole_unless_pinned([], None, own=False)

    def test_a_marked_or_missing_file_is_not_touched(self):
        shard = self.folder / "marked.gguf"
        shard.write_bytes(b"x")
        setup.mark(shard, "sha256 abc")
        setup.mark_whole_unless_pinned([shard, self.folder / "absent.gguf"], self.hf, own=False)
        self.assertEqual(shard.with_name(shard.name + ".done").read_text(encoding="utf-8").strip(), "sha256 abc")

    def test_the_damaged_pinned_file_is_then_hashed_by_the_download(self):
        """The whole path: full length, no mark, wrong bytes -> not marked -> download() takes its 'stopped before the check' branch and hashes it."""
        shard = self.folder / self.name
        shard.write_bytes(b"x" * 8)
        pin = (8, "ab" * 32)                                                  # a pin of this file's size, so the "full length" branch is the one taken
        calls = []
        with mock.patch.dict(setup.REPO_PINS[self.repo], {self.name: pin}), mock.patch.object(setup, "range_total", return_value=8),                 mock.patch.object(setup, "_urlopen", lambda *a, **k: type("R", (), {"headers": {"Content-Length": "8"}, "__enter__": lambda s: s, "__exit__": lambda s, *e: False, "close": lambda s: None})()),                 mock.patch.object(setup, "verify_sha256", lambda *a, **k: calls.append(a)):
            setup.mark_whole_unless_pinned([shard], self.hf, own=False)
            self.assertFalse(setup.done(shard))
            setup.download(self.hf + self.name, shard)
        self.assertTrue(calls and calls[0][1:3] == pin, "the pinned hash was checked")
