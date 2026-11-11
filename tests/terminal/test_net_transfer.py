"""Lane NET item 2: the update path's downloads (strata_control/transfer.py: the engine, the shape engine, Control, the app, DFlash2, the picture
reader) and the feed read under the network rule, against local servers: a black hole fails fast with the plain sentence; a connection that breaks
after bytes came is resumed; a link that trickles is never cut; 5 minutes (scaled) without a new byte ends it with W6. The real-time proof (bytes every
20-50 s for 365 s, the rule's own 60 s stall) runs with STRATA_SLOW_NET_TESTS=1."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from strata_control import feed, netpolicy, transfer
from tests.terminal.test_netpolicy import AT_ONCE_S, REFUSED_WINDOW, BlackHole, Server, refused_port

BODY = bytes(range(256)) * 40          # 10,240 bytes
SHA = hashlib.sha256(BODY).hexdigest()


def ranged(cut_first=None, every=0.0, chunk=None, silent_after=None):
    """An HTTP file server with Range: `cut_first` bytes then the connection is closed on the first request; `every` seconds between
    `chunk`-byte writes; `silent_after`: after that many bytes the server stops sending (keeps the connection)."""
    state = {"n": 0, "ranges": []}                    # "ranges": the Range header of every request, in order

    def behave(h, n):
        state["n"] += 1
        start = 0
        rng = h.headers.get("Range")
        state["ranges"].append(rng)
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
        body = BODY[start:]
        h.send_response(206 if rng else 200)
        if rng:
            h.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        if h.command == "HEAD":
            return
        limit = cut_first if (cut_first is not None and state["n"] == 1) else len(body)
        step = chunk or len(body)
        sent = 0
        while sent < min(limit, len(body)):
            if silent_after is not None and start + sent >= silent_after:
                time.sleep(30)
                return
            piece = body[sent:min(sent + step, limit)]
            try:
                h.wfile.write(piece)
                h.wfile.flush()
            except OSError:
                return
            sent += len(piece)
            if every:
                time.sleep(every)
        if limit < len(body):
            h.close_connection = True
    return behave, state


class Scaled(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(netpolicy, "CONNECT_LIMITS", (0.4, 0.8)), mock.patch.object(netpolicy, "RETRY_GAP_S", 0.05),
                  mock.patch.object(netpolicy, "HOSTS", netpolicy.HostMemory()),
                  mock.patch.object(transfer, "_opener", netpolicy.build_opener(urllib.request.ProxyHandler({}), transfer._PolicyRedirects))):
            p.start()
            self.addCleanup(p.stop)
        t = tempfile.TemporaryDirectory(prefix="strata-nettransfer-")
        self.addCleanup(t.cleanup)
        self.dir = Path(t.name)

    def fetch(self, url, **kw):
        return transfer.download(url, self.dir / "f.part", size=len(BODY), sha256=SHA, **kw)


class TransferRule(Scaled):
    def test_a_black_hole_fails_fast_with_the_plain_sentence(self):
        hole = BlackHole()
        self.addCleanup(hole.close)
        lines = []
        t0 = time.monotonic()
        with mock.patch.object(netpolicy, "WAIT_LINE_AFTER_S", 0.2), mock.patch.object(netpolicy, "WAIT_LINE_EVERY_S", 0.3):
            with self.assertRaises(transfer.Unreachable) as cm:
                self.fetch(f"http://127.0.0.1:{hole.port}/e.tar.gz", what="the engine 0.1.59 (cuda)", say=lines.append)
        self.assertLess(time.monotonic() - t0, 2.5)
        self.assertTrue(str(cm.exception).startswith("127.0.0.1 did not answer within 1 seconds, so the engine 0.1.59 (cuda) could not be downloaded. "
                                                     "Check that this computer can open https://127.0.0.1"), str(cm.exception))
        self.assertTrue(lines and lines[0].startswith("Waiting for 127.0.0.1 to answer ("), lines)

    def test_a_refused_port_fails_at_once(self):
        t0 = time.monotonic()
        with mock.patch.object(netpolicy, "CONNECT_LIMITS", REFUSED_WINDOW), self.assertRaises(transfer.Unreachable) as cm:
            self.fetch(f"http://127.0.0.1:{refused_port()}/e.tar.gz")
        self.assertLess(time.monotonic() - t0, AT_ONCE_S)
        self.assertEqual(cm.exception.failure.kind, "offline")

    def test_a_working_download(self):
        behave, _st = ranged()
        srv = Server(behave)
        self.addCleanup(srv.close)
        self.assertEqual(self.fetch(srv.url("/e")).read_bytes(), BODY)

    def test_a_connection_that_breaks_after_bytes_came_is_resumed_in_the_same_run(self):
        behave, st = ranged(cut_first=3000)
        srv = Server(behave)
        self.addCleanup(srv.close)
        self.assertEqual(self.fetch(srv.url("/e")).read_bytes(), BODY)       # (the whole file's SHA-256 is checked by download())
        self.assertEqual(st["ranges"], [None, "bytes=3000-"], "exactly one more request, from the byte the cut reached: no byte twice")

    def test_a_trickle_is_never_cut(self):
        behave, _st = ranged(every=0.15, chunk=512)                # 20 writes, 3 s in all, each gap far under the stall time
        srv = Server(behave)
        self.addCleanup(srv.close)
        with mock.patch.object(netpolicy, "GIVE_UP_S", 1.0):
            self.assertEqual(self.fetch(srv.url("/e"), timeout=1.0).read_bytes(), BODY)

    def test_five_minutes_without_a_byte_end_the_download_with_w6(self):
        behave, _st = ranged(silent_after=5120, chunk=1024)       # half the file, then nothing (the connection stays open)
        srv = Server(behave)
        self.addCleanup(srv.close)
        with mock.patch.object(netpolicy, "GIVE_UP_S", 1.5):
            t0 = time.monotonic()
            with self.assertRaises(transfer.TransferError) as cm:
                self.fetch(srv.url("/e"), timeout=0.4, what="the engine 0.1.59 (cuda)")
        self.assertLess(time.monotonic() - t0, 6)
        self.assertEqual(str(cm.exception), "No data arrived for 1 minutes, so the download of the engine 0.1.59 (cuda) stopped at 50%. "
                                            "Run the same command again: it continues where it stopped.")
        self.assertTrue(cm.exception.resumable)
        self.assertEqual((self.dir / "f.part").stat().st_size, 5120)
        # the next run (a server that now answers) continues from the kept size: one request with the header "Range: bytes=5120-", the whole file's hash is checked
        again, st2 = ranged()
        srv2 = Server(again)
        self.addCleanup(srv2.close)
        self.assertEqual(self.fetch(srv2.url("/e")).read_bytes(), BODY)
        self.assertEqual(st2["ranges"], ["bytes=5120-"])

    @unittest.skipUnless(os.environ.get("STRATA_SLOW_NET_TESTS") == "1", "real-time: STRATA_SLOW_NET_TESTS=1")
    def test_real_time_trickle_for_more_than_five_minutes_with_the_rules_own_numbers(self):
        """A few bytes every 20-50 s for 365 s, STALL_S 60 and GIVE_UP_S 300 as shipped: the slow link that works is not cut."""
        gaps = [20, 35, 50, 25, 45, 30, 50, 40, 20, 50]
        data = b"abcdefgh" * len(gaps)
        sha = hashlib.sha256(data).hexdigest()

        def behave(h, n):
            h.send_response(200)
            h.send_header("Content-Length", str(len(data)))
            h.end_headers()
            for i, g in enumerate(gaps):
                time.sleep(g)
                h.wfile.write(data[8 * i:8 * i + 8])
                h.wfile.flush()
        srv = Server(behave)
        self.addCleanup(srv.close)
        t0 = time.monotonic()
        with mock.patch.object(netpolicy, "CONNECT_LIMITS", (netpolicy.FIRST_CONNECT_S, netpolicy.RETRY_CONNECT_S)):
            got = transfer.download(srv.url("/slow"), self.dir / "slow.part", size=len(data), sha256=sha)
        self.assertEqual(got.read_bytes(), data)
        self.assertGreater(time.monotonic() - t0, 360)


class FeedRule(Scaled):
    def test_a_feed_server_that_does_not_answer_says_w7(self):
        hole = BlackHole()
        self.addCleanup(hole.close)
        with mock.patch.object(transfer, "FIRST_TRY_SECONDS", 0.4):
            with self.assertRaises(feed.FeedError) as cm:
                feed.fetch(f"http://127.0.0.1:{hole.port}/strata-feed.json", {"k": b"x" * 32}, timeout=0.8)
        self.assertEqual(str(cm.exception), "127.0.0.1 did not answer within 1 seconds, so the update check could not run. Check that this computer "
                                            "can open https://127.0.0.1, then try again. Strata keeps working with what is installed.")
        t0 = time.monotonic()                                       # the same run asks again: no second wait
        with self.assertRaises(feed.FeedError) as cm:
            feed.fetch(f"http://127.0.0.1:{hole.port}/strata-feed.json", {"k": b"x" * 32}, timeout=0.8)
        self.assertLess(time.monotonic() - t0, 0.2)
        self.assertEqual(str(cm.exception), "127.0.0.1 did not answer a moment ago, so the update check was not tried again. "
                                            "Strata keeps working with what is installed.")

    def test_an_offline_computer_gets_the_answer_at_once(self):
        t0 = time.monotonic()
        with self.assertRaises(feed.FeedError) as cm:
            feed.fetch(f"http://127.0.0.1:{refused_port()}/strata-feed.json", {"k": b"x" * 32}, timeout=30)
        self.assertLess(time.monotonic() - t0, AT_ONCE_S)           # no wait and no retry for an offline user (owner 11-01)
        self.assertEqual(str(cm.exception), "this computer could not connect to 127.0.0.1 (it looks offline), so the update check could not run. "
                                            "Strata keeps working with what is installed.")


class LocalSource(Scaled):
    """A local source (a plain path or a file: address) that cannot be OPENED fails at once with the error it had before the network rule (a
    TransferError, resumable, "the download of ... was interrupted after 0 of N bytes (FileNotFoundError: ...)"), not after the rule's 5 minutes
    of retries; a read that fails after the open is still resumed in the same run (a USB or network drive that hiccups mid-copy)."""

    def gone(self, location):
        t0 = time.monotonic()
        with self.assertRaises(transfer.TransferError) as cm:
            self.fetch(location)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertIs(type(cm.exception), transfer.TransferError, "a plain TransferError, as before: not an Interrupted that is retried")
        self.assertTrue(cm.exception.resumable)
        self.assertTrue(str(cm.exception).startswith(f"the download of {location} was interrupted after 0 of {len(BODY)} bytes ("), str(cm.exception))
        self.assertTrue(str(cm.exception).endswith("); the partial file is kept and the next try resumes it"), str(cm.exception))
        self.assertFalse((self.dir / "f.part").exists(), "nothing written")
        return str(cm.exception)

    def test_a_missing_file_address_fails_at_once(self):
        said = self.gone((self.dir / "no-such-file.pack").as_uri())
        self.assertIn("(FileNotFoundError: ", said)

    def test_a_missing_plain_path_fails_at_once(self):
        said = self.gone(str(self.dir / "no-such-folder" / "file.pack"))
        self.assertIn("(FileNotFoundError: ", said)

    def test_a_folder_given_as_the_file_fails_at_once(self):
        (self.dir / "a-folder").mkdir()
        self.gone(str(self.dir / "a-folder"))               # IsADirectoryError (Linux, macOS) or PermissionError (Windows)

    def test_a_read_that_fails_after_the_open_is_resumed_in_the_same_run(self):
        source = self.dir / "source.bin"
        source.write_bytes(BODY)
        real_copy = transfer._copy
        calls = []

        class Hiccup:                                       # the first copy's source breaks after 3000 bytes, once
            def __init__(self, f):
                self.f, self.n = f, 0

            def read1(self, k):
                if self.n >= 3000:
                    raise OSError(5, "Input/output error")
                piece = self.f.read(min(k, 3000 - self.n))
                self.n += len(piece)
                return piece

        def copy(src, partial, mode, done, size, progress, cancel):
            calls.append((mode, done))
            return real_copy(Hiccup(src) if len(calls) == 1 else src, partial, mode, done, size, progress, cancel)

        with mock.patch.object(transfer, "_copy", copy):
            self.assertEqual(self.fetch(str(source)).read_bytes(), BODY)
        self.assertEqual(calls, [("wb", 0), ("ab", 3000)], "one more copy, from the byte the first one reached: no byte twice")


class Destination(Scaled):
    """The .part file cannot be written (no permission, a full disk, a quota, a read-only drive): raised at once with the TransferError and words
    it had before the network rule ("... (PermissionError: ...)" / "(OSError: [Errno 28] No space left on device)"), not retried for 5 minutes;
    what was written stays, so freeing the space and fetching again resumes it."""

    def fast_failure(self, url):
        t0 = time.monotonic()
        with self.assertRaises(transfer.TransferError) as cm:
            self.fetch(url)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertIs(type(cm.exception), transfer.TransferError, "a plain TransferError, as before: not an Interrupted that is retried")
        self.assertTrue(cm.exception.resumable)
        self.assertTrue(str(cm.exception).endswith("); the partial file is kept and the next try resumes it"), str(cm.exception))
        return str(cm.exception)

    def test_a_part_file_that_may_not_be_opened_fails_at_once(self):
        source = self.dir / "source.bin"
        source.write_bytes(BODY)
        real_open = Path.open

        def no_part(path, *a, **k):
            if path.name == "f.part":
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, *a, **k)

        with mock.patch.object(Path, "open", no_part):
            said = self.fast_failure(str(source))
        self.assertIn("after 0 of 10240 bytes (PermissionError: [Errno 13] Permission denied: ", said)
        self.assertFalse((self.dir / "f.part").exists())

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() != 0, "a folder without write permission: POSIX, not root")
    def test_a_folder_without_write_permission_fails_at_once(self):
        source = self.dir / "source.bin"
        source.write_bytes(BODY)
        locked = self.dir / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        self.addCleanup(locked.chmod, 0o700)
        t0 = time.monotonic()
        with self.assertRaises(transfer.TransferError) as cm:
            transfer.download(str(source), locked / "f.part", size=len(BODY), sha256=SHA)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertIs(type(cm.exception), transfer.TransferError)
        self.assertIn("(PermissionError: [Errno 13] Permission denied: ", str(cm.exception))
        self.assertFalse((locked / "f.part").exists())

    def test_a_full_disk_fails_at_once_keeps_what_was_written_and_resumes_once_there_is_room(self):
        behave, st = ranged(chunk=2000, every=0.01)
        srv = Server(behave)
        self.addCleanup(srv.close)
        real_open = Path.open

        class Full:                                        # the disk takes the first write, then: no space left
            def __init__(self, f):
                self.f, self.n = f, 0

            def write(self, data):
                if self.n:
                    raise OSError(28, "No space left on device")
                self.n += len(data)
                return self.f.write(data)

            def flush(self):
                self.f.flush()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.f.close()

        def full_part(path, *a, **k):
            f = real_open(path, *a, **k)
            return Full(f) if path.name == "f.part" else f

        with mock.patch.object(Path, "open", full_part):
            said = self.fast_failure(srv.url("/e"))
        kept = (self.dir / "f.part").read_bytes()
        self.assertTrue(0 < len(kept) < len(BODY), len(kept))
        self.assertEqual(kept, BODY[:len(kept)], "what was written is kept")
        self.assertIn(f"after {len(kept)} of 10240 bytes (OSError: [Errno 28] No space left on device)", said)
        self.assertEqual(self.fetch(srv.url("/e")).read_bytes(), BODY)       # room again: the next try resumes
        self.assertEqual(st["ranges"][-1], f"bytes={len(kept)}-")


if __name__ == "__main__":
    unittest.main()
