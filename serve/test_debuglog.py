# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Packing the debug chat log: old days, idle chats, the size limit and failures."""
import contextlib
import io
import lzma
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve import debuglog


class DebugLogPacking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.log = debuglog.DebugLog(self.folder, 1)

    def old_day(self, lines=100):
        body = b"".join(b'{"n": %d, "t": "%s"}\n' % (i, b"x" * 400) for i in range(lines))
        path = self.folder / "requests-2000-01-01.jsonl"
        path.write_bytes(body)
        return path, body

    @staticmethod
    def packed(path):
        return path.with_name(path.name + ".xz")

    @staticmethod
    def pause_compression():
        real = lzma.LZMACompressor
        started, release = threading.Event(), threading.Event()

        class Pausing:
            def __init__(self, *a, **kw):
                self.inner = real(*a, **kw)
                self.first = True

            def compress(self, data):
                if self.first:
                    self.first = False
                    started.set()
                    release.wait(2)
                return self.inner.compress(data)

            def flush(self):
                return self.inner.flush()

        return mock.patch.object(debuglog.lzma, "LZMACompressor", Pausing), started, release

    def test_old_day_is_packed_with_the_same_content(self):
        path, body = self.old_day()
        self.log.pack()
        self.assertFalse(path.exists())
        self.assertEqual(lzma.open(self.packed(path), "rb").read(), body)
        self.assertLess(self.packed(path).stat().st_size, len(body))

    def test_limit_counts_the_packed_size(self):
        path, body = self.old_day(4000)
        self.assertGreater(len(body), self.log.limit)   # the plain day was over the limit
        self.log.pack()
        self.assertLess(self.packed(path).stat().st_size, self.log.limit)
        self.log.request({"n": "today"})
        self.assertTrue(self.packed(path).exists())     # the packed day is well under it
        self.assertTrue((self.folder / f"requests-{time.strftime('%Y-%m-%d')}.jsonl").exists())

    def test_a_chat_saved_again_is_plain_and_has_no_packed_copy(self):
        self.assertTrue(self.log.chat("c1", {"m": 1}))
        plain = self.folder / "chats" / "c1.json"
        packed = self.packed(plain)
        old = time.time() - debuglog.CHAT_IDLE_S - 60
        os.utime(plain, (old, old))
        self.log.pack()
        self.assertFalse(plain.exists())
        self.assertEqual(lzma.open(packed, "rb").read(), b'{"m": 1}')
        self.assertAlmostEqual(packed.stat().st_mtime, old, delta=2)
        self.log.chat("c1", {"m": 2})
        self.assertEqual(plain.read_text(), '{"m": 2}')
        self.assertFalse(packed.exists())

    def test_the_first_request_of_a_new_day_packs_in_the_background(self):
        path, body = self.old_day()
        self.log.day = "1999-12-31"
        self.log.request({"n": "first of the new day"})
        packed = self.packed(path)
        # the files swap inside the packing, the flag clears when the whole round is over (the trim follows the
        # swap): wait for the flag too, not for the files alone
        deadline = time.monotonic() + 5
        while (path.exists() or not packed.exists() or self.log.packing) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(path.exists())
        self.assertEqual(lzma.open(packed, "rb").read(), body)
        self.assertFalse(self.log.packing)

    def test_lines_written_after_packing_are_appended(self):
        path, body = self.old_day()
        self.log.pack()
        packed = self.packed(path)
        path.write_bytes(b'{"late": 1}\n')
        self.log.pack()
        self.assertEqual(lzma.open(packed, "rb").read(), body + b'{"late": 1}\n')
        self.assertFalse(path.exists())

    def test_a_day_written_while_packing_is_left_plain(self):
        path, body = self.old_day()
        patcher, started, release = self.pause_compression()
        with patcher:
            thread = threading.Thread(target=self.log.pack)
            thread.start()
            self.assertTrue(started.wait(2))
            path.write_bytes(body + b'{"late": 1}\n')
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(path.exists())                  # the lines added meanwhile stay plain
        self.assertFalse(self.packed(path).exists())
        self.log.pack()
        self.assertEqual(lzma.open(self.packed(path), "rb").read(), body + b'{"late": 1}\n')

    def test_requests_go_on_while_packing(self):
        path, body = self.old_day()
        patcher, started, release = self.pause_compression()
        with patcher:
            thread = threading.Thread(target=self.log.pack)
            thread.start()
            self.assertTrue(started.wait(2))
            done = threading.Event()

            def request():
                self.log.request({"n": "while packing"})
                done.set()

            threading.Thread(target=request).start()
            self.assertTrue(done.wait(2))               # the compression holds no lock
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(lzma.open(self.packed(path), "rb").read(), body)

    def test_bad_packed_file_keeps_the_plain_and_is_said_once(self):
        path, body = self.old_day()
        self.packed(path).write_bytes(b"junk")
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.log.pack()
            self.log.pack()
        self.assertEqual(path.read_bytes(), body)
        self.assertEqual(log.getvalue().count("debug chat log"), 1)


if __name__ == "__main__":
    unittest.main()
