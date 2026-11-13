# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""upstream b776e1cd0 (#4479, #2481): a console that stops taking output must not hold the thread that prints.

The control below shows the dependency the request lifecycle had on the console: a plain print into a pipe nobody reads
blocks.  NonBlockingOut queues it instead, keeps the newest lines in order, drops (and counts) the oldest, and drains
when the console comes back.
"""
import os
import threading
import time
import unittest

from serve.nonblocking_out import NonBlockingOut


class BlockedConsole(unittest.TestCase):
    def pipe(self):
        r, w = os.pipe()
        self.addCleanup(lambda: [os.close(x) for x in (r,) if True])
        return r, os.fdopen(w, "w", encoding="utf-8", buffering=1)

    def test_a_print_to_a_full_pipe_blocks_without_the_wrapper(self):
        """The control: this is the dependency the request lifecycle had on the console."""
        r, w = self.pipe()
        done = threading.Event()

        def spam():
            try:
                for i in range(20000):
                    w.write(f"[xestrata] progress line {i} " + "x" * 40 + "\n")
                    w.flush()
            except OSError:                                  # the test ended and closed the pipe under it
                return
            done.set()
        threading.Thread(target=spam, daemon=True).start()
        self.assertFalse(done.wait(1.0), "a pipe nobody reads must block a plain print")

    def test_the_wrapper_never_blocks_and_keeps_the_newest_lines(self):
        r, w = self.pipe()
        out = NonBlockingOut(w, max_bytes=64 * 1024, max_chunks=500)
        done = threading.Event()

        def spam():
            for i in range(20000):
                print(f"[xestrata] progress line {i} " + "x" * 40, file=out, flush=True)
            done.set()
        threading.Thread(target=spam, daemon=True).start()
        self.assertTrue(done.wait(5.0), "printing into a console nobody reads must not block")
        self.assertGreater(out.dropped_chunks, 0)
        self.assertLessEqual(len(out._q), 500)
        # the console wakes up: everything that was kept arrives, in order, ending with the newest line
        data = b""
        end = time.monotonic() + 10
        while time.monotonic() < end and b"progress line 19999" not in data:
            data += os.read(r, 65536)
        text = data.decode()
        self.assertIn("progress line 19999 ", text)
        seen = [int(line.split()[3]) for line in text.splitlines() if line.startswith("[xestrata] progress line")]
        self.assertEqual(seen, sorted(seen), "order is kept")
        out.close(1.0)

    def test_the_console_is_restored_and_drained(self):
        import io
        real = io.StringIO()
        out = NonBlockingOut(real)
        out.write("hello\n")
        out.write("world\n")
        self.assertTrue(out.drain(2.0))
        self.assertEqual(real.getvalue(), "hello\nworld\n")
        out.close()
        self.assertEqual(out.write("late\n"), 5)                     # a late write after close is not an error


if __name__ == "__main__":
    unittest.main()
