"""#4479 / #2481: the server's console output must not hold a request.

A console that stops taking output (a Windows console in QuickEdit/selection mode, a terminal that is paused, a launcher
that stopped reading the server's pipe) blocks `print()` in whatever thread called it - and some of those threads own the
engine's turn, a cancel, or an admission.  `NonBlockingOut` stands in for sys.stdout / sys.stderr while the server runs:
write() puts the text on a bounded queue and returns; one daemon thread writes it to the real stream, and may block there
as long as it likes.  When the queue is full the OLDEST text is dropped (the newest lines matter most) and counted in
`dropped_chunks` / `dropped_bytes`; nothing about a request waits for the console.

Only diagnostics go through here.  Protocol output (the engine's pipe, the HTTP replies) does not.
"""
from __future__ import annotations

import collections
import contextlib
import sys
import threading


class NonBlockingOut:
    def __init__(self, real, max_bytes: int = 1 << 20, max_chunks: int = 4000):
        self._real = real
        self.max_bytes, self.max_chunks = max_bytes, max_chunks
        self._q: collections.deque[str] = collections.deque()
        self._bytes = 0
        self._cv = threading.Condition()
        self._closing = False
        self._idle = True
        self.dropped_chunks = 0
        self.dropped_bytes = 0
        self._thread = threading.Thread(target=self._run, name="strata-console", daemon=True)
        self._thread.start()

    # ---- what the program calls: never blocks on the console
    def write(self, text) -> int:
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return 0
        with self._cv:
            if self._closing:
                return len(text)
            self._q.append(text)
            self._bytes += len(text)
            while self._q and (self._bytes > self.max_bytes or len(self._q) > self.max_chunks):
                old = self._q.popleft()
                self._bytes -= len(old)
                self.dropped_chunks += 1
                self.dropped_bytes += len(old)
            self._cv.notify()
        return len(text)

    def flush(self) -> None:
        return None                                    # the writer thread flushes the real stream

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        try:
            return bool(self._real.isatty())
        except (OSError, ValueError, AttributeError):
            return False

    def __getattr__(self, name):                        # encoding, errors, fileno, buffer ...: the real stream's
        return getattr(self._real, name)

    # ---- the one thread that may block
    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._q and not self._closing:
                    self._idle = True
                    self._cv.notify_all()
                    self._cv.wait()
                if not self._q:
                    self._idle = True
                    self._cv.notify_all()
                    return
                self._idle = False
                text = self._q.popleft()
                self._bytes -= len(text)
            try:
                self._real.write(text)
                self._real.flush()
            except (OSError, ValueError, UnicodeError):
                pass                                    # a closed or broken console: the text is lost, the server is not held

    def drain(self, timeout: float = 2.0) -> bool:
        """Wait (bounded) until everything queued has been handed to the real stream."""
        import time
        end = time.monotonic() + timeout
        with self._cv:
            while (self._q or not self._idle) and time.monotonic() < end:
                self._cv.wait(timeout=max(0.01, end - time.monotonic()))
            return not self._q and self._idle

    def close(self, timeout: float = 2.0) -> None:
        self.drain(timeout)
        with self._cv:
            self._closing = True
            self._cv.notify_all()


@contextlib.contextmanager
def decoupled_console():
    """While the server runs: stdout and stderr go through NonBlockingOut, when they are the process's own streams (a
    test runner's capture, or a stream somebody else replaced, is left alone).  Restored, and drained for a moment, at
    the end."""
    installed = []
    for name in ("stdout", "stderr"):
        cur = getattr(sys, name, None)
        if cur is not None and cur is getattr(sys, "__" + name + "__", None):
            wrapper = NonBlockingOut(cur)
            setattr(sys, name, wrapper)
            installed.append((name, cur, wrapper))
    try:
        yield [w for _, _, w in installed]
    finally:
        for name, orig, wrapper in installed:
            wrapper.close()
            if getattr(sys, name, None) is wrapper:
                setattr(sys, name, orig)
