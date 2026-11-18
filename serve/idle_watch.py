# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""One scheduler for idle actions in each process, including the independent image daemon."""
from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections.abc import Callable

FAIL_LIMIT = 3   # rounds in a row a repeating action may fail before the watch drops it


class Watch:
    def __init__(self):
        self.cv = threading.Condition()
        self.items: list[tuple[float, int, float, Callable[[], bool | None], int]] = []
        self.serial = itertools.count()
        self.thread: threading.Thread | None = None

    def add(self, delay: float, action: Callable[[], bool | None], repeat: bool = False, fails: int = 0) -> None:
        """Run an action after delay, optionally repeating after it finishes. Actions do not block each other.
        `fails`: the rounds of this action that failed already; the scheduler carries it, a caller does not pass it."""
        with self.cv:
            heapq.heappush(self.items, (time.monotonic() + delay, next(self.serial), delay if repeat else 0, action,
                                        fails))
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True, name="xestrata-idle-watch")
                self.thread.start()
            self.cv.notify()

    def _run(self) -> None:
        while True:
            with self.cv:
                while not self.items:
                    self.cv.wait()
                due, _, interval, action, fails = self.items[0]
                wait = due - time.monotonic()
                if wait > 0:
                    self.cv.wait(wait)
                    continue
                heapq.heappop(self.items)
            threading.Thread(target=self._call, args=(interval, action, fails), daemon=True).start()

    def _call(self, interval: float, action: Callable[[], bool | None], fails: int = 0) -> None:
        """The action, on its own thread.  An exception is said in the log and counted, and a repeating action is not
        re-armed after FAIL_LIMIT rounds in a row: the thread's traceback is the only thing a server would ever hear
        of an action that always fails, and it would print one every interval forever."""
        again: bool | None = None
        try:
            again = action()
        except Exception as e:
            print(f"[xestrata] WARNING: idle action {getattr(action, '__qualname__', None) or repr(action)} "
                  f"failed ({e!r})", flush=True)
            fails += 1
        else:
            fails = 0                 # FAIL_LIMIT is rounds in a row, not failures over the watch's whole life
        if interval and again is not False and fails < FAIL_LIMIT:
            self.add(interval, action, repeat=True, fails=fails)


watch = Watch()
