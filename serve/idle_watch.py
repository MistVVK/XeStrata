# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""One scheduler for idle actions in each process, including the independent image daemon."""
from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections.abc import Callable


class Watch:
    def __init__(self):
        self.cv = threading.Condition()
        self.items: list[tuple[float, int, float, Callable[[], bool | None]]] = []
        self.serial = itertools.count()
        self.thread: threading.Thread | None = None

    def add(self, delay: float, action: Callable[[], bool | None], repeat: bool = False) -> None:
        """Run an action after delay, optionally repeating after it finishes. Actions do not block each other."""
        with self.cv:
            heapq.heappush(self.items, (time.monotonic() + delay, next(self.serial), delay if repeat else 0, action))
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True, name="xestrata-idle-watch")
                self.thread.start()
            self.cv.notify()

    def _run(self) -> None:
        while True:
            with self.cv:
                while not self.items:
                    self.cv.wait()
                due, _, interval, action = self.items[0]
                wait = due - time.monotonic()
                if wait > 0:
                    self.cv.wait(wait)
                    continue
                heapq.heappop(self.items)
            threading.Thread(target=self._call, args=(interval, action), daemon=True).start()

    def _call(self, interval: float, action: Callable[[], bool | None]) -> None:
        again: bool | None = None
        try:
            again = action()
        finally:
            if interval and again is not False:
                self.add(interval, action, repeat=True)


watch = Watch()
