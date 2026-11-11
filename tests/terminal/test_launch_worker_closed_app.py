"""The launch screen's worker thread (LaunchApp._work) reports the run's end to the screen with call_from_thread.  When
the person quits while the run is ending, the app's event loop is already closed and call_from_thread raises
RuntimeError ("Event loop is closed") in that thread, which printed a traceback after the screen was gone.  Every one of
those calls must be as guarded as _emit's and _ask_from_run's already are.  Here the app's loop is a closed asyncio loop
(the state after a quit) and _work runs each way a run can end."""

from __future__ import annotations

import asyncio
import gc
import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strata_cli.launch.app import LaunchApp  # noqa: E402
from strata_cli.launch.runner import Cancelled, LaunchError  # noqa: E402


class WorkerAfterQuit(unittest.TestCase):
    def app(self, run):
        app = LaunchApp.__new__(LaunchApp)
        loop = asyncio.new_event_loop()
        loop.close()                                       # the app has quit: its loop is closed
        app._loop = loop
        app._thread_id = -1                                # (the worker is another thread than the app's)
        app.run_screen = mock.Mock()
        app.runner = SimpleNamespace(run=run)
        app.ctx = SimpleNamespace(settings=SimpleNamespace(name=None))
        app._save_pending = lambda: None
        app.running = True
        return app

    def check(self, run):
        app = self.app(run)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(RuntimeError):          # the precondition: this app's loop refuses a call
                app.call_from_thread(lambda: None)
            gc.collect()                                   # (that call's own unawaited coroutine, not _work's)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            app._work()                                    # must not raise in the worker thread
            gc.collect()
        self.assertFalse(app.running)
        self.assertEqual([str(w.message) for w in seen if "never awaited" in str(w.message)], [],
                         "a call to the closed app left a coroutine behind (a warning on the terminal)")

    def test_finished(self):
        self.check(lambda: "result")

    def test_cancelled(self):
        def run():
            raise Cancelled()
        self.check(run)

    def test_launch_error(self):
        def run():
            raise LaunchError("could not start", "try again")
        self.check(run)

    def test_unexpected_error(self):
        def run():
            raise KeyError("x")
        self.check(run)


if __name__ == "__main__":
    unittest.main()
