"""#6055: a server stopped without --shutdown-timeout refuses new inference while the engine closes.

Before, admission stayed open during the stop (begin_drain ran only with --shutdown-timeout), so a chat request that
arrived in those seconds was accepted, the engine closed under it, and it came back as 200 with 0 tokens and
finish_reason "length". Now the stop always closes admission first: such a request gets 503 server_draining.
"""

import contextlib
import io
import os
import threading
import time as real_time
import unittest
from unittest import mock

from serve import server


class _Clock:
    """serve.server's `time`, except that the main thread's idle `time.sleep(1)` is a Ctrl+C (the stop)."""

    def __getattr__(self, name):
        return getattr(real_time, name)

    @staticmethod
    def sleep(seconds):
        if threading.current_thread() is threading.main_thread() and seconds == 1:
            raise KeyboardInterrupt
        real_time.sleep(seconds)


class StopClosesAdmission(unittest.TestCase):
    def run_stop(self, extra):
        drains, states = [], []
        original = server.Service.begin_drain

        def begin_drain(svc):
            drains.append(svc)
            pending = original(svc)
            states.append(svc.reserve_inference())        # what a request arriving now gets: None = 503 server_draining
            return pending

        env = {k: v for k, v in os.environ.items() if k not in ("STRATA_API_KEY", "STRATA_TELEMETRY_KEY")}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(server, "time", _Clock()), \
                mock.patch.object(server.lifeline, "install_stop_handlers", lambda *a, **k: None), \
                mock.patch.object(server.Service, "begin_drain", begin_drain), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            code = server.main(["--engine", "mock", "--port", "0", *extra])
        return code, drains, states, out.getvalue()

    def test_stop_without_shutdown_timeout_refuses_new_requests(self):
        code, drains, states, out = self.run_stop([])
        self.assertIn("[strata] stopped", out)
        self.assertEqual(len(drains), 1, "the stop closes admission even without --shutdown-timeout")
        self.assertEqual(states, [None])
        self.assertNotIn("draining", out)                 # no wait and no drain line: only admission is closed
        self.assertEqual(code, 0)                         # a clean stop: nothing was in flight, so nothing undrained

    def test_stop_with_shutdown_timeout_is_unchanged(self):
        code, drains, states, _out = self.run_stop(["--shutdown-timeout", "0"])
        self.assertEqual(len(drains), 1)
        self.assertEqual(states, [None])


if __name__ == "__main__":
    unittest.main()
