"""The quiet line that says a newer engine is available, and the once-a-day check behind it (owner, 11-01).

* the line comes from the last check's cached answer (a local file): one sentence, one next step, no network;
* it is in the launch start screen and the launch monitor (and `strata update --check`), nowhere else;
* the daily check is made by the launch monitor only, in a thread of its own, once a day, with a short timeout, and says NOTHING when it cannot reach
  the feed: no internet, blocked DNS, a proxy that refuses, a captive portal that answers junk, a connection that never answers.  The attempt is
  written before the network is touched, so a failure is not retried until tomorrow.  Offline, the launch, the monitor and the line behave exactly as
  they do with a working network that has nothing new.
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import io
import json
import logging
import os
import socket
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_cli.launch import app as launch_app  # noqa: E402
from strata_cli.launch.monitor_ext import LaunchMonitor  # noqa: E402
from strata_control import engine_component as ec, engine_update, updater  # noqa: E402
from tests.terminal import update_fixture as fx  # noqa: E402
from tests.terminal.launch_support import text_of  # noqa: E402
from tests.terminal.test_launch_monitor import INFO, flat  # noqa: E402
from tests.terminal.test_launch_tui import Base as TuiBase, make_ctx  # noqa: E402
from tests.terminal.test_overview import Source, metrics  # noqa: E402
from tests.terminal.test_updater import UpdaterCase  # noqa: E402


class Cached(UpdaterCase):
    """The line is built from the cached check only."""

    def remember(self, **offers):
        state = self.components / ".updates" / "state.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        last = {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "engine": None, "app": None, "dsv41": None,
                "control": ec.runtime_version(), "engine_installed": ec.newest_installed_version(self.components), **offers}
        state.write_text(json.dumps({"schema_version": 1, "last_check": last}))

    def test_nothing_cached_nothing_shown(self):
        self.install_current("0.1.45")
        self.assertIsNone(engine_update.notice(self.components))
        self.assertIsNone(engine_update.notice_line(self.components))
        self.assertIsNone(engine_update.monitor_note(self.components))

    def test_a_newer_engine_is_one_sentence_and_one_next_step(self):
        self.install_current("0.1.45")
        self.remember(engine="0.1.56")
        line = engine_update.notice_line(self.components)
        self.assertEqual(line, "A new engine 0.1.56 (you have 0.1.45) is available. Update with `strata update`.")
        self.assertEqual(engine_update.monitor_note(self.components), "New engine 0.1.56 is available. Press u to update.")
        self.assertNotIn("\n", line)

    def test_a_cached_engine_that_is_not_newer_is_never_shown(self):
        self.install_current("0.1.56")
        self.remember(engine="0.1.56", engine_installed="0.1.56")
        self.assertIsNone(engine_update.notice(self.components))

    def test_the_line_is_dropped_when_the_engine_changed_since_the_check(self):
        self.install_current("0.1.45")
        self.remember(engine="0.1.56", engine_installed="0.1.44")
        self.assertIsNone(engine_update.notice(self.components))

    def test_reading_the_line_touches_no_network(self):
        self.install_current("0.1.45")
        self.remember(engine="0.1.56")
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network")), mock.patch("socket.create_connection", side_effect=AssertionError("network")):
            self.assertIsNotNone(engine_update.notice_line(self.components))

    def test_a_check_writes_what_the_line_reads(self):
        self.install_current("0.1.45")
        self.publish(("0.1.56", fx.GOOD_ENGINE))
        updater.check(self.ctx())
        self.assertIn("0.1.56", engine_update.notice_line(self.components))
        self.assertEqual(self.archive_requests(), [], "a check downloads no engine")


class DailyCheck(UpdaterCase):
    def setUp(self):
        super().setUp()
        self.install_current("0.1.45")
        self.calls = 0

    def check(self):
        self.calls += 1

    def attempt_file(self):
        return self.components / ".updates" / engine_update.ATTEMPT_FILE

    def test_it_runs_once_and_not_again_the_same_day(self):
        self.assertTrue(engine_update.refresh_cache(self.check, self.components))
        self.assertFalse(engine_update.refresh_cache(self.check, self.components))
        self.assertEqual(self.calls, 1)

    def test_a_failed_check_is_not_retried_until_tomorrow(self):
        def broken():
            self.calls += 1
            raise OSError("no route to host")
        self.assertFalse(engine_update.refresh_cache(broken, self.components))
        self.assertFalse(engine_update.refresh_cache(broken, self.components))
        self.assertEqual(self.calls, 1, "the attempt was written before the network was touched")
        old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=25)).isoformat()
        self.attempt_file().write_text(json.dumps({"at": old}))
        self.assertTrue(engine_update.refresh_cache(self.check, self.components))
        self.assertEqual(self.calls, 2)

    def test_a_check_somebody_made_today_counts(self):
        state = self.components / ".updates" / "state.json"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"last_check": {"at": dt.datetime.now(dt.timezone.utc).isoformat()}}))
        self.assertFalse(engine_update.refresh_cache(self.check, self.components))
        self.assertEqual(self.calls, 0)

    def test_it_can_be_switched_off(self):
        for value in ("0", "off", "never"):
            with mock.patch.dict(os.environ, {engine_update.CHECK_ENV: value}):
                self.assertFalse(engine_update.refresh_cache(self.check, self.components))
        self.assertEqual(self.calls, 0)
        self.assertFalse(self.attempt_file().exists())

    def test_it_never_runs_in_a_process_that_forbids_acquiring(self):
        with mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": "1"}):
            self.assertIsNone(engine_update.start_daily_check(root=self.components))

    def test_an_attempt_that_cannot_be_remembered_is_not_made(self):
        with mock.patch.object(Path, "write_text", side_effect=OSError("read-only")):
            self.assertFalse(engine_update.refresh_cache(self.check, self.components))
        self.assertEqual(self.calls, 0)


class Portal(BaseHTTPRequestHandler):
    """A captive portal: 200 and a web page for every address."""

    def do_GET(self):
        body = b"<html><body>Please sign in to the hotel WiFi</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class Offline(UpdaterCase):
    """The real daily check against a network that is not there.  Whatever is wrong, the thread ends without a word and the person sees nothing."""

    def setUp(self):
        super().setUp()
        self.install_current("0.1.45")

    def run_check(self, feed, join=60.0, env=None):
        out, err = io.StringIO(), io.StringIO()
        before = time.monotonic()
        from tests.terminal import net_dead
        with mock.patch.dict(os.environ, env or {}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                self.assertNoLogs(level=logging.INFO), net_dead.no_name_resolution():          # (no real name service: the offline PC is faked)
            thread = engine_update.start_daily_check(root=self.components, feed=feed)
            started_in = time.monotonic() - before
            thread.join(join)
        self.assertFalse(thread.is_alive(), "the check gave up by itself")
        self.assertLess(started_in, 1.0, "starting the check does not wait for the network")
        self.assertEqual((out.getvalue(), err.getvalue()), ("", ""), "nothing is printed")
        self.assertIsNone(engine_update.notice(self.components))
        self.assertTrue((self.components / ".updates" / engine_update.ATTEMPT_FILE).is_file(), "tomorrow, not in a minute")
        self.assertEqual(self.archive_requests(), [])

    def test_nothing_listens(self):
        self.run_check("http://127.0.0.1:9/strata-feed.json")

    def test_dns_is_blocked(self):
        self.run_check("https://strata-feed.invalid/strata-feed.json")

    def test_a_proxy_that_refuses(self):
        self.run_check("https://example.invalid/strata-feed.json",
                       env={"HTTPS_PROXY": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9"})

    def test_a_captive_portal_that_answers_junk(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Portal)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.run_check(f"http://127.0.0.1:{server.server_address[1]}/strata-feed.json")

    def test_a_connection_that_never_answers(self):
        hang = socket.socket()
        hang.bind(("127.0.0.1", 0))
        hang.listen(5)
        self.addCleanup(hang.close)
        started = time.monotonic()
        self.run_check(f"http://127.0.0.1:{hang.getsockname()[1]}/strata-feed.json", join=60.0)
        self.assertLess(time.monotonic() - started, 30, "the short timeout ended it")

    def test_a_second_try_the_same_day_makes_no_request_at_all(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Portal)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        seen = []
        original = Portal.do_GET
        Portal.do_GET = lambda handler: (seen.append(handler.path), original(handler))[1]
        self.addCleanup(setattr, Portal, "do_GET", original)
        feed = f"http://127.0.0.1:{server.server_address[1]}/strata-feed.json"
        self.run_check(feed)
        first = len(seen)
        self.assertGreater(first, 0)
        engine_update.start_daily_check(root=self.components, feed=feed).join(10)
        self.assertEqual(len(seen), first)


class OnTheScreens(TuiBase):
    def note(self, text="A new engine 0.1.56 (you have 0.1.55) is available. Update with `strata update`."):
        patch = mock.patch.object(engine_update, "notice_line", return_value=text)
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_start_screen_shows_the_line_once(self):
        self.note()

        async def case(app, pilot, ctx):
            t = " ".join(text_of(app).split())
            self.assertEqual(t.count("A new engine 0.1.56"), 1)
            self.assertIn("Update with `strata update`", t)
        self.run_app(case)

    def test_the_start_screen_without_a_newer_engine_is_as_before(self):
        async def case(app, pilot, ctx):
            self.assertNotIn("new engine", text_of(app).lower())
        self.run_app(case)


class Monitor(unittest.TestCase):
    def run_app(self, coro, info=None, note=None):
        async def go():
            app = LaunchMonitor(Source(metrics("idle")), None, info=info or dict(INFO), interval=30)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause(0.5)
                app.quiet_note = note
                app._paint()
                await pilot.pause(0.2)
                return await coro(app, pilot)
        return asyncio.run(go())

    def test_the_line_and_the_key(self):
        async def case(app, pilot):
            t = flat(app)
            self.assertIn("New engine 0.1.56 is available. Press u to update.", t)
            self.assertEqual(t.count("is available"), 1)
        self.run_app(case, note="New engine 0.1.56 is available. Press u to update.")

    def test_no_line_when_nothing_is_new(self):
        async def case(app, pilot):
            self.assertNotIn("Press u", flat(app))
        self.run_app(case)

    def test_the_daily_check_is_started_by_the_launch_monitor_only(self):
        with mock.patch.object(engine_update, "start_daily_check") as start:
            self.run_app(lambda app, pilot: asyncio.sleep(0))
            start.assert_not_called()                                        # (plain monitors, tests, `strata monitor --url`)
            self.run_app(lambda app, pilot: asyncio.sleep(0), info={**INFO, "daily_check": True})
            start.assert_called_once()

    def test_run_monitor_asks_for_the_check(self):
        seen = {}

        class Stub:
            def __init__(self, endpoint, snapshot, info, **kw):
                seen.update(info)
                self.stop_requested = False

            def run(self):
                return None
        with mock.patch("strata_cli.launch.monitor_ext.LaunchMonitor", Stub):
            launch_app.run_monitor({"url": "http://127.0.0.1:8080", "urls": ["http://127.0.0.1:8080"], "key": "", "profile": "launch-x"})
        self.assertTrue(seen.get("daily_check"))

    def test_a_monitor_beside_a_hanging_feed_is_not_slowed(self):
        hang = socket.socket()
        hang.bind(("127.0.0.1", 0))
        hang.listen(5)
        self.addCleanup(hang.close)
        with tempfile_home() as home:
            env = {"STRATA_COMPONENTS_DIR": home, "STRATA_UPDATE_FEED": f"http://127.0.0.1:{hang.getsockname()[1]}/strata-feed.json"}
            with mock.patch.dict(os.environ, env):
                started = time.monotonic()

                async def case(app, pilot):
                    self.assertNotIn("Press u", flat(app))
                    return time.monotonic() - started
                took = self.run_app(case, info={**INFO, "daily_check": True})
        self.assertLess(took, 5.0, "the monitor drew and ran while the check was still waiting")

    def test_u_asks_names_the_model_and_shows_what_happened(self):
        def flow(*, ask, say, **kw):
            say("Checking for a newer engine...")
            if not ask("Stop Demo model (port 8080) for the update?", False):
                raise engine_update.Declined("Nothing was changed. The model keeps running.")
            return {"results": [{"component": "engine", "status": "updated", "installed_after": "0.1.56"}],
                    "models": {"restarts": [{"name": "Demo model", "restarted": True, "message": "Demo model is serving again with the same settings."}]}}

        async def case(app, pilot):
            await pilot.press("u")
            await pilot.pause(0.6)
            t = flat(app)
            self.assertIn("Stop Demo model (port 8080) for the update?", t)
            await pilot.press("enter")                                       # Yes
            await pilot.pause(0.8)
            t = flat(app)
            self.assertIn("Engine is now 0.1.56", t)
            self.assertIn("serving again", t)
            self.assertFalse(app.updating)
        with mock.patch.object(engine_update, "run_flow", side_effect=flow):
            self.run_app(case, note="New engine 0.1.56 is available. Press u to update.")

    def test_u_and_no_changes_nothing(self):
        def flow(*, ask, say, **kw):
            if not ask("Stop Demo model for the update?", False):
                raise engine_update.Declined("Nothing was changed. The model keeps running.")
            raise AssertionError("must not get here")

        async def case(app, pilot):
            await pilot.press("u")
            await pilot.pause(0.6)
            await pilot.press("down", "enter")                               # No
            await pilot.pause(0.8)
            self.assertIn("Nothing was changed. The model keeps running.", flat(app))
        with mock.patch.object(engine_update, "run_flow", side_effect=flow):
            self.run_app(case)

    def test_a_flow_that_breaks_is_a_sentence_not_a_trace(self):
        async def case(app, pilot):
            await pilot.press("u")
            await pilot.pause(0.8)
            self.assertIn("The update did not run: boom", flat(app))
            self.assertTrue(app.is_running)
        with mock.patch.object(engine_update, "run_flow", side_effect=RuntimeError("boom")):
            self.run_app(case)


@contextlib.contextmanager
def tempfile_home():
    import tempfile
    # A check the app started in the background may still hold .updates/update.lock when the case ends (a file in use cannot be deleted on
    # Windows); the folder is scratch, so a leftover is not a failure of the case.
    with tempfile.TemporaryDirectory(prefix="strata-notice-", ignore_cleanup_errors=True) as d:
        yield d


if __name__ == "__main__":
    unittest.main()
