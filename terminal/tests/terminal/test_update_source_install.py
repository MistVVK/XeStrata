"""#4463: `strata update` on a source install.

Gap 1: with no folder named, it used the per-user component folder even when the checkout has its own engine store (`engine-feed`, which setup.py
fills); the download, its half-written state and its part file then sat in a folder the checkout never read.
Gap 2: the busy check did not see an engine the desktop app's source install had started (it runs serve/server.py directly, which took no lease), so
`strata update` test-started a second copy of the model beside it.

    python -m pytest tests/terminal/test_update_source_install.py
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import server
from strata_cli import update_cmd
from strata_control import engine_component as ec
from strata_control import updater


class DefaultFolder(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-updsrc-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.user = self.root / "user-components"
        self.store = self.root / "checkout" / "engine-feed"
        for folder in (self.user, self.store):
            folder.mkdir(parents=True)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(ec.ENV_DIR, None)
        self.patches = [mock.patch.object(ec, "components_dir", lambda *a, **k: self.user)]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _scan(self, user_has, store_has):
        def scan(root=None):
            has = user_has if Path(root) == self.user else store_has if Path(root) == self.store else False
            return ([object()] if has else []), []
        return mock.patch.object(ec, "scan", scan)

    def test_a_checkout_with_its_own_store_updates_it_when_the_user_folder_has_no_engine(self):
        with self._scan(False, True), mock.patch.object(update_cmd, "source_store", return_value=self.store):
            self.assertEqual(update_cmd.default_components_dir(ec), self.store)

    def test_a_user_folder_that_holds_an_engine_is_kept(self):
        with self._scan(True, True), mock.patch.object(update_cmd, "source_store", return_value=self.store):
            self.assertEqual(update_cmd.default_components_dir(ec), self.user)

    def test_no_store_in_the_checkout_changes_nothing(self):
        with self._scan(False, False), mock.patch.object(update_cmd, "source_store", return_value=None):
            self.assertEqual(update_cmd.default_components_dir(ec), self.user)

    def test_an_explicit_folder_in_the_environment_always_wins(self):
        with self._scan(False, True), mock.patch.object(update_cmd, "source_store", return_value=self.store), \
                mock.patch.dict(os.environ, {ec.ENV_DIR: str(self.root / "named")}):
            self.assertEqual(update_cmd.default_components_dir(ec), self.user)      # (components_dir() itself returns the named folder)

    def test_a_store_without_an_engine_package_is_not_one(self):
        with mock.patch.object(ec, "scan", return_value=([], [])):
            self.assertIsNone(update_cmd.source_store(self.root / "checkout"))


class AppLaunchedEngine(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-updsrc-rt-")
        self.addCleanup(self.temp.cleanup)
        env = mock.patch.dict(os.environ, {"STRATA_RUNTIME_DIR": str(Path(self.temp.name) / "runtime")})
        env.start()
        self.addCleanup(env.stop)

    def test_a_server_started_by_running_serve_server_is_seen_by_the_busy_check(self):
        seen = {}

        def serving(*args, **kwargs):
            seen["during"] = updater.serving_now()
            return 0
        self.assertEqual(updater.serving_now(), [])
        with mock.patch.object(server, "_main", serving):
            self.assertEqual(server.main([]), 0)
        self.assertTrue(seen["during"], "the update's busy check did not see the running server")
        self.assertEqual(updater.serving_now(), [], "the lease is released when the server ends")

    def test_serving_does_not_depend_on_the_lease(self):
        from strata_control import runtime
        with mock.patch.object(runtime, "acquire_activity_lease", side_effect=runtime.ProfileError("a managed runtime or calibration owns the local engine lease", 6)), \
                mock.patch.object(server, "_main", return_value=7):
            self.assertEqual(server.main([]), 7)


if __name__ == "__main__":
    unittest.main()
