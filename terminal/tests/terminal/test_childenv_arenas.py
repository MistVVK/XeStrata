"""P11 (0.2.16): MALLOC_ARENA_MAX=2 for the server and the engine on Linux.

SOAK-SERVE pair (l620, same head, engine, seed and schedule, 120 active minutes): server RSS 312 MiB and +4.5 %/h by default, 252 MiB and
+0.69 %/h with MALLOC_ARENA_MAX=2. People never set environment variables, so Control sets it where it starts the server and the engine.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from strata_control import childenv  # noqa: E402

LINUX = sys.platform.startswith("linux")


class ArenaEnv(unittest.TestCase):
    @unittest.skipUnless(LINUX, "Linux only")
    def test_it_is_set_for_a_child_on_linux(self):
        env = childenv.arena_env({"PATH": "/usr/bin"})
        self.assertEqual(env["MALLOC_ARENA_MAX"], "2")
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_an_explicit_value_always_wins(self):
        env = childenv.arena_env({"MALLOC_ARENA_MAX": "8"})
        self.assertEqual(env["MALLOC_ARENA_MAX"], "8")

    def test_it_changes_and_returns_the_dict_it_was_given(self):
        given = {}
        self.assertIs(childenv.arena_env(given), given)

    def test_other_systems_get_nothing(self):
        with mock.patch.object(childenv.sys, "platform", "win32"):
            self.assertNotIn("MALLOC_ARENA_MAX", childenv.arena_env({}))
        with mock.patch.object(childenv.sys, "platform", "darwin"):
            self.assertNotIn("MALLOC_ARENA_MAX", childenv.arena_env({}))

    @unittest.skipUnless(LINUX, "Linux only")
    def test_without_a_dict_it_is_this_process_environment(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MALLOC_ARENA_MAX", None)
            childenv.arena_env()
            self.assertEqual(os.environ["MALLOC_ARENA_MAX"], "2")


class OwnArenas(unittest.TestCase):
    def test_not_when_the_variable_was_there_at_start(self):
        with mock.patch.dict(os.environ, {"MALLOC_ARENA_MAX": "2"}):
            self.assertFalse(childenv.limit_own_arenas())

    def test_not_off_linux(self):
        with mock.patch.object(childenv.sys, "platform", "win32"):
            self.assertFalse(childenv.limit_own_arenas())

    @unittest.skipUnless(LINUX, "Linux only")
    def test_a_c_library_without_the_call_is_survived(self):
        class NoMallopt:
            def __getattr__(self, name):
                raise AttributeError(name)
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch("ctypes.CDLL", return_value=NoMallopt()):
            os.environ.pop("MALLOC_ARENA_MAX", None)
            self.assertFalse(childenv.limit_own_arenas())

    @unittest.skipUnless(LINUX, "Linux only")
    def test_glibc_takes_the_limit_while_running(self):
        import ctypes
        libc = ctypes.CDLL(None)
        if not hasattr(libc, "mallopt"):
            self.skipTest("not glibc")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MALLOC_ARENA_MAX", None)
            self.assertTrue(childenv.limit_own_arenas())          # mallopt returns 1 on success


class WhereItIsSet(unittest.TestCase):
    def test_the_hosts_server_environment_and_the_engines_environment_use_it(self):
        runtime = (ROOT / "strata_control" / "runtime.py").read_text(encoding="utf-8")
        server = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
        self.assertIn("child_env = arena_env(quiet_crash_env(os.environ.copy()))", runtime)      # host -> server
        self.assertIn("env=arena_env(os.environ.copy())", runtime)                                # launcher -> host
        self.assertIn("env = arena_env(quiet_crash_env(dict(os.environ)))", server)               # server -> engine
        self.assertIn("limit_own_arenas()", runtime)
        self.assertIn("limit_own_arenas()", server)


if __name__ == "__main__":
    unittest.main()
