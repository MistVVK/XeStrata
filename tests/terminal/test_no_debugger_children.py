"""The product never starts a debugger on a user's machine (strata_control/childenv.py).

ggml's abort path forks `gdb --batch -ex attach <pid>` (Linux; lldb and backtrace_symbols after it).  Every child Strata starts therefore gets
GGML_NO_BACKTRACE=1, unless a developer sets STRATA_BACKTRACE=1.  These tests check the helper, the environment of every spawn site that builds one, and
list every place product code starts a process, so a new one has to be looked at (does it start a Strata program? then it needs the variable).

    python -m pytest tests/terminal/test_no_debugger_children.py
"""
import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import childenv
from strata_control.childenv import DEVELOPER_SWITCH, NO_BACKTRACE, quiet_crash_env

ROOT = Path(__file__).resolve().parents[2]


class Helper(unittest.TestCase):
    def test_sets_the_variable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(quiet_crash_env({})[NO_BACKTRACE], "1")

    def test_without_an_argument_it_changes_this_process(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            quiet_crash_env()
            self.assertEqual(os.environ[NO_BACKTRACE], "1")

    def test_a_value_that_is_already_there_is_kept(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(quiet_crash_env({NO_BACKTRACE: "0"})[NO_BACKTRACE], "0")      # ggml reads any value as "skip"

    def test_the_developer_switch_leaves_the_backtrace_on(self):
        for how in ("1", "yes", "true", "anything"):
            with self.subTest(how=how), mock.patch.dict(os.environ, {DEVELOPER_SWITCH: how}, clear=True):
                self.assertNotIn(NO_BACKTRACE, quiet_crash_env(dict(os.environ)))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertNotIn(NO_BACKTRACE, quiet_crash_env({DEVELOPER_SWITCH: "1"}))       # named in the dict being built, not in this process

    def test_the_switch_set_to_nothing_or_zero_is_not_a_request(self):
        for how in ("", "0", "no", "off", "False"):
            with self.subTest(how=how), mock.patch.dict(os.environ, {DEVELOPER_SWITCH: how}, clear=True):
                self.assertEqual(quiet_crash_env(dict(os.environ))[NO_BACKTRACE], "1")

    def test_it_returns_the_dict_it_was_given(self):
        env = {"A": "1"}
        self.assertIs(quiet_crash_env(env), env)


class SpawnEnvironments(unittest.TestCase):
    """The environment each site that builds one hands to its child."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k not in (NO_BACKTRACE, DEVELOPER_SWITCH)}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_engine_and_the_picture_reader(self):
        import serve.server as server
        env = server.child_env({})
        self.assertEqual(env[NO_BACKTRACE], "1")
        self.assertEqual(server.vision_env({}, env)[NO_BACKTRACE], "1")
        self.assertEqual(server.vision_env({"vision": {"cuda_device": 1}}, env)[NO_BACKTRACE], "1")     # the copy for a reader on its own card
        self.assertEqual(server.child_env({}, offline=True)[NO_BACKTRACE], "1")

    def test_the_developer_switch_reaches_the_engine_environment(self):
        import serve.server as server
        with mock.patch.dict(os.environ, {DEVELOPER_SWITCH: "1"}):
            env = server.child_env({})
        self.assertNotIn(NO_BACKTRACE, env)

    def test_the_config_may_name_it_itself(self):
        import serve.server as server
        self.assertEqual(server.child_env({"env": {NO_BACKTRACE: "0"}})[NO_BACKTRACE], "0")

    def test_sd_cli(self):
        from serve import images
        self.assertEqual(images.image_env({})[NO_BACKTRACE], "1")
        self.assertEqual(images.image_env({}, base={"PATH": "x"})[NO_BACKTRACE], "1")

    def test_the_launcher_children(self):
        from strata_cli.launch import runner as R
        r = R.Runner.__new__(R.Runner)               # _env() reads only components_dir
        r.components_dir = None
        self.assertEqual(r._env()[NO_BACKTRACE], "1")
        r.components_dir = "/components"
        self.assertEqual(r._env()[NO_BACKTRACE], "1")

    def test_the_tuning_engine_environment(self):
        from strata_control import engine_component, tuning
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "engine-component.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(engine_component, "package_lib_dirs", return_value=[]):
                self.assertEqual(tuning.engine_env(Path(d) / "strata")[NO_BACKTRACE], "1")

    def test_the_tuning_workers(self):
        from serve import lifeline
        from strata_control import tune_api, tune_engine
        seen = []

        def fake_popen(*args, **kw):
            seen.append(kw.get("env"))
            raise OSError("not started in this test")

        with tempfile.TemporaryDirectory() as d, mock.patch.object(lifeline, "popen", fake_popen):
            exe = Path(d) / "strata"
            exe.write_text("", encoding="utf-8")
            for env in ({"PATH": "x"}, None):
                seen.clear()
                with self.assertRaises(tune_engine.MeasurementError):
                    tune_engine.EngineHarness(exe, {}, cwd=Path(d), env=env).start()
                with self.assertRaises(tune_engine.MeasurementError):
                    tune_api.ApiHarness(python=sys.executable, serve_dir=Path(d), config_path=Path(d) / "c.json", env=env).start()
                if env is None:
                    self.assertEqual(seen, [None, None], "no environment given: the child inherits this process's, set at the entry point")
                else:
                    self.assertEqual([e[NO_BACKTRACE] for e in seen], ["1", "1"])
                    self.assertEqual([e["PATH"] for e in seen], ["x", "x"])


class EntryPointsSetItForEveryChild(unittest.TestCase):
    """Programs started without an environment of their own inherit this process's: the entry points set it there."""

    def test_each_entry_point_calls_the_helper(self):
        for rel, anchor in (("strata_cli/cli.py", "def main("), ("serve/server.py", "def main("), ("strata_control/runtime.py", "def main("),
                            ("setup.py", "def main(")):
            text = (ROOT / rel).read_text(encoding="utf-8")
            body = text[text.index("\n" + anchor):]
            self.assertIn("quiet_crash_env()", body[:6000], rel)

    def test_the_server_and_the_tuning_session_environments_use_it(self):
        for rel, needle in (("strata_control/runtime.py", "quiet_crash_env(os.environ.copy())"),
                            ("strata_control/tune_session.py", "quiet_crash_env(dict(os.environ))")):
            self.assertIn(needle, (ROOT / rel).read_text(encoding="utf-8"), rel)


# Every place product code starts a process, with what it starts.  A file may hold at most this many; a new site fails the test until it is
# listed here (and, when it starts a Strata program - the engine, the picture reader, sd-cli, setup, a worker - it builds its environment with
# quiet_crash_env or inherits the one an entry point set).
STRATA_PROGRAMS = {
    "serve/server.py": 7,              # the engine, the picture reader (env from child_env / vision_env), the strata-device probe, before_load commands
    "serve/images.py": 1,              # sd-cli (image_env)
    "serve/lifeline.py": 4,            # the wrapper every child goes through (ps, a watcher); no program of ours of its own
    "serve/winjob.py": 2,              # lifeline.popen's Windows start inside the job (suspended, then resumed) + the plain Popen off Windows: the caller's env as given
    "strata_control/runtime.py": 4,    # the runtime host (inherits) and the server (quiet_crash_env on its environment); taskkill
    "strata_control/tune_api.py": 1,   # tuning worker (wrapped)
    "strata_control/tune_engine.py": 2,  # tuning engine (wrapped) + ps
    "strata_control/tune_session.py": 1,  # tuning worker (quiet_crash_env)
    "strata_control/tuning.py": 1,     # engine --help (engine_env)
    "strata_control/shape_engine.py": 2,  # engine --version / --geometry (inherit)
    "strata_control/dsv41_engine.py": 1,  # --version (inherits)
    "strata_control/dsv41_flags.py": 1,   # --help (inherits)
    "strata_control/dsv41_paths.py": 1,   # --version (inherits)
    "strata_control/updater.py": 3,       # --version probes (inherit)
    "strata_cli/bench_standard.py": 2,    # `strata.py serve` (inherits: cli.main sets it) + taskkill
    "strata_cli/launch/runner.py": 2,     # setup.py and the other children (_env)
    "strata_cli/launch/params.py": 2,     # engine --help (inherits)
    "strata_cli/cli.py": 2,               # setup.py, the revise command
    "setup.py": 17,                       # builds, engine probes, the helpers it runs (main sets it)
}
OTHER_PROGRAMS = {
    "serve/mcp.py": 2,                 # the person's MCP servers and taskkill
    "serve/telemetry.py": 2,           # sysctl, a Metal tool
    "strata_control/doctor.py": 1,
    "strata_control/engine_component.py": 4,   # nvidia-smi, powershell, lspci-style probes
    "strata_control/gpu_roles.py": 1,          # python -c
    "strata_control/procs.py": 3,              # ps, taskkill
    "strata_control/profiles.py": 1,           # the person's editor
    "strata_control/recipes.py": 1,            # the person's editor
    "strata_control/split_engine.py": 1,       # nvidia-smi
    "strata_control/time_skip.py": 3,          # the person's check commands
    "strata_control/tune_machine.py": 3,       # sysctl, vm_stat, a probe
    "strata_cli/bench_facts.py": 1,            # a tool for the facts
    "strata_cli/launch/access.py": 5,          # tunnel programs, powershell, ps
}

_SUBPROCESS = {"Popen", "run", "call", "check_output", "check_call"}
_OS = {"execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe", "spawnv", "spawnve", "spawnvp", "spawnl", "spawnle", "system", "startfile"}


def _is_spawn(node) -> bool:
    f = node.func
    if isinstance(f, ast.Attribute):
        name = f.value.id if isinstance(f.value, ast.Name) else ""
        return (name == "subprocess" and f.attr in _SUBPROCESS) or (name == "os" and f.attr in _OS) or (f.attr == "popen" and name in ("lifeline", "_lifeline"))
    return isinstance(f, ast.Name) and f.id in ("popen", "Popen", "_popen", "execv", "execve")


def spawn_sites() -> dict:
    files = []
    for d in ("serve", "strata_control", "strata_cli"):
        files += sorted((ROOT / d).rglob("*.py"))
    files += [ROOT / "strata.py", ROOT / "setup.py"]
    out = {}
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if p.name.startswith("test_") or "/tests/" in rel or not p.is_file():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        n = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call) and _is_spawn(node))
        if n:
            out[rel] = n
    return out


class EverySpawnSiteIsListed(unittest.TestCase):
    def test_no_process_is_started_that_nobody_has_looked_at(self):
        known = {**STRATA_PROGRAMS, **OTHER_PROGRAMS}
        new = {rel: n for rel, n in spawn_sites().items() if n > known.get(rel, 0)}
        self.assertEqual(new, {}, "a new place that starts a process: if it starts a Strata program (engine, picture reader, sd-cli, setup, a worker) "
                                  "its environment must carry GGML_NO_BACKTRACE (strata_control.childenv.quiet_crash_env); then list it in "
                                  "STRATA_PROGRAMS (or OTHER_PROGRAMS when it starts nothing of ours) in tests/terminal/test_no_debugger_children.py")

    def test_the_two_lists_do_not_overlap(self):
        self.assertEqual(set(STRATA_PROGRAMS) & set(OTHER_PROGRAMS), set())


if __name__ == "__main__":
    unittest.main()
