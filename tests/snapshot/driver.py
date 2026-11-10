"""Drive each entry point for one (machine class, model set) and return what the engine would be started with, normalized.

A record has the same shape for every entry point:

  entry            which one
  command          what the person (or the app) asked for, in words
  fetches          what the drafter fetch was asked to download: file name, variant, backend (nothing is downloaded)
  setup_args       what the entry point hands to setup.py (Terminal launch only)
  config_file      the run config the entry point leaves on disk (normalized)
  engine           {exe, argv, env_set, notes}: what serve/server.py would start: engine_args(cfg, exe) and the variables child_env(cfg) adds
  not_driven       (setup only) present instead of the above for a class whose setup is another script (Intel Arc, Mac), with the reason
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

from .matrix import ModelSet
from .normalize import normalize
from .world import ROOT, World, context_for

HERE = Path(__file__).resolve().parent


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    from strata_cli.cli import main
    out, err = io.StringIO(), io.StringIO()
    saved = dict(os.environ)            # main() sets variables on its own process for the programs it starts (GGML_NO_BACKTRACE, strata_control/childenv.py): that is
    try:                                # the command's doing, not the machine's, so it must not become the baseline the engine's environment is compared with
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as e:
                code = int(e.code or 0) if not isinstance(e.code, str) else 1
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return code, out.getvalue(), err.getvalue()

def engine_start(world: World, cfg: dict) -> dict:
    """What serve/server.py hands to the engine for this run config: the arguments (engine_args) and the variables child_env adds to the environment."""
    from serve import server
    baseline = dict(os.environ)
    notes = io.StringIO()
    with contextlib.redirect_stdout(notes):
        argv = server.engine_args(dict(cfg), cfg["exe"])
        env = server.child_env(dict(cfg))
    env_set = {k: v for k, v in sorted(env.items()) if baseline.get(k) != v}
    return {"exe": cfg["exe"], "argv": [str(a) for a in argv], "env_set": env_set,
            "notes": [l for l in notes.getvalue().splitlines() if l.strip()]}


def drive_launch(world: World) -> dict:
    """`strata launch --model <set> --yes --no-tui` through the real command, with the engine start, the model download and the tuning replaced."""
    from strata_cli.launch import catalog as C, runner as R
    seen: dict = {}
    cat = C.Catalog(root=world.install, hardware=world.hardware_report())

    def ensure_model(self):
        seen["setup_args"] = self.setup_args()
        return world.config_path

    with mock.patch.object(C, "Catalog", lambda *a, **k: cat), \
            mock.patch.object(R.Runner, "ensure_engine", lambda self: None), \
            mock.patch.object(R.Runner, "ensure_model", ensure_model), \
            mock.patch.object(R.Runner, "tune", lambda self: None), \
            mock.patch.object(R.Runner, "start", lambda self: seen.setdefault("started", True) and None), \
            mock.patch.object(R.Runner, "open_access", lambda self: None), \
            mock.patch.object(R.Runner, "check_serving", lambda self: None):
        command = ["launch", "--model", world.mset.model, "--yes", "--no-tui", "--context", str(context_for(world.machine, world.mset))]
        orig_run = R.Runner.run

        def run(self, **kw):
            seen["runner"] = self
            return orig_run(self, **kw)

        with mock.patch.object(R.Runner, "run", run):
            code, out, err = run_cli(command)
    runner = seen.get("runner")
    if code != 0 or runner is None or runner.config_path is None:
        return {"entry": "launch", "command": "strata " + " ".join(command), "failed": {"exit": code, "stderr": err[-400:], "stdout": out[-400:]}}
    cfg = json.loads(Path(runner.config_path).read_text(encoding="utf-8"))
    return {"entry": "launch", "command": "strata " + " ".join(command), "fetches": list(world.fetches),
            "setup_args": seen.get("setup_args"), "config_file": cfg, "engine": engine_start(world, cfg)}


def drive_app(world: World, which: str) -> dict:
    """The app's Load, in a Node process: the host's own createNativeControl with a fake Control for everything but the Drafting calls, which go through the
    real strata_control (python -m tests.snapshot.control_shim) on the same simulated machine.  Returns the run config the host wrote and the Python side's
    engine start for it."""
    node = os.environ.get("STRATA_SNAPSHOT_NODE") or "node"
    world_file = world.root / "world.json"
    world_file.write_text(json.dumps({"machine": world.machine.key, "set": world.mset.key, "root": str(world.root)}), encoding="utf-8")
    env = {**os.environ, "STRATA_SNAPSHOT_PYTHON": sys.executable, "STRATA_SNAPSHOT_REPO": str(ROOT)}
    proc = subprocess.run([node, str(HERE / "app_driver.mjs"), str(world_file), which], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=240)
    last = [l for l in proc.stdout.splitlines() if l.startswith("SNAPSHOT-RESULT ")]
    if proc.returncode != 0 or not last:
        return {"entry": "app_" + which, "failed": {"exit": proc.returncode, "stderr": proc.stderr[-1500:], "stdout": proc.stdout[-800:]}}
    got = json.loads(last[-1][len("SNAPSHOT-RESULT "):])
    cfg = got.get("config_file")
    out = {"entry": "app_" + which, "command": got.get("command"), "fetches": got.get("fetches"), "control_calls": got.get("control_calls"), "config_file": cfg}
    if cfg:
        with world.patched():
            out["engine"] = engine_start(world, cfg)
    return out


def drive(machine_key: str, set_key: str, entry: str, root: Path | None = None) -> dict:
    """One record, normalized.  A fresh scratch world every time."""
    from .matrix import machine as get_machine, model_set as get_set
    world = World(get_machine(machine_key), get_set(set_key), root).build()
    try:
        if entry == "setup":
            from .setup_driver import drive_setup
            rec = drive_setup(world)
        elif entry == "launch":
            with world.patched():
                rec = drive_launch(world)
        elif entry in ("app_plain", "app_models"):
            rec = drive_app(world, entry[4:])
        else:
            raise ValueError(entry)
        return normalize(rec, world)
    finally:
        world.close()
