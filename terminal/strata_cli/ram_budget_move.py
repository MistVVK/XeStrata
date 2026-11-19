"""Move the RAM budget setup wrote under the 0.2.16 rule (the RAM less 24 GB, 40 on 64 GB) to the 0.2.17 rule (the RAM less 17 GB, 47), once, without a setup run.

`strata update` installs a newer Control beside the install and never rewrites a run config, and the app's Load runs setup again only when the context, KV type or
pictures change, so an installed 0.2.16 model kept its 40 for ever (RESULTS-0217-RAM64).  Owner 11-19: it moves.  The first command a new Control runs (strata.py
calls this before anything heavy, the app host's first `strata models list --installed` among them) looks at every folder of run configs once and leaves a mark
there (`ram-budget-0217.done`).  The rule and every check of what is setup's and what is the person's are setup.migrate_ram_budget (one place); this module only
finds the folders, imports setup.py when a config has a budget and no "ui_budget_work_gib" yet, and writes the mark.  A config with nothing to move is not
written (byte-identical).  Every failure is swallowed: a command must never fail because of this.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

BUDGET = "--resident-budget-gib"
WORK_KEY = "ui_budget_work_gib"
MARK = "ram-budget-0217.done"


def _folders() -> list[Path]:
    from strata_control import models
    return [p for p, _src in models.all_installs()]


def _needs_setup(folder: Path) -> bool:
    """True when a config of the folder has a RAM budget and no second number: the only ones the move can change (stdlib only, no setup.py import)."""
    for path in folder.glob("strata-*.json"):
        if path.name.endswith(".shared-settings.json"):
            continue
        try:
            cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict) and isinstance(cfg.get("args"), list) and BUDGET in cfg["args"] and WORK_KEY not in cfg:
            return True
    return False


def _load_setup():
    path = Path(__file__).resolve().parents[1] / "setup.py"
    spec = importlib.util.spec_from_file_location("strata_setup_ram_budget", path)
    module = importlib.util.module_from_spec(spec)
    real = sys.stdout
    sys.stdout = sys.stderr
    try:
        spec.loader.exec_module(module)
    finally:
        sys.stdout = real
    return module


def run(folders=None, setup_module=None, say=None) -> dict:
    """{folder: [config names moved]} for the folders looked at now (a folder with its mark is not looked at again)."""
    say = say or (lambda text: sys.stderr.write(text + "\n"))
    done: dict = {}
    for folder in (_folders() if folders is None else [Path(f) for f in folders]):
        try:
            if not folder.is_dir() or (folder / MARK).exists():
                continue
            moved = []
            if _needs_setup(folder):
                moved = (setup_module or _load_setup()).migrate_run_configs(folder)
            (folder / MARK).write_text("looked at by 0.2.17: the RAM budget setup wrote under the old rule is moved to the new one once (strata_cli/ram_budget_move.py)\n", encoding="utf-8")
            done[str(folder)] = moved
            if moved:
                say("strata: the RAM budget of " + ", ".join(moved) + " moved to the 0.2.17 rule (the RAM less 17 GB); the earlier file is kept as <name>.bak")
        except Exception:                              # noqa: BLE001 - never a reason for the command to fail
            continue
    return done
