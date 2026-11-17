"""``strata split-engine apply``: the all-cards split engine for a run config the app wrote (the Terminal does the same inside ``strata launch``).

The desktop app builds its run config itself and starts it as a saved profile, so the step that turns a layer-split run config into the split engine's
(strata_control/split_engine.py ``apply_run_config``) is asked of Control here, with the rule, the engine package and the draft layer checked on this computer.
``--config`` is read and never changed; ``--out`` receives the split engine's run config only when the answer is yes. Nothing is downloaded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def add_parser(sp) -> None:
    p = sp.add_parser("split-engine", help="the all-cards split engine for a run config (the app's start asks it; the Terminal does the same in `strata launch`)")
    s_sp = p.add_subparsers(dest="split_engine_command", required=True)
    one = s_sp.add_parser("apply", help="write the split engine's run config for this one when the rule says yes; the config itself is not changed")
    one.add_argument("--config", required=True, metavar="FILE", help="the run config the app is about to start")
    one.add_argument("--out", metavar="FILE", help="where the split engine's run config goes when the answer is yes")
    one.add_argument("--output", choices=("human", "json"), default="human")


def run(a) -> int:
    machine = a.output == "json"

    def fail(message: str) -> int:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": "split_engine_failed", "message": message}}, ensure_ascii=False))
        else:
            print(f"strata: {message}", file=sys.stderr)
        return 5

    try:
        cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
        if not isinstance(cfg, dict):
            raise ValueError("not an object")
    except (OSError, ValueError) as exc:
        return fail(f"The run config could not be read ({type(exc).__name__}).")
    from strata_control import split_engine as SE
    work = json.loads(json.dumps(cfg))                       # a copy: apply_run_config rewrites it in place
    decision, line, warn = SE.apply_run_config(work, allow_untested=os.environ.get("STRATA_SPLIT_ALLOW_UNTESTED") == "1")
    wrote = None
    if decision.use_split and a.out:
        out = Path(a.out)
        tmp = out.with_name(out.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(work, f, indent=2)
            f.write("\n")
        os.replace(tmp, out)
        wrote = str(out)
    if machine:
        print(json.dumps({"schema_version": 1, "ok": True, "split": bool(decision.use_split), "line": line, "warn": bool(warn), "reasons": list(decision.reasons),
                          "cards": decision.cards, "written": wrote}, ensure_ascii=False))
    else:
        print(line or ("Using the normal layer split." if not decision.use_split else decision.line))
    return 0
