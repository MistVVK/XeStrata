"""The Python Control the app's host talks to, for the config snapshot: ``python control_shim.py drafting plan|fetch ...``.

The app's host runs ``strata drafting plan --config ...`` and ``strata drafting fetch --config ... --if-needed`` before a Load.  Here the same commands run through the
real strata_cli on the simulated machine (STRATA_SNAPSHOT_WORLD names the world file), with the download replaced by a recorder (tests/snapshot/world.py)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv: list[str]) -> int:
    from tests.snapshot.driver import run_cli
    from tests.snapshot.matrix import machine, model_set
    from tests.snapshot.world import World
    spec = json.loads(Path(os.environ["STRATA_SNAPSHOT_WORLD"]).read_text(encoding="utf-8"))
    world = World(machine(spec["machine"]), model_set(spec["set"]), Path(spec["root"]))
    with world.patched():
        code, out, err = run_cli(argv)
        log = Path(spec["root"]) / "shim-fetches.jsonl"
        with log.open("a", encoding="utf-8") as f:
            for item in world.fetches:
                f.write(json.dumps(item) + "\n")
    sys.stdout.write(out)
    sys.stderr.write(err)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
