"""serve/commit_limit.py - the engine's Windows commit-limit warning (#1607), as a fact the Monitor and the app can read.

On Windows the engine compares what a run plans to allocate on the host with the commit that is left (RAM + page file) and, when it will not fit, writes one line
to its log: "strata serve: WARNING the Windows commit limit is 92.8 GiB (RAM + page file) and 0.8 GiB of it is left, but this run plans 2.0 GiB of host
allocations and keeps 4.0 GiB free: it can end at the limit with no message (exit code 0xC0000409, #1607) while RAM is still free. ..." The engine has no
structured field for it (0.2.17: give it one), so the server reads the line from this start's log and puts the numbers in engine.info["commit_warning"], which /metrics
carries in `engine` like the other INFO facts. The pattern is pinned to the engine's own text by the unit test (serve/test_commit_limit.py).
"""
from __future__ import annotations

import re

_LINE = re.compile(
    r"WARNING the Windows commit limit is (?P<limit>[0-9]+(?:\.[0-9]+)?) GiB \(RAM \+ page file\) and (?P<left>[0-9]+(?:\.[0-9]+)?) GiB of it is left, "
    r"but this run plans (?P<planned>[0-9]+(?:\.[0-9]+)?) GiB of host allocations and keeps (?P<floor>[0-9]+(?:\.[0-9]+)?) GiB free")


def commit_warning(log_text: str) -> dict | None:
    """The numbers of the engine's commit-limit warning in this start's log text (the last such line), or None when there is none:
    {"limit_gib", "left_gib", "planned_gib", "floor_gib"}."""
    found = None
    for line in (log_text or "").splitlines():
        m = _LINE.search(line)
        if m:
            found = m
    if found is None:
        return None
    return {"limit_gib": float(found.group("limit")), "left_gib": float(found.group("left")),
            "planned_gib": float(found.group("planned")), "floor_gib": float(found.group("floor"))}
