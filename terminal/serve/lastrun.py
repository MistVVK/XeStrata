"""Say, at the next start, what happened to the last one (0.2.16 retest P4b, Linux).

When a launch is ended by force - systemd-oomd ends the whole terminal scope, the window is closed hard, the PC runs out of memory - nothing is left
to say so: the window and the engine just vanish (USERTEST-0216-LIN F1).  So each server leaves a small marker while it runs and removes it on every
normal end.  A marker that is still there at the next start, whose server is gone, means the last run did not end normally; the user journal (readable
without root) says whether it was systemd-oomd.  One plain paragraph is printed at the start; it never stops a start.

Linux only: another system's hard ends (a closed console, the app's job object) are ordinary there and would only alarm.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MARKER_PREFIX = "last-run-"
JOURNAL_WAIT_S = 5

OOMD_WORDS = ("[strata] Last time, this computer's memory guard (systemd-oomd) ended Strata: it stops a program when memory has "
              "been under strain for 20 seconds, and loading a large model can do that. Strata now frees memory before the load and pauses it for a few seconds when memory gets tight, "
              "If it happens again: close other programs (a web browser uses a lot of memory) and start again, "
              "or start Strata from a text console or over ssh, which the memory guard does not watch.")
UNCLEAN_WORDS = ("[strata] Last time, Strata did not end normally (the window was closed hard, the power went off, or the computer ran out of memory). "
                 "Your files are fine and this start goes on as usual. If it stops again while loading, close other programs and start again.")


def wants(argv=None) -> bool:
    """Only a server with the real engine leaves a marker (a mock-engine server is a test's)."""
    args = list(sys.argv[1:] if argv is None else argv)
    for i, a in enumerate(args):
        if a == "--engine" and i + 1 < len(args):
            return args[i + 1] == "strata"
        if a == "--engine=strata":
            return True
    return False


def marker_path(profile: str | None = None, state_base: Path | None = None) -> Path:
    if state_base is None:
        from strata_control import paths
        state_base = paths.state_base()
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (profile or "serve"))[:80]
    return Path(state_base) / f"{MARKER_PREFIX}{safe}.json"


def _token(pid: int) -> str | None:
    try:
        from strata_control import procs
        return procs.start_token(pid)
    except Exception:                                    # noqa: BLE001
        return None


def _gone(marker: dict) -> bool:
    """True when the server the marker names is not running any more (a pid that now belongs to another process counts as gone)."""
    pid = marker.get("pid")
    if type(pid) is not int or pid <= 0:
        return True
    token = _token(pid)
    return token is None or token != marker.get("token")


def oomd_killed_since(epoch: float, run=subprocess.run) -> bool:
    """Whether the user journal (our own, no root needed) has a "systemd-oomd killed" line since `epoch`.  False when it cannot be read."""
    try:
        done = run(["journalctl", "--user", "--no-pager", "-q", "-o", "cat", "--since", f"@{int(epoch) - 5}"],
                   capture_output=True, text=True, timeout=JOURNAL_WAIT_S, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and any("systemd-oomd killed" in line for line in (done.stdout or "").splitlines())


def previous_end_words(marker: dict | None, *, journal=oomd_killed_since) -> str:
    """The sentence(s) for a leftover marker of a server that is gone; "" when there is nothing to say."""
    if not isinstance(marker, dict) or not _gone(marker):
        return ""
    started = marker.get("started")
    if isinstance(started, (int, float)) and journal(float(started)):
        return OOMD_WORDS
    return UNCLEAN_WORDS


def begin(profile: str | None = None, *, state_base: Path | None = None, say=None, journal=oomd_killed_since) -> Path | None:
    """At a server's start: say what happened to the last run when its marker was left behind, then leave a marker of this run.  Returns the marker's
    path (for end()), or None when nothing is kept (not Linux, or a marker of a server that is still running).  Never raises."""
    try:
        if not sys.platform.startswith("linux") or os.environ.get("STRATA_LASTRUN") == "0":
            return None
        path = marker_path(profile, state_base)
        old = None
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = None
        if isinstance(old, dict) and not _gone(old):
            return None                                  # another server of this profile is running: its marker stays
        words = previous_end_words(old, journal=journal)
        if words:
            (say or (lambda text: print(text, flush=True)))(words)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"pid": os.getpid(), "token": _token(os.getpid()), "started": time.time(), "profile": profile}), encoding="utf-8")
        os.replace(tmp, path)
        return path
    except Exception:                                    # noqa: BLE001 - a nicety, never a reason for a start to fail
        return None


def end(path: Path | None) -> None:
    """At every normal end: remove this run's marker (only when it is still ours)."""
    if path is None:
        return
    try:
        marker = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(marker, dict) and marker.get("pid") == os.getpid():
            Path(path).unlink(missing_ok=True)
    except (OSError, ValueError):
        pass
