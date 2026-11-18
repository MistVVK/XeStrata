"""Heal an old Windows install whose PATH folder still holds a `strata.ps1` (0.2.14, 0.2.15 and earlier).

PowerShell runs a `strata.ps1` before a `strata.cmd` in the same folder, and the Windows default execution policy (Restricted) refuses a
script, so a typed `strata` failed with "running scripts is disabled on this system". From 0.2.16 the archive carries the PowerShell
launcher as `strata-direct.ps1` and `strata` can only mean `strata.cmd`. `strata update` installs a newer Control beside the install and
never touches the install's own `bin` folder, so an older install keeps its `strata.ps1`: the first command a new Control runs renames it.

Rules: Windows only; only a file whose first line is Strata's own header, in a folder that also holds Strata's `strata.cmd`; a rename, never
a delete of anything else; every failure is swallowed (a launcher must never fail because of this); one line goes to stderr when a file was
renamed, for support. Standard library only (strata.py runs this before any heavy import).
"""

from __future__ import annotations

import os
import sys
import time

HEADER = "# Strata Terminal launcher for PowerShell"
RENAMED_TO = "strata-direct.ps1"
LOG_NAME = "launcher-heal.log"                   # beside the renamed file: one line per rename


def installed_bin():
    """The one-line installer's launcher folder (<data>/terminal/current/bin), which the user PATH names: found from the components folder."""
    from strata_cli.handover import components_dir
    return components_dir().parent / "terminal" / "current" / "bin"


def stale_launchers(path_value: str | None = None, extra: list | None = None) -> list[str]:
    """The strata.ps1 files in PATH folders (and `extra` folders) that are Strata's own and sit beside a strata.cmd."""
    folders = [p.strip().strip('"') for p in (os.environ.get("PATH", "") if path_value is None else path_value).split(os.pathsep)]
    folders += [str(p) for p in (extra or [])]
    seen, found = set(), []
    for folder in folders:
        if not folder:
            continue
        key = os.path.normcase(os.path.abspath(folder))
        if key in seen:
            continue
        seen.add(key)
        ps1, cmd = os.path.join(folder, "strata.ps1"), os.path.join(folder, "strata.cmd")
        try:
            if not (os.path.isfile(ps1) and os.path.isfile(cmd)):
                continue
            with open(ps1, "r", encoding="utf-8-sig", errors="replace") as handle:
                first = handle.readline()
        except OSError:
            continue
        if first.startswith(HEADER):
            found.append(ps1)
    return found


def heal(path_value: str | None = None, extra: list | None = None, say=None, windows: bool | None = None) -> list[str]:
    """Rename each stale strata.ps1 to strata-direct.ps1 (replacing an older strata-direct.ps1). Returns the renamed paths."""
    if not (os.name == "nt" if windows is None else windows):
        return []
    say = say or (lambda text: sys.stderr.write(text + "\n"))
    renamed = []
    try:
        for ps1 in stale_launchers(path_value, extra):
            target = os.path.join(os.path.dirname(ps1), RENAMED_TO)
            try:
                os.replace(ps1, target)
            except OSError:
                continue
            renamed.append(ps1)
            try:                                         # for support: when a update's own test start ran this, nobody saw the line below
                with open(os.path.join(os.path.dirname(ps1), LOG_NAME), "a", encoding="utf-8") as log:
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} strata.ps1 was renamed to {RENAMED_TO} (a typed `strata` in PowerShell "
                              "runs a .ps1 before a .cmd, and the default execution policy refuses it)\n")
            except OSError:
                pass
            say(f"strata: {ps1} was renamed to {RENAMED_TO}. PowerShell runs a .ps1 file before a .cmd file and refuses it with its "
                "default settings; now typing  strata  works there.")
    except Exception:                                    # noqa: BLE001 - never a reason for a launcher to fail
        return renamed
    return renamed
