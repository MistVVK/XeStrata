# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/vram_ledger.py - the VRAM a GPU's engines have reserved, shared between processes (the image engine and the
translation engine started by any server of the user's).

One file a GPU in the temp folder (beside the image jobs' turn, GpuTurn, under the same GPU key) with a lock of its own,
held only while an entry is read or written: an image job that holds its GPU's turn for minutes does not keep a
translation from reserving beside it.  An entry is an engine's memory on the GPU: its owner (an id, the pid and the
process's start time), MiB, what it is for and its state:

    reserved   asked for, not allocated yet          allocated  the engine holds it
    releasing  the engine is giving it back (counted as allocated until its entry is removed)

Every entry is charged to the ledger until it is removed, in whatever state (a report that has not caught up with an
allocation does not hand its memory to someone else).  The room a new entry can take is the smaller of

    the device's free memory now - every reserved entry - the device's margin
    the base free - every entry - the device's margin

where the base is the device's free memory read while the ledger was empty and had been for `settle_s` seconds after
its last removal (a report read right after a release can still miss the memory given back).  The first term takes in
what is allocated outside the ledger (the chat's engine), the second what is in the ledger but not yet in the report.
What an outsider gives back while entries are there is not used until the ledger is empty again.  An entry whose owner
is gone (no such pid, or one started at another time: a reused pid) is removed whenever the ledger is read.

A reservation can still fail to allocate (an outsider took the memory meanwhile): the caller cancels its entry, frees
what it got and plans again once with the room now (fewer layers on the GPU, or the CPU alone).
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path

RESERVED, ALLOCATED, RELEASING = "reserved", "allocated", "releasing"

# How far a device's free-memory report follows its allocations (records' 2026-11-09-translate/s0-margin, allocating and
# filling 64 MiB to 2 GiB, then freeing them): the RTX 4070's and the RX 9060 XT's to within 2 MiB both ways at once;
# the B70's an allocation within 70 MiB (1 GiB shown as 962 MiB), and a free not at all while the process lives for
# 64 and 256 MiB (the runtime keeps them), in part after 5 s for 1 GiB.  The margin takes that and a new engine's
# context and runtime (a few hundred MiB on each): 1/32 of the total, at least 256 MiB; the settling time 10 s.
SETTLE_S = 10.0


def device_margin_mib(total_mib: int) -> int:
    """The margin the ledger keeps below a device's report, from its total memory."""
    return max(256, int(total_mib) // 32)


def process_start(pid: int) -> float | None:
    """When process `pid` started (seconds since some fixed point, the same for every reading of one process), None
    when there is no such process."""
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except ImportError:
        pass
    except Exception:   # psutil's NoSuchProcess, AccessDenied
        return None
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            return float(stat.rsplit(")", 1)[1].split()[19])   # field 22: starttime, in clock ticks since boot
        except (OSError, IndexError, ValueError):
            return None
    return None


def _user() -> str:
    return str(os.getuid()) if hasattr(os, "getuid") else re.sub(r"\W", "", os.environ.get("USERNAME", "user"))


class Ledger:
    def __init__(self, key: str, margin_mib: int = 256, settle_s: float = SETTLE_S, folder: str | None = None):
        base = Path(folder or tempfile.gettempdir()) / f"xestrata-vram-{_user()}-{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}"
        self.path, self.lock_path = Path(str(base) + ".json"), Path(str(base) + ".lock")
        self.margin_mib = int(margin_mib)
        self.settle_s = float(settle_s)

    # ---------------------------------------------------------------- the file, under the short lock
    @contextlib.contextmanager
    def _locked(self):
        f = open(self.lock_path, "a+b")
        try:
            if sys.platform == "win32":
                import msvcrt
                while True:
                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(0.01)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            state = self._read()
            yield state
            self._write(state)
        finally:
            if sys.platform == "win32":
                with contextlib.suppress(OSError):
                    import msvcrt
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            f.close()

    def _read(self) -> dict:
        try:
            got = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(got, dict) and isinstance(got.get("entries"), list):
                return got
        except (OSError, ValueError):
            pass
        return {"entries": [], "base_mib": None, "last_removed": 0.0}

    def _write(self, state: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def _alive(e: dict) -> bool:
        start = process_start(int(e.get("pid", -1)))
        return start is not None and abs(start - float(e.get("start", -1.0))) < 1e-3

    def _purge(self, state: dict, now: float) -> None:
        kept = [e for e in state["entries"] if self._alive(e)]
        if len(kept) != len(state["entries"]):
            state["last_removed"] = now
        state["entries"] = kept

    def _room(self, state: dict, device_free_mib: int, now: float) -> int:
        entries = state["entries"]
        if not entries and (state.get("base_mib") is None or now - float(state.get("last_removed", 0.0)) >= self.settle_s):
            state["base_mib"] = int(device_free_mib)
        reserved = sum(int(e["mib"]) for e in entries if e["state"] == RESERVED)
        held = sum(int(e["mib"]) for e in entries)
        base = state.get("base_mib")
        room = int(device_free_mib) - reserved - self.margin_mib
        if base is not None:
            room = min(room, int(base) - held - self.margin_mib)
        return max(0, room)

    # ---------------------------------------------------------------- the calls
    def room(self, device_free_mib: int) -> int:
        """The MiB a new entry can take now, `device_free_mib` the device's report."""
        now = time.time()
        with self._locked() as state:
            self._purge(state, now)
            return self._room(state, device_free_mib, now)

    def reserve(self, mib: int, purpose: str, device_free_mib: int, pid: int | None = None) -> str | None:
        """An entry of `mib` reserved for process `pid` (this one by default), its id; None when the room is short."""
        now = time.time()
        pid = os.getpid() if pid is None else pid
        start = process_start(pid)
        if start is None:
            return None
        with self._locked() as state:
            self._purge(state, now)
            if int(mib) > self._room(state, device_free_mib, now):
                return None
            eid = uuid.uuid4().hex
            state["entries"].append({"id": eid, "pid": pid, "start": start, "mib": int(mib), "purpose": purpose,
                                     "state": RESERVED, "time": now})
            return eid

    def _set(self, eid: str, **changes) -> bool:
        now = time.time()
        with self._locked() as state:
            self._purge(state, now)
            for e in state["entries"]:
                if e["id"] == eid:
                    e.update(changes)
                    e["time"] = now
                    return True
        return False

    def allocated(self, eid: str, pid: int | None = None) -> bool:
        """The memory is held (by process `pid` from now on, the engine that holds it, when given)."""
        changes: dict = {"state": ALLOCATED}
        if pid is not None:
            start = process_start(pid)
            if start is None:
                return False
            changes.update(pid=pid, start=start)
        return self._set(eid, **changes)

    def releasing(self, eid: str) -> bool:
        """The memory is being given back: still charged until remove()."""
        return self._set(eid, state=RELEASING)

    def remove(self, eid: str) -> None:
        """The memory is given back (or was never allocated): the entry goes, its time noted for the base."""
        now = time.time()
        with self._locked() as state:
            before = len(state["entries"])
            state["entries"] = [e for e in state["entries"] if e["id"] != eid]
            if len(state["entries"]) != before:
                state["last_removed"] = now

    def entries(self) -> list[dict]:
        now = time.time()
        with self._locked() as state:
            self._purge(state, now)
            return [dict(e) for e in state["entries"]]
