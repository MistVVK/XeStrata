# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/vram_ledger.py - the VRAM a GPU's engines hold or have reserved, shared between processes (the image engine,
the translation engine, the text engine's VRAM it gives and takes back, the vision reader; any server of the user's).

One file a GPU in the temp folder (beside the image jobs' turn, GpuTurn, under the same GPU key) with a lock of its own,
held only while an entry is read or written: an image job that holds its GPU's turn for minutes does not keep a
translation from reserving beside it.  An entry is an engine's memory on the GPU: its owner (the pid and the process's
start time of what holds the memory: the image engine's, not the server's, so it outlives the server that asked for it
and the next server finds it), MiB, what it is for and its state:

    reserved   asked for, not allocated yet          allocated  the engine holds it
    releasing  the engine is giving it back (counted as allocated until its entry is removed)

and two more fields: `exact`, whether its MiB is what the holder says it holds (the image engine's FREE, the text
engine's VRAM answer) rather than a budget or an estimate; and `base`, how much of it the ledger's base already holds.
Every entry is charged to the ledger until it is removed, in whatever state (a report that has not caught up with an
allocation does not hand its memory to someone else), less what of it the base holds.  The room a new entry can take
is the smaller of

    the device's free memory now - every reserved entry - the device's margin
    the base free - sum(each entry's MiB - its base) - the device's margin

where the base is the device's free memory read while every entry was exact and allocated and none had changed for
`settle_s` seconds (nor any been removed: a report read right after a release can still miss the memory given back);
each entry's base is then what it held.  An engine that held its memory before the ledger knew of it (the text engine,
started before the server; an image engine the ledger lost) is entered as in the base (hold(in_base=True)): the base is
read again at the next report, with it.  What such an engine gives back later (the text engine's expert cache shrunk
for an image, its helper card handed over) is then room, what it takes back charged: only the change is counted, not
what the base already shows.  When an entry goes, what the base held of it is free again: added back where the entry
was all its process holds (the image engine's), else (the text engine ended or unloaded: more than its cache) not
known, and the base read again only once the report has caught up with the release (`settle_s` after it): until then
the room counts no part of what went as free.  The first term takes in what is allocated outside the ledger,
the second what is in the ledger but not yet in the report.  An entry whose owner is gone (no such pid, or one started
at another time: a reused pid) is removed whenever the ledger is read.

A reservation can still fail to allocate (an outsider took the memory meanwhile): the caller cancels its entry, frees
what it got and plans again once with the room now (fewer layers on the GPU, or the CPU alone).

A caller that cannot read the device's report (the server for the translation engine, which reads its own) passes
None for it: the room is then the second term alone, or None (no limit known) while no base is known, and the base is
left as it is.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import stat
import sys
import tempfile
import time
import uuid
from pathlib import Path

from serve import file_lock

RESERVED, ALLOCATED, RELEASING = "reserved", "allocated", "releasing"

# How far a device's free-memory report follows its allocations (records' 2026-11-09-translate/s0-margin, allocating and
# filling 64 MiB to 2 GiB, then freeing them): the RTX 4070's and the RX 9060 XT's to within 2 MiB both ways at once;
# the B70's an allocation within 70 MiB (1 GiB shown as 962 MiB), and a free not at all while the process lives for
# 64 and 256 MiB (the runtime keeps them), in part after 5 s for 1 GiB.  The margin also covers the runtime: the
# core::vram_room formula with a 256 MiB floor and a 768 MiB ceiling; the settling time is 10 s.
SETTLE_S = 10.0


def device_margin_mib(total_mib: int) -> int:
    """The margin the ledger keeps below a device's report, from its total memory."""
    return min(768, max(256, int(total_mib) // 32))


def process_start(pid: int) -> float | None:
    """When process `pid` started (seconds since some fixed point, the same for every reading of one process), None
    when there is no such process."""
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            return float(stat.rsplit(")", 1)[1].split()[19])   # field 22: clock ticks since boot
        except (OSError, IndexError, ValueError):
            return None
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception:   # ImportError, NoSuchProcess, AccessDenied
        return None


def drm_process_memory_mib(pid: int, pci: str, proc: Path = Path("/proc")) -> int | None:
    """The process's DRM VRAM on one PCI device, with each DRM client counted once (gpu.cpp's rule)."""
    try:
        files = list((proc / str(pid) / "fdinfo").iterdir())
    except OSError:
        return None
    clients: dict[str, int] = {}
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        driver = device = client = ""
        xe = amd = 0
        found_xe = found_amd = False
        for line in lines:
            key, sep, value = line.partition(":")
            if not sep:
                continue
            fields = value.split()
            if key == "drm-driver" and fields:
                driver = fields[0]
            elif key == "drm-pdev" and fields:
                device = fields[0]
            elif key == "drm-client-id" and fields:
                client = fields[0]
            elif key.startswith("drm-total-vram") or key == "drm-memory-vram":
                if not fields or len(fields) > 2 or (len(fields) == 2 and fields[1] not in ("B", "KiB", "MiB")):
                    continue
                try:
                    value_bytes = int(fields[0]) * ({"B": 1, "KiB": 1024, "MiB": 1048576}.get(
                        fields[1], 1) if len(fields) == 2 else 1)
                except ValueError:
                    continue
                if value_bytes < 0:
                    continue
                if key.startswith("drm-total-vram"):
                    xe += value_bytes
                    found_xe = True
                else:
                    amd += value_bytes
                    found_amd = True
        if device.lower() != pci.lower() or (driver == "xe" and not found_xe) or \
                (driver == "amdgpu" and not found_amd) or driver not in ("xe", "amdgpu"):
            continue
        name = client or f.name
        clients[name] = max(clients.get(name, 0), xe if driver == "xe" else amd)
    return sum(clients.values()) >> 20 if clients else None


def process_memory_mib(pid: int, pci: str, source: str | None) -> tuple[int, str] | None:
    """Read a process on this card now, or None when its source cannot identify the process."""
    if not sys.platform.startswith("linux"):
        return None
    if source in (None, "fdinfo"):
        used = drm_process_memory_mib(pid, pci)
        if used is not None:
            return used, "fdinfo"
    if source in (None, "nvml"):
        from serve import telemetry
        used = telemetry.nvml_process_memory_mib(pid, pci)
        if used is not None:
            return used, "nvml"
    return None


def _user() -> str:
    return str(os.getuid()) if hasattr(os, "getuid") else re.sub(r"\W", "", os.environ.get("USERNAME", "user"))


class Ledger:
    def __init__(self, key: str, margin_mib: int | None = None, settle_s: float = SETTLE_S,
                 folder: str | None = None, total_mib: int = 0, engine_total: bool = False):
        # The system's temp folder is writable by everyone.  Work in a directory only this user can enter: another
        # user cannot plant its .lock, its .json or a link at its .tmp, and lstat below refuses a planted directory.
        root = Path(folder or tempfile.gettempdir()) / f"xestrata-vram-{_user()}"
        root.mkdir(mode=0o700, exist_ok=True)
        own = root.lstat()                          # not is_dir(): a link to someone else's folder is not ours
        if not stat.S_ISDIR(own.st_mode) or (hasattr(os, "getuid") and own.st_uid != os.getuid()):
            raise PermissionError(f"{root} is not this user's directory")
        base = root / re.sub(r'[^A-Za-z0-9_.-]', '_', key)
        self.path, self.lock_path = Path(str(base) + ".json"), Path(str(base) + ".lock")
        self.margin_mib = margin_mib
        self.total_mib = int(total_mib)
        self.engine_total = engine_total
        self.settle_s = float(settle_s)

    # ---------------------------------------------------------------- the file, under the short lock
    @staticmethod
    def _owned(fd: int) -> bool:
        st = os.fstat(fd)
        return stat.S_ISREG(st.st_mode) and (not hasattr(os, "getuid") or st.st_uid == os.getuid())

    @contextlib.contextmanager
    def _locked(self):
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not self._owned(fd):
                raise PermissionError(f"{self.lock_path} is not this user's file")
            f = os.fdopen(fd, "a+b")            # fdopen owns fd from here; no double close on the error paths above
        except BaseException:
            os.close(fd)
            raise
        try:
            if os.name == "nt":
                while True:
                    try:
                        file_lock.acquire(f)
                        break
                    except OSError:
                        time.sleep(0.01)
            else:
                file_lock.acquire(f, blocking=True)
            state = self._read()
            # Telemetry may lead the first image engine; its total then replaces telemetry's and stays authoritative.
            if self.total_mib > 0 and (self.engine_total or state.get("total_source") != "engine"):
                state["total_mib"] = self.total_mib
                state["total_source"] = "engine" if self.engine_total else "telemetry"
            yield state
            self._write(state)
        finally:
            with contextlib.suppress(OSError):
                file_lock.release(f)
            f.close()

    def _read(self) -> dict:
        empty: dict = {"entries": [], "base_mib": None, "last_removed": 0.0}
        try:
            st = self.path.lstat()
            if not stat.S_ISREG(st.st_mode) or (hasattr(os, "getuid") and st.st_uid != os.getuid()):
                return empty
            got = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(got, dict) and isinstance(got.get("entries"), list):
                return got
        except (OSError, ValueError):
            pass
        return empty

    def _write(self, state: dict) -> None:
        # A unique name with O_EXCL: one server cannot overwrite another's half-written temporary file, and a link
        # planted at a fixed name is never followed.  The parent is only writable by this user.
        tmp = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(state))
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

    @staticmethod
    def _alive(e: dict) -> bool:
        start = process_start(int(e.get("pid", -1)))
        return start is not None and abs(start - float(e.get("start", -1.0))) < 1e-3

    def _drop(self, state: dict, gone: list, now: float) -> None:
        """`gone` left the entries: the time noted for the base.  What the base held of one is free again: added to
        the base where the entry was all its process holds (`whole`: the image engine's), else an amount the ledger
        does not know (the text engine's cache is part of the engine): the base goes, and is read again only once the
        report has caught up with the release (`base_ready`), since a report read right after it still shows the
        memory as held."""
        if gone:
            state["last_removed"] = now
        for e in gone:
            if int(e.get("base", 0)) > 0 and state.get("base_mib") is not None:
                if e.get("whole"):
                    state["base_mib"] = int(state["base_mib"]) + int(e["base"])
                else:
                    state["base_mib"] = None
                    state["base_ready"] = now + self.settle_s

    def _purge(self, state: dict, now: float) -> None:
        kept = [e for e in state["entries"] if self._alive(e)]
        self._drop(state, [e for e in state["entries"] if e not in kept], now)
        state["entries"] = kept

    def _room(self, state: dict, device_free_mib: int | None, now: float) -> int | None:
        entries = state["entries"]
        # a base read from a report that has not caught up with a release would keep its too small a value until the
        # next settling: until `base_ready` no base is read at all, and the report's term alone bounds the room
        ready = now >= float(state.get("base_ready", 0.0))
        if device_free_mib is not None and ready and (state.get("base_mib") is None or (
                now - float(state.get("last_removed", 0.0)) >= self.settle_s
                and all(e.get("exact") and e["state"] == ALLOCATED and now - float(e.get("time", 0.0)) >= self.settle_s
                        for e in entries))):
            state["base_mib"] = int(device_free_mib)
            for e in entries:
                e["base"] = int(e["mib"]) if e.get("exact") and e["state"] == ALLOCATED else 0
        reserved = sum(int(e["mib"]) for e in entries if e["state"] == RESERVED)
        held = sum(int(e["mib"]) - int(e.get("base", 0)) for e in entries)
        base = state.get("base_mib")
        margin = self.margin_mib if self.margin_mib is not None else device_margin_mib(state.get("total_mib") or 0)
        terms = []
        if device_free_mib is not None:
            terms.append(int(device_free_mib) - reserved - margin)
        if base is not None:
            terms.append(int(base) - held - margin)
        return max(0, min(terms)) if terms else None

    # ---------------------------------------------------------------- the calls
    def room(self, device_free_mib: int | None) -> int | None:
        """The MiB a new entry can take now, `device_free_mib` the device's report (None: not known; the room is then
        None while no base is known)."""
        now = time.time()
        with self._locked() as state:
            self._purge(state, now)
            return self._room(state, device_free_mib, now)

    def reserve(self, mib: int, purpose: str, device_free_mib: int | None, pid: int | None = None,
                until: float | None = None) -> str | None:
        """An entry of `mib` reserved for process `pid` (this one by default), its id; None when the room is short.
        `until`: when its owner expects to give it back (time.time(); an image job's estimate), for the others to
        weigh waiting against making do with less."""
        now = time.time()
        pid = os.getpid() if pid is None else pid
        start = process_start(pid)
        if start is None:
            return None
        with self._locked() as state:
            self._purge(state, now)
            room = self._room(state, device_free_mib, now)
            if room is not None and int(mib) > room:
                return None
            eid = uuid.uuid4().hex
            state["entries"].append({"id": eid, "pid": pid, "start": start, "mib": int(mib), "purpose": purpose,
                                     "state": RESERVED, "time": now, "until": until, "exact": False, "base": 0})
            return eid

    def hold(self, purpose: str, mib: int, pid: int, exact: bool = True, in_base: bool = False,
             until: float | None = None, whole: bool = False) -> str | None:
        """Process `pid`'s entry for `purpose` (the one entry a holder has for it: found again by another server)
        set to `mib`, allocated, made when there is none -> its id; None when `pid` is gone.  `in_base`: a new
        entry's memory was held before the ledger knew of it, so the device's report already shows it: the base is
        read again at the next report.  An entry already there only changes by its MiB: what it gives back is room,
        what it takes charged.  `whole`: its MiB are all `pid` holds on the device (see _drop)."""
        now = time.time()
        start = process_start(pid)
        if start is None:
            return None
        with self._locked() as state:
            self._purge(state, now)
            for e in state["entries"]:
                if e.get("purpose") == purpose and int(e["pid"]) == pid:
                    changes = {"mib": int(mib), "state": ALLOCATED, "exact": bool(exact), "until": until,
                               "whole": bool(whole)}
                    if any(e.get(k) != v for k, v in changes.items()):   # the same again is no change for the base
                        e.update(changes, time=now)
                    return str(e["id"])
            eid = uuid.uuid4().hex
            # a new entry of what is held already (exact) changes nothing on the device: the base can be read with it
            state["entries"].append({"id": eid, "pid": pid, "start": start, "mib": int(mib), "purpose": purpose,
                                     "state": ALLOCATED, "time": 0.0 if exact else now, "until": until,
                                     "exact": bool(exact), "base": int(mib) if in_base else 0,
                                     "whole": bool(whole)})
            if in_base:
                state["base_mib"], state["base_ready"] = None, 0.0   # nothing was just released: read it again at once
            return eid

    def find(self, purpose: str, pid: int) -> str | None:
        """Process `pid`'s entry for `purpose`, its id; None when there is none."""
        for e in self.entries():
            if e.get("purpose") == purpose and int(e["pid"]) == pid:
                return str(e["id"])
        return None

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
            self._drop(state, [e for e in state["entries"] if e["id"] == eid], now)
            state["entries"] = [e for e in state["entries"] if e["id"] != eid]

    def entries(self) -> list[dict]:
        now = time.time()
        with self._locked() as state:
            self._purge(state, now)
            return [dict(e) for e in state["entries"]]
