# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/disk_watch.py - warn when a volume serve writes to is filling up.

The paths come from the run config and the places of serve (see DESIGN.md in the data folder's records): the config's
own folder, its log, conversation_save, the model and image folders, the image tools' and LoRA folders, the image
output folder, the state folder and the stdout log.  Paths on one filesystem are sampled once.  A loop file's own
free space is not the host's, so a loop volume is reported as unmeasured; the bind-mounted data volume is the one
whose free space counts.  tmpfs is marked "memory", not "disk".

One Watcher lives in the serve process, another in the image daemon; both use idle_watch's scheduler.  The result is
also kept in a small file in this user's temp folder, so a server that starts after the other process warned sees the
last level.  Nothing here talks to an engine: an engine that crashed cannot report a full disk.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from serve import file_lock, idle_watch, image_aux, places

G = 1 << 30
SAMPLE_S = 15.0
FIRST_REPEAT_S = 60.0
MAX_REPEAT_S = 300.0
HYSTERESIS_MIN = 5 * G
HYSTERESIS_FRAC = 0.02
CHANGE_G = 2 * G

OK, WARN, DANGER, UNMEASURED = "ok", "warn", "danger", "unmeasured"
LEVELS = (OK, WARN, DANGER)

# Patched in the unit tests; no test reads or writes a real data volume.
disk_usage = shutil.disk_usage

_watcher: Watcher | None = None
_EMPTY = {"at": 0, "event_at": 0, "level": OK, "measured": False, "free_gib": None, "total_gib": None,
          "warn_gib": None, "danger_gib": None, "volume": None, "kind": None, "message": None}


def thresholds(total: int) -> tuple[int, int]:
    """The free bytes that start the warning and the danger level: absolute floors and scaled ceilings."""
    warn = min(int(0.25 * total), max(20 * G, int(0.10 * total)))
    danger = min(int(0.10 * total), max(8 * G, int(0.03 * total)))
    return warn, danger


def severity(line: str) -> str:
    """The word before a console line, matching the server's existing [xestrata] WARNING/ERROR/INFO lines."""
    if line.startswith("disk is critically full"):
        return "ERROR"
    if line.startswith(("disk space recovered", "disk is no longer critically full", "disk:")):
        return "INFO"
    return "WARNING"


def _user() -> str:
    return str(os.getuid()) if hasattr(os, "getuid") else re.sub(r"\W", "", os.environ.get("USERNAME", "user"))


def state_paths() -> tuple[Path, Path]:
    """The shared state and its lock, in a directory only this user can enter."""
    root = Path(tempfile.gettempdir()) / f"xestrata-disk-{_user()}"
    root.mkdir(mode=0o700, exist_ok=True)
    own = root.lstat()                          # not is_dir(): a planted link is not ours
    if not stat.S_ISDIR(own.st_mode) or (hasattr(os, "getuid") and own.st_uid != os.getuid()):
        raise PermissionError(f"{root} is not this user's directory")
    return root / "state.json", root / "state.lock"


def _mount_kind(path: Path) -> str:
    """disk, loop or memory (tmpfs); unknown when the platform says nothing."""
    if sys.platform == "win32":
        return "unknown"
    try:
        text = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unknown"
    target = str(path)
    best, kind = -1, "unknown"
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        source, mount, fstype = fields[:3]
        mount = mount.replace("\\040", " ").replace("\\011", "\t")
        if not (target == mount or target.startswith(mount.rstrip("/") + "/")) or len(mount) <= best:
            continue
        best = len(mount)
        kind = "memory" if fstype == "tmpfs" else "loop" if source.startswith("/dev/loop") else "disk"
    return kind


def _dir(raw) -> Path | None:
    """The directory to ask about for a path that may be a file or may not exist yet."""
    if not isinstance(raw, (str, Path)) or not str(raw):
        return None
    p = Path(raw)
    if p.is_dir():
        return p
    if p.parent.is_dir():
        return p.parent
    return None


def _add(out: list[tuple[str, Path]], label: str, raw) -> None:
    p = _dir(raw)
    if p is not None:
        out.append((label, p))


def _in_config(cfg: dict, key: str, base: Path) -> Path | None:
    raw = cfg.get(key)
    if not isinstance(raw, str) or not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else base / p


def image_paths(cfg: dict, log_path=None) -> list[tuple[str, Path]]:
    """The folders one image model writes to: its output, its weights, tool weights, LoRAs and engine log."""
    out: list[tuple[str, Path]] = []
    if not isinstance(cfg, dict):
        return out
    keys = ("out_dir", "aux_dir", "lora_dir", "diffusion_model", "vae", "llm", "vision", "tokenizer")
    if not any(cfg.get(key) for key in keys) and log_path is None:
        return out                              # no images configured: do not claim the temp folder is theirs
    base = Path(cfg.get("cwd") or ".").resolve()
    _add(out, "image output", _in_config(cfg, "out_dir", base) or Path(tempfile.gettempdir()))
    for key in ("diffusion_model", "vae", "llm", "vision", "tokenizer"):
        _add(out, "image weights", _in_config(cfg, key, base))
    _add(out, "image tools", _in_config(cfg, "aux_dir", base) or image_aux.aux_dir(cfg))
    _add(out, "LoRAs", _in_config(cfg, "lora_dir", base))
    _add(out, "image engine log", log_path)
    return out


def process_paths(config_path=None) -> list[tuple[str, Path]]:
    """The folders that do not come from the image config."""
    out: list[tuple[str, Path]] = []
    if config_path:
        _add(out, "config", Path(config_path).parent)
    _add(out, "state", places.user_dir("state"))
    _add(out, "settings", places.settings_path())
    _add(out, "temp", Path(tempfile.gettempdir()))
    if sys.platform.startswith("linux"):
        try:                                    # the shell redirect of this process' stdout, when it is a file
            fd = os.readlink("/proc/self/fd/1")
            if os.path.isfile(fd):              # a real redirect, not a pipe or /dev/null
                _add(out, "console log", fd)
        except OSError:
            pass
    return out


def paths(cfg: dict | None, config_path=None) -> list[tuple[str, Path]]:
    """Every folder the server may write to, without creating any of them."""
    cfg = cfg if isinstance(cfg, dict) else {}
    base = Path(cfg.get("cwd") or ".").resolve()
    out = process_paths(config_path)
    for key in ("log", "conversation_save"):
        _add(out, key, _in_config(cfg, key, base))
    for arg in cfg.get("args") or ():
        if isinstance(arg, str) and arg.startswith("/"):
            _add(out, "model data", arg)
    for key in ("lora_dir", "data_dir", "models_dir"):
        _add(out, key, _in_config(cfg, key, base))
    _add(out, "LoRAs", _in_config(cfg, "lora_dir", base))
    for section in [cfg["images"], *(cfg.get("image_models") or ())] if isinstance(cfg.get("images"), dict) else ():
        if isinstance(section, dict):
            out += image_paths(section)
    # one path per location, keeping the first label
    seen: set[str] = set()
    unique = []
    for label, p in out:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append((label, p))
    return unique


def _device(path: Path):
    return os.stat(path).st_dev


def collect(paths: list[tuple[str, Path]]) -> list[dict]:
    """One reading per volume; the labels share it when they are on the same filesystem."""
    groups: dict[object, dict] = {}
    for label, raw in paths:
        p = _dir(raw)
        if p is None:
            continue
        try:
            dev = _device(p)
            usage = disk_usage(p)
        except OSError:
            continue
        row = groups.get(dev)
        if row is None:
            groups[dev] = {"volume": label, "path": p, "total": int(usage.total), "free": int(usage.free),
                           "kind": _mount_kind(p)}
        elif label not in row["labels"]:
            row["labels"].append(label)
    for row in groups.values():
        row.setdefault("labels", [])
    return list(groups.values())


def _snapshot(volumes: list[dict], clock, previous: str | None = None) -> dict:
    measured = [v for v in volumes if v["kind"] in ("disk", "memory")]
    if not measured:
        return {**_EMPTY, "at": int(clock.time()), "level": UNMEASURED, "kind": "loop" if volumes else None,
                "volume": "loop file" if volumes else None}
    primary = next((v for v in measured if v["kind"] == "disk"), measured[0])
    total, free = primary["total"], primary["free"]
    warn, danger = thresholds(total)
    his = max(HYSTERESIS_MIN, int(HYSTERESIS_FRAC * total))
    # leaving danger needs more than the danger line; warn has its own band
    if previous == DANGER:
        level = DANGER if free < danger + his else WARN if free < warn else OK
    elif previous == WARN:
        level = DANGER if free < danger else WARN if free < warn + his else OK
    else:
        level = DANGER if free < danger else WARN if free < warn else OK
    return {"at": int(clock.time()), "level": level, "measured": True, "free": free, "total": total,
            "warn": warn, "danger": danger, "his": his, "free_gib": round(free / G, 1),
            "total_gib": round(total / G, 1), "warn_gib": round(warn / G, 1),
            "danger_gib": round(danger / G, 1), "volume": primary["volume"], "kind": primary["kind"],
            "message": None}


def _line(kind: str, s: dict, repeat: int = 0) -> str:
    where = f"{s['volume']} ({s['total_gib']:.1f} GiB)"
    count = f" (repeat {repeat})" if repeat else ""
    if kind == DANGER:
        return (f"disk is critically full: {where} has {s['free_gib']:.1f} GiB free "
                f"(danger at {s['danger_gib']:.1f} GiB). Stop downloads and image jobs, or delete files; "
                f"the text engine can fail to start.{count}")
    if kind == WARN:
        return (f"disk is nearly full: {where} has {s['free_gib']:.1f} GiB free "
                f"(warn at {s['warn_gib']:.1f} GiB). Free space before the next download.{count}")
    if kind == "out":
        tail = " (still under the warning level)" if s["level"] == WARN else ""
        return f"disk is no longer critically full: {where} has {s['free_gib']:.1f} GiB free{tail}"
    return f"disk space recovered: {where} has {s['free_gib']:.1f} GiB free"


class Watcher:
    """One process' state machine; `check` is the action idle_watch repeats."""

    def __init__(self, paths: list[tuple[str, Path]], say: Callable[[str], None] | None = None,
                 clock=None, state_path: Path | None = None, save: Callable[[dict], None] | None = None):
        self.paths = list(paths)
        self.say = say
        self.clock = clock if clock is not None else time
        self.state_path = state_path
        self.save = save if save is not None else self._write
        self.level: str | None = None
        self.free: int | None = None
        self.next_repeat = 0.0
        self.repeat_delay = FIRST_REPEAT_S
        self.repeats = 0
        self.said_loop = False
        self.view_data: dict = dict(_EMPTY)

    def _emit(self, text: str) -> None:
        if self.say is not None:
            try:
                self.say(text)
            except Exception:  # noqa: BLE001 - a notification must never take the server down
                pass

    def _persist(self, snap: dict) -> None:
        try:
            self.save(snap)
        except Exception:  # noqa: BLE001 - the console line is the notice; the state file is only a copy
            pass

    def _write(self, snap: dict) -> None:
        try:
            if self.state_path is None:
                state, lock = state_paths()
            else:
                state, lock = self.state_path, self.state_path.with_suffix(".lock")
            with open(lock, "a+b") as f:
                file_lock.acquire(f)
                try:
                    tmp = state.with_name(state.name + ".tmp")
                    tmp.write_text(json.dumps(snap), encoding="utf-8")
                    os.replace(tmp, state)
                finally:
                    file_lock.release(f)
        except Exception:  # noqa: BLE001 - a full state folder or a foreign lock must not stop the watch
            pass                                # the console line still went out

    def check(self) -> bool:
        """Sample once.  Never raises: idle_watch drops a repeating action after three failures."""
        try:
            volumes = collect(self.paths)
            snap = _snapshot(volumes, self.clock, self.level)
            if any(v["kind"] == "loop" for v in volumes) and not self.said_loop:
                self.said_loop = True
                self._emit("disk: this root is a loop file; its own free space is not the host's")
        except Exception:  # noqa: BLE001 - show no number when a reading is not trustworthy
            snap = {**_EMPTY, "at": int(self.clock.time()), "level": UNMEASURED}
        level, free = snap["level"], snap.get("free")
        events: list[str] = []
        if self.level is None:
            if level in (WARN, DANGER):         # a healthy first reading says nothing
                events.append(level)
        elif level != self.level:
            if level == DANGER or (self.level != DANGER and level == WARN):
                events.append(level)
            elif self.level == DANGER and level in LEVELS:
                events.append("out")
            if level == OK and self.level in (WARN, DANGER):
                events.append("recovered")
        now = self.clock.monotonic()
        if free is not None and (self.free is None or abs(free - self.free) >= CHANGE_G):
            self.next_repeat = now + FIRST_REPEAT_S if level == DANGER else 0.0
            self.repeat_delay, self.repeats = FIRST_REPEAT_S, 0
        elif level == DANGER and self.level == DANGER and self.next_repeat and now >= self.next_repeat:
            self.repeats += 1
            events.append("repeat")
            self.repeat_delay = min(MAX_REPEAT_S, self.repeat_delay * 2)
            self.next_repeat = now + self.repeat_delay
        if level == DANGER and not self.next_repeat:
            self.next_repeat = now + FIRST_REPEAT_S
        for event in events:
            text = _line("danger" if event == "repeat" else event, snap, self.repeats)
            if self.level is None and event in LEVELS:
                text += " (already at this level when this process started)"
            snap["message"] = text
            snap["event_at"] = int(self.clock.time())   # the UI toasts on events, not on every 15-second reading
            self._emit(text)
        self.level, self.free = level, free
        view = {k: v for k, v in snap.items() if k not in ("free", "total", "warn", "danger", "his")}
        view["event_at"] = snap.get("event_at", self.view_data.get("event_at", 0))
        self.view_data = view
        self._persist(self.view_data)
        return True


def read_state() -> dict:
    """The last level another process in this user's temp folder wrote, or an empty view."""
    try:
        state, _ = state_paths()
        data = json.loads(state.read_text(encoding="utf-8"))
        return dict(_EMPTY, **{k: v for k, v in data.items() if k in _EMPTY})
    except (OSError, ValueError, TypeError):
        return dict(_EMPTY)


def start(paths: list[tuple[str, Path]] | None = None, say: Callable[[str], None] | None = None) -> Watcher:
    """Start the 15-second watcher in this process and take the first reading now."""
    global _watcher
    if _watcher is None:
        _watcher = Watcher([*(paths or ()), *process_paths()], say=say)
        idle_watch.watch.add(SAMPLE_S, _watcher.check, repeat=True)
        _watcher.check()
    return _watcher


def view() -> dict:
    """The API's read-only view; no disk is read here."""
    return dict(_watcher.view_data if _watcher is not None and _watcher.view_data.get("at") else read_state())


def warning_line() -> str | None:
    """A short public line for /health: no volume path leaves the server."""
    v = view()
    if v["level"] not in (WARN, DANGER) or v["free_gib"] is None:
        return None
    words = "critically full" if v["level"] == DANGER else "nearly full"
    return f"Disk is {words}: {v['free_gib']:.1f} GiB free"
