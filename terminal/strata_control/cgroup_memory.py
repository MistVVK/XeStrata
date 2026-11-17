"""The RAM this process gets on Linux, read the way the engine reads it.

Inside a container /proc/meminfo shows the host's memory; the container's cgroup limit is the RAM there is. The limit is
the smallest of cgroup v2 memory.max and memory.high from the process's group up to the mount root (cut at the deepest
level that is visible: a container without its own cgroup namespace sees its host path, but only its own subtree), or of
cgroup v1 memory.limit_in_bytes when there are no v2 memory files.

Under a limit below MemTotal the available figure is the limit less the group's usage, with clean file cache on the
active and inactive lists counted as free again (memory.stat: active_file + inactive_file, bounded by the usage, less
dirty and writeback pages; shmem and locked pages are on other lists), as the engine's memory guard counts it. Without
a limit, or with a limit at or above MemTotal, both figures are /proc/meminfo's, as before.
"""
from __future__ import annotations

import os

SELF_CGROUP = "/proc/self/cgroup"
CGROUP_ROOT = "/sys/fs/cgroup"
MEMINFO = "/proc/meminfo"


def _levels(root: str, group: str) -> list[str]:
    """The cgroup directories from the mount `root` down to `group` ("/a/b"), cut at the first one that is not there."""
    levels = [root]
    at = root
    for part in [p for p in group.split("/") if p]:
        if part in (".", ".."):
            return []
        at = os.path.join(at, part)
        if not os.path.isdir(at):
            break
        levels.append(at)
    return levels if os.path.isdir(root) else []


def _number(path: str, v1: bool = False) -> int | None:
    """A cgroup number file: its value, or None for "max", an absent or unreadable file, or v1's "unlimited"."""
    try:
        with open(path) as stream:
            text = stream.read().strip()
    except OSError:
        return None
    if not text.isdigit():
        return None
    value = int(text)
    if v1 and value >= 1 << 62:
        return None
    return value


def limited_levels(self_cgroup: str = SELF_CGROUP, root: str = CGROUP_ROOT) -> list[tuple[int, str, bool]]:
    """(limit in bytes, the level's directory, is cgroup v1) for every level of this process's cgroup that sets one."""
    try:
        with open(self_cgroup) as stream:
            lines = stream.read().splitlines()
    except OSError:
        lines = ["0::/"]
    v2_group = v1_group = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[0] == "0" and parts[1] == "":
            v2_group = parts[2]
        elif "memory" in parts[1].split(","):
            v1_group = parts[2]
    out: list[tuple[int, str, bool]] = []
    v2_files = False
    if v2_group is not None:
        for level in _levels(root, v2_group):
            if not os.path.exists(os.path.join(level, "memory.current")):
                continue
            v2_files = True
            values = [v for v in (_number(os.path.join(level, n)) for n in ("memory.max", "memory.high")) if v is not None]
            if values:
                out.append((min(values), level, False))
    if not v2_files and v1_group is not None:
        for level in _levels(os.path.join(root, "memory"), v1_group):
            value = _number(os.path.join(level, "memory.limit_in_bytes"), v1=True)
            if value is not None:
                out.append((value, level, True))
    return out


def memory_limit(self_cgroup: str = SELF_CGROUP, root: str = CGROUP_ROOT) -> int | None:
    """The smallest memory limit (bytes) on this process's cgroup and its ancestors; None when no level sets one."""
    limits = [limit for limit, _, _ in limited_levels(self_cgroup, root)]
    return min(limits) if limits else None


def clean_file_cache(stat_text: str, usage: int, v1: bool = False) -> int:
    """Clean file cache on the active and inactive lists (bytes), bounded by the usage, from a memory.stat text.
    0 when inactive_file is missing or unreadable; an active_file that cannot be read adds nothing."""
    prefix = "total_" if v1 else ""
    keys = {"inactive": prefix + "inactive_file", "active": prefix + "active_file",
            "dirty": "total_dirty" if v1 else "file_dirty", "writeback": "total_writeback" if v1 else "file_writeback"}
    seen: dict[str, int] = {}
    active_bad = False
    for line in stat_text.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        key, value = fields
        name = next((k for k, v in keys.items() if v == key), None)
        if name is None:
            continue
        ok = value.isdigit()
        if name == "inactive":
            if not ok or "inactive" in seen:
                return 0
        elif name == "active":
            if not ok or "active" in seen:
                active_bad = True
                continue
        elif not ok:
            return 0
        seen[name] = int(value)
    if "inactive" not in seen:
        return 0
    cached = seen["inactive"] + (seen["active"] if "active" in seen and not active_bad else 0)
    credit = min(cached, usage)
    for name in ("dirty", "writeback"):
        credit = max(credit - seen.get(name, 0), 0)
    return credit


def _level_room(limit: int, level: str, v1: bool) -> int:
    usage = _number(os.path.join(level, "memory.usage_in_bytes" if v1 else "memory.current"))
    if usage is None:
        return limit
    try:
        with open(os.path.join(level, "memory.stat")) as stream:
            stat_text = stream.read()
    except OSError:
        stat_text = ""
    used = usage - clean_file_cache(stat_text, usage, v1)
    return limit - used if used < limit else 0


def ram_bytes(meminfo: str = MEMINFO, self_cgroup: str = SELF_CGROUP, root: str = CGROUP_ROOT) -> tuple[int | None, int | None]:
    """(total, available) bytes of RAM for this process on Linux; None for a figure /proc/meminfo does not give (and
    both None where there is no /proc/meminfo)."""
    info: dict[str, int] = {}
    try:
        with open(meminfo, encoding="ascii") as stream:
            for line in stream:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    info[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None, None
    total, available = info.get("MemTotal"), info.get("MemAvailable")
    if total is None:
        return None, available
    mem_total = total
    for limit, level, v1 in limited_levels(self_cgroup, root):
        if limit >= mem_total:          # not a container's limit: /proc/meminfo's figures stand
            continue
        total = min(total, limit)
        if available is not None:
            available = min(available, _level_room(limit, level, v1))
    return total, available
