"""Tests for setup.py's RAM figure in a container: ram_gb() takes the smaller of /proc/meminfo MemTotal (the host's,
inside a container) and the cgroup memory limit, read as the engine's memory guard reads it.  Fake /proc and cgroup
trees in a temporary folder; no container needed.

    python -m unittest tools.test_setup_container_ram
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

GIB = 1 << 30
HOST_KB = 527_000_000          # a 503 GiB host, as /proc/meminfo shows it inside the rented container
POD_LIMIT = 61_998_571_520     # memory.max of that container: 57.74 GiB


class Tree:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "cg").mkdir()

    def put(self, rel: str, text) -> None:
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{text}\n")

    def self_cgroup(self, text: str) -> None:
        self.put("self_cgroup", text)

    def limit(self) -> int | None:
        return setup._cgroup_memory_limit(str(self.dir / "self_cgroup"), str(self.dir / "cg"))

    def close(self):
        self.tmp.cleanup()


class CgroupLimit(unittest.TestCase):
    def setUp(self):
        self.t = Tree()

    def tearDown(self):
        self.t.close()

    def test_container_limit_at_the_namespace_root(self):
        self.t.self_cgroup("0::/")
        self.t.put("cg/memory.max", POD_LIMIT)
        self.t.put("cg/memory.current", 7 * GIB)
        self.assertEqual(self.t.limit(), POD_LIMIT)

    def test_no_limit_anywhere(self):
        self.t.self_cgroup("0::/user.slice/u.scope")
        self.t.put("cg/user.slice/memory.max", "max")
        self.t.put("cg/user.slice/memory.current", 3 * GIB)
        self.t.put("cg/user.slice/u.scope/memory.max", "max")
        self.t.put("cg/user.slice/u.scope/memory.high", "max")
        self.t.put("cg/user.slice/u.scope/memory.current", GIB)
        self.assertIsNone(self.t.limit())

    def test_limit_on_a_parent_level(self):
        self.t.self_cgroup("0::/a/b")
        self.t.put("cg/a/memory.max", 40 * GIB)
        self.t.put("cg/a/memory.current", 10 * GIB)
        self.t.put("cg/a/b/memory.max", "max")
        self.t.put("cg/a/b/memory.current", 5 * GIB)
        self.assertEqual(self.t.limit(), 40 * GIB)

    def test_tightest_level_wins(self):
        self.t.self_cgroup("0::/a/b")
        self.t.put("cg/a/memory.max", 40 * GIB)
        self.t.put("cg/a/memory.current", 10 * GIB)
        self.t.put("cg/a/b/memory.max", 24 * GIB)
        self.t.put("cg/a/b/memory.current", 5 * GIB)
        self.assertEqual(self.t.limit(), 24 * GIB)

    def test_memory_high_below_max_counts(self):
        self.t.self_cgroup("0::/g")
        self.t.put("cg/g/memory.max", 64 * GIB)
        self.t.put("cg/g/memory.high", 48 * GIB)
        self.t.put("cg/g/memory.current", GIB)
        self.assertEqual(self.t.limit(), 48 * GIB)

    def test_host_path_without_a_namespace(self):
        # the container sees /docker/abc in /proc/self/cgroup, but only its own group at the mount
        self.t.self_cgroup("0::/system.slice/docker-abc.scope")
        self.t.put("cg/memory.max", 32 * GIB)
        self.t.put("cg/memory.current", GIB)
        self.assertEqual(self.t.limit(), 32 * GIB)

    def test_level_without_memory_files_is_skipped(self):
        self.t.self_cgroup("0::/a")
        self.t.put("cg/cgroup.controllers", "cpu memory")   # the host's root group: no memory.max
        self.t.put("cg/a/memory.max", 20 * GIB)
        self.t.put("cg/a/memory.current", GIB)
        self.assertEqual(self.t.limit(), 20 * GIB)

    def test_cgroup_v1(self):
        self.t.self_cgroup("12:memory:/docker/abc\n11:cpu,cpuacct:/docker/abc")
        self.t.put("cg/memory/memory.limit_in_bytes", 9223372036854771712)   # v1's "unlimited"
        self.t.put("cg/memory/docker/memory.limit_in_bytes", 9223372036854771712)
        self.t.put("cg/memory/docker/abc/memory.limit_in_bytes", 16 * GIB)
        self.assertEqual(self.t.limit(), 16 * GIB)

    def test_cgroup_v1_unlimited(self):
        self.t.self_cgroup("12:memory:/")
        self.t.put("cg/memory/memory.limit_in_bytes", 9223372036854771712)
        self.assertIsNone(self.t.limit())

    def test_v2_files_present_v1_not_read(self):
        # the engine reads v1 only when no v2 memory files are there
        self.t.self_cgroup("0::/g\n12:memory:/g")
        self.t.put("cg/g/memory.max", "max")
        self.t.put("cg/g/memory.current", GIB)
        self.t.put("cg/memory/g/memory.limit_in_bytes", 8 * GIB)
        self.assertIsNone(self.t.limit())

    def test_unreadable_values_give_no_limit(self):
        self.t.self_cgroup("0::/g")
        self.t.put("cg/g/memory.max", "garbage")
        self.t.put("cg/g/memory.current", GIB)
        self.assertIsNone(self.t.limit())

    def test_no_self_cgroup_file_reads_the_mount_root(self):
        self.t.put("cg/memory.max", 12 * GIB)
        self.t.put("cg/memory.current", GIB)
        self.assertEqual(self.t.limit(), 12 * GIB)

    def test_dot_dot_is_refused(self):
        self.t.self_cgroup("0::/../x")
        self.t.put("cg/memory.max", 12 * GIB)
        self.t.put("cg/memory.current", GIB)
        self.assertIsNone(self.t.limit())


class RamGb(unittest.TestCase):
    def ram(self, meminfo_kb: int, limit) -> float:
        def fake_open(path, *a, **kw):
            if path == "/proc/meminfo":
                import io
                return io.StringIO(f"MemTotal:       {meminfo_kb} kB\nMemFree:        1000 kB\n")
            raise OSError(path)
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch("builtins.open", fake_open), mock.patch.object(setup, "_cgroup_memory_limit", lambda: limit):
            return setup.ram_gb()

    def test_container_takes_the_limit(self):
        self.assertAlmostEqual(self.ram(HOST_KB, POD_LIMIT), POD_LIMIT / GIB, places=6)

    def test_bare_metal_unchanged(self):
        self.assertEqual(self.ram(HOST_KB, None), HOST_KB * 1024 / GIB)

    def test_limit_above_memtotal_unchanged(self):
        self.assertEqual(self.ram(65_000_000, 600 * GIB), 65_000_000 * 1024 / GIB)

    def test_limit_equal_to_memtotal_unchanged(self):
        self.assertEqual(self.ram(65_000_000, 65_000_000 * 1024), 65_000_000 * 1024 / GIB)

    def test_no_meminfo_stays_zero(self):
        def no_open(path, *a, **kw):
            raise OSError(path)
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch("builtins.open", no_open), mock.patch.object(setup, "_cgroup_memory_limit", lambda: 8 * GIB):
            self.assertEqual(setup.ram_gb(), 0.0)


if __name__ == "__main__":
    unittest.main()
