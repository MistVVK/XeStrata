"""The RAM figure the Terminal's tools read (strata_control/cgroup_memory.py; split_engine.ram_gib, tune_machine.memory,
and setup's ram_gb through the same limit reader): inside a container the cgroup limit, outside it /proc/meminfo exactly
as before.  Fake /proc and cgroup trees in a temporary folder; no container needed.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import cgroup_memory as C
from strata_control import split_engine, tune_machine

GIB = 1 << 30
MIB = 1 << 20
HOST_KB = 527_799_316            # a 503 GiB host, as /proc/meminfo shows it inside the rented container
HOST_AVAIL_KB = 500_000_000
POD_LIMIT = 61_998_571_520       # that container's memory.max: 57.74 GiB


def stat(inactive=0, active=0, dirty=0, writeback=0, shmem=0, v1=False):
    p = "total_" if v1 else ""
    d, w = ("total_dirty", "total_writeback") if v1 else ("file_dirty", "file_writeback")
    return (f"{p}anon 1000\n{p}file {inactive + active + shmem}\n{p}shmem {shmem}\n{p}inactive_anon {shmem}\n"
            f"{p}active_file {active}\n{p}inactive_file {inactive}\n{d} {dirty}\n{w} {writeback}\n")


class Tree:
    def __init__(self, total_kb=HOST_KB, avail_kb=HOST_AVAIL_KB):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "cg").mkdir()
        self.put("meminfo", f"MemTotal:       {total_kb} kB\nMemFree:        1000 kB\nMemAvailable:   {avail_kb} kB")

    def put(self, rel, text):
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{text}\n")

    def group(self, rel, limit=None, current=0, high=None, stat_text="", v1=False):
        base = f"cg/memory/{rel}" if v1 else f"cg/{rel}"
        if v1:
            self.put(f"{base}/memory.limit_in_bytes", limit if limit is not None else 9223372036854771712)
            self.put(f"{base}/memory.usage_in_bytes", current)
        else:
            self.put(f"{base}/memory.max", limit if limit is not None else "max")
            self.put(f"{base}/memory.high", high if high is not None else "max")
            self.put(f"{base}/memory.current", current)
        self.put(f"{base}/memory.stat", stat_text)

    def ram(self):
        return C.ram_bytes(str(self.dir / "meminfo"), str(self.dir / "self_cgroup"), str(self.dir / "cg"))

    def close(self):
        self.tmp.cleanup()


class RamBytes(unittest.TestCase):
    def setUp(self):
        self.t = Tree()

    def tearDown(self):
        self.t.close()

    def test_v2_container_limit(self):
        # the pod: 20 GiB used, of which 8 GiB clean file cache (5 active, 3 inactive) and 1 GiB of it dirty
        self.t.put("self_cgroup", "0::/")
        self.t.group(".", limit=POD_LIMIT, current=20 * GIB, stat_text=stat(inactive=3 * GIB, active=5 * GIB, dirty=GIB))
        total, avail = self.t.ram()
        self.assertEqual(total, POD_LIMIT)
        self.assertEqual(avail, POD_LIMIT - (20 * GIB - 7 * GIB))

    def test_shmem_is_not_credited(self):
        self.t.put("self_cgroup", "0::/")
        self.t.group(".", limit=24 * GIB, current=10 * GIB, stat_text=stat(inactive=2 * GIB, shmem=3 * GIB))
        self.assertEqual(self.t.ram(), (24 * GIB, 24 * GIB - 8 * GIB))

    def test_memory_high_counts_as_the_limit(self):
        self.t.put("self_cgroup", "0::/")
        self.t.group(".", limit=40 * GIB, high=30 * GIB, current=10 * GIB, stat_text=stat())
        self.assertEqual(self.t.ram(), (30 * GIB, 20 * GIB))

    def test_v1_container_limit(self):
        self.t.put("self_cgroup", "4:memory:/docker/abc\n1:cpu,cpuacct:/docker/abc")
        self.t.group("docker/abc", limit=32 * GIB, current=12 * GIB, stat_text=stat(inactive=GIB, active=GIB, v1=True), v1=True)
        self.t.group("docker", v1=True)
        self.t.group(".", v1=True)
        self.assertEqual(self.t.ram(), (32 * GIB, 32 * GIB - 10 * GIB))

    def test_limit_on_a_parent_level(self):
        # the process's own group has no limit; its parent's 24 GiB is the one that holds (and its usage counts)
        self.t.put("self_cgroup", "0::/box/job")
        self.t.group(".")
        self.t.group("box", limit=24 * GIB, current=18 * GIB, stat_text=stat(active=2 * GIB))
        self.t.group("box/job", current=6 * GIB, stat_text=stat(active=GIB))
        self.assertEqual(self.t.ram(), (24 * GIB, 24 * GIB - 16 * GIB))

    def test_the_tighter_of_two_levels(self):
        self.t.put("self_cgroup", "0::/box/job")
        self.t.group(".")
        self.t.group("box", limit=40 * GIB, current=36 * GIB, stat_text=stat())
        self.t.group("box/job", limit=30 * GIB, current=10 * GIB, stat_text=stat())
        self.assertEqual(self.t.ram(), (30 * GIB, 4 * GIB))

    def test_no_limit_is_meminfo_exactly(self):
        self.t.put("self_cgroup", "0::/user.slice/u.scope")
        self.t.group(".")
        self.t.group("user.slice", current=50 * GIB, stat_text=stat())
        self.t.group("user.slice/u.scope", current=GIB, stat_text=stat())
        self.assertEqual(self.t.ram(), (HOST_KB * 1024, HOST_AVAIL_KB * 1024))

    def test_limit_at_or_above_memtotal_is_meminfo_exactly(self):
        for limit in (HOST_KB * 1024, HOST_KB * 1024 + GIB):
            self.t.put("self_cgroup", "0::/")
            self.t.group(".", limit=limit, current=400 * GIB, stat_text=stat())
            self.assertEqual(self.t.ram(), (HOST_KB * 1024, HOST_AVAIL_KB * 1024), limit)

    def test_no_cgroup_files_is_meminfo_exactly(self):
        self.t.put("self_cgroup", "0::/")
        self.assertEqual(self.t.ram(), (HOST_KB * 1024, HOST_AVAIL_KB * 1024))

    def test_container_free_never_above_the_hosts(self):
        t = Tree(avail_kb=4 * MIB)            # the host has 4 GiB available; the container's room is larger
        try:
            t.put("self_cgroup", "0::/")
            t.group(".", limit=24 * GIB, current=GIB, stat_text=stat())
            self.assertEqual(t.ram(), (24 * GIB, 4 * GIB))
        finally:
            t.close()

    def test_no_meminfo(self):
        self.assertEqual(C.ram_bytes(str(self.t.dir / "none"), str(self.t.dir / "self_cgroup"), str(self.t.dir / "cg")),
                         (None, None))

    def test_limit_reader_matches_ram(self):
        self.t.put("self_cgroup", "0::/box")
        self.t.group(".")
        self.t.group("box", limit=24 * GIB, current=GIB, stat_text=stat())
        self.assertEqual(C.memory_limit(str(self.t.dir / "self_cgroup"), str(self.t.dir / "cg")), 24 * GIB)


class Callers(unittest.TestCase):
    """split_engine.ram_gib and tune_machine.memory give the helper's figures, in their units; outside a container the
    same numbers as their earlier /proc/meminfo arithmetic."""

    def test_split_engine_ram_gib(self):
        with mock.patch.object(C, "ram_bytes", lambda: (POD_LIMIT, 13 * GIB)):
            self.assertEqual(split_engine.ram_gib(), (POD_LIMIT / GIB, 13.0))
        with mock.patch.object(C, "ram_bytes", lambda: (HOST_KB * 1024, HOST_AVAIL_KB * 1024)):
            self.assertEqual(split_engine.ram_gib(), (HOST_KB / (1 << 20), HOST_AVAIL_KB / (1 << 20)))
        with mock.patch.object(os, "name", "posix"), mock.patch.object(C, "ram_bytes", lambda: (None, None)):
            self.assertEqual(split_engine.ram_gib(), (0.0, 0.0))

    @unittest.skipIf(os.name == "nt", "the Linux branch of tune_machine.memory")
    def test_tune_machine_memory(self):
        with mock.patch.object(tune_machine.sys, "platform", "linux"), \
                mock.patch.object(C, "ram_bytes", lambda: (HOST_KB * 1024, HOST_AVAIL_KB * 1024)):
            self.assertEqual(tune_machine.memory(), {"total_mib": HOST_KB // 1024, "available_mib": HOST_AVAIL_KB // 1024})
        with mock.patch.object(tune_machine.sys, "platform", "linux"), \
                mock.patch.object(C, "ram_bytes", lambda: (POD_LIMIT, 13 * GIB)):
            self.assertEqual(tune_machine.memory(), {"total_mib": POD_LIMIT // MIB, "available_mib": 13 * 1024})
        with mock.patch.object(tune_machine.sys, "platform", "linux"), mock.patch.object(C, "ram_bytes", lambda: (None, None)):
            self.assertEqual(tune_machine.memory(), {"total_mib": None, "available_mib": None})


if __name__ == "__main__":
    unittest.main()
