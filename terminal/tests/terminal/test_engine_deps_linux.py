"""The Linux rule of an engine package: every library a bundled file needs (DT_NEEDED) is in the package or on the short base-system list.

boxia 2026-11-15: the hip-rx6000 / rx9070 / rx7900 packages of 0.2.14 hold librocm_sysdeps_numa.so.1, which needs libatomic.so.1; no package
bundled it, because the old list called libatomic "a library every distribution has". A container without the libatomic1 package could not start
the engine. Proves: the base list is glibc and the C++ runtime only; a finished archive is checked member by member (a symlink is not bundled,
the install keeps none); collect_libs.py bundles libatomic.so.1 from the build machine as a real file under its soname, with its notice; and,
on the owner PC where the published 0.2.14 files are (C:\\rc214\\FINAL4\\UPLOAD, read only, or STRATA_FINAL4_DIR), the archive check finds
exactly today's gap: libatomic.so.1 in the three AMD packages, nothing in hip-halo or the CUDA ones. Tiny real ELF files, no toolchain needed."""

import io
import os
import struct
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "tools" / "packaging"))
import binary_deps  # noqa: E402
import collect_libs  # noqa: E402

FINAL4 = Path(os.environ.get("STRATA_FINAL4_DIR", r"C:\rc214\FINAL4\UPLOAD"))


def make_elf(needed, soname=None) -> bytes:
    """A minimal ELF64 little-endian file with a .dynstr and a .dynamic section naming `needed` (DT_NEEDED) and `soname` (DT_SONAME)."""
    strtab = bytearray(b"\0")
    entries = []
    for name in needed:
        entries.append((1, len(strtab)))
        strtab += name.encode() + b"\0"
    if soname:
        entries.append((14, len(strtab)))
        strtab += soname.encode() + b"\0"
    entries.append((0, 0))
    str_off = 64
    dyn_off = (str_off + len(strtab) + 7) // 8 * 8
    dynamic = b"".join(struct.pack("<qQ", tag, val) for tag, val in entries)
    sh_off = (dyn_off + len(dynamic) + 7) // 8 * 8
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4], header[5], header[6] = 2, 1, 1
    struct.pack_into("<HHI", header, 16, 3, 0x3E, 1)
    struct.pack_into("<Q", header, 0x28, sh_off)
    struct.pack_into("<HHHHHH", header, 0x34, 64, 0, 0, 64, 3, 1)

    def section(stype, offset, size, link=0, entsize=0):
        return struct.pack("<IIQQQQIIQQ", 0, stype, 0, 0, offset, size, link, 0, 8, entsize)
    sections = section(0, 0, 0) + section(3, str_off, len(strtab)) + section(6, dyn_off, len(dynamic), link=1, entsize=16)
    out = bytearray(header)
    out += b"\0" * (str_off - len(out)) + strtab
    out += b"\0" * (dyn_off - len(out)) + dynamic
    out += b"\0" * (sh_off - len(out)) + sections
    return bytes(out)


def make_tar(path: Path, files: dict, links: dict | None = None) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)


class BaseList(unittest.TestCase):
    def test_the_reader_reads_the_tiny_elf(self):
        self.assertEqual(binary_deps.elf_needed_bytes(make_elf(["libatomic.so.1", "libc.so.6"], soname="librocm_sysdeps_numa.so.1")),
                         ["libatomic.so.1", "libc.so.6"])

    def test_the_base_list_is_glibc_and_the_cpp_runtime(self):
        for name in ("libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1", "ld-linux-x86-64.so.2", "libgcc_s.so.1", "libstdc++.so.6"):
            self.assertTrue(binary_deps._match(name, binary_deps.LINUX_OS, False), name)
        for name in ("libatomic.so.1", "libgomp.so.1", "libz.so.1", "libnuma.so.1", "libdrm.so.2", "libelf.so.1", "libtinfo.so.6"):
            self.assertFalse(binary_deps._match(name, binary_deps.LINUX_OS, False), name)

    def test_a_package_folder_without_libatomic_is_refused_and_with_it_passes(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d)
            (pkg / "strata").write_bytes(make_elf(["librocm_sysdeps_numa.so.1", "libc.so.6", "libstdc++.so.6"]))
            (pkg / "librocm_sysdeps_numa.so.1").write_bytes(make_elf(["libatomic.so.1", "libc.so.6"]))
            self.assertEqual(binary_deps.uncovered(pkg), [("librocm_sysdeps_numa.so.1", "libatomic.so.1")])
            (pkg / "libatomic.so.1").write_bytes(make_elf(["libc.so.6"], soname="libatomic.so.1"))
            self.assertEqual(binary_deps.uncovered(pkg), [])


class Archive(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.files = {"strata": make_elf(["librocm_sysdeps_numa.so.1", "libhsa-runtime64.so.1", "libc.so.6"]),
                      "librocm_sysdeps_numa.so.1": make_elf(["libatomic.so.1", "libc.so.6"]),
                      "BUILD.json": b"{}", "licenses/notice.txt": b"text"}

    def test_the_archive_check_names_the_gap(self):
        tar = self.root / "a.tar.gz"
        make_tar(tar, self.files)
        self.assertEqual(binary_deps.archive_uncovered(tar), [("librocm_sysdeps_numa.so.1", "libatomic.so.1")])
        self.assertEqual(binary_deps.main(["binary_deps.py", str(tar)]), 1)

    def test_a_bundled_libatomic_closes_it_and_a_symlink_does_not(self):
        tar = self.root / "b.tar.gz"
        make_tar(tar, self.files, links={"libatomic.so.1": "libatomic.so.1.2.0"})
        self.assertEqual(binary_deps.archive_uncovered(tar), [("librocm_sysdeps_numa.so.1", "libatomic.so.1")])
        tar = self.root / "c.tar.gz"
        make_tar(tar, {**self.files, "libatomic.so.1": make_elf(["libc.so.6"], soname="libatomic.so.1")})
        self.assertEqual(binary_deps.archive_uncovered(tar), [])
        self.assertEqual(binary_deps.main(["binary_deps.py", str(tar)]), 0)


class Collect(unittest.TestCase):
    """collect_libs.py with ldd faked: the build machine's libatomic is copied as a real file under its soname, with its notice; libc stays the OS's."""

    def test_libatomic_is_bundled_with_its_notice(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            system = root / "usr" / "lib"
            system.mkdir(parents=True)
            real = system / "libatomic.so.1.2.0"
            real.write_bytes(make_elf(["libc.so.6"], soname="libatomic.so.1"))
            (system / "libc.so.6").write_bytes(make_elf([]))
            rocm = root / "rocm"
            rocm.mkdir()
            (rocm / "librocm_sysdeps_numa.so.1").write_bytes(make_elf(["libatomic.so.1", "libc.so.6"]))
            binary = root / "strata"
            binary.write_bytes(make_elf(["librocm_sysdeps_numa.so.1", "libc.so.6"]))
            out = root / "stage"

            def deps(item, _search):
                name = Path(item).name
                if name == "strata":
                    return {"librocm_sysdeps_numa.so.1": str(rocm / "librocm_sysdeps_numa.so.1"), "libc.so.6": str(system / "libc.so.6")}, []
                if name == "librocm_sysdeps_numa.so.1":
                    return {"libatomic.so.1": str(real), "libc.so.6": str(system / "libc.so.6")}, []
                return {"libc.so.6": str(system / "libc.so.6")}, []
            with mock.patch.object(collect_libs, "dependencies", deps), \
                    mock.patch.object(collect_libs, "is_system", lambda p: str(p).startswith(str(system))), \
                    mock.patch.object(collect_libs, "_owner_package", return_value=("libatomic1", "14.2.0-4ubuntu2~24.04.1")):
                self.assertEqual(collect_libs.main(["--binary", str(binary), "--out", str(out)]), 0)
            atomic = out / "libatomic.so.1"
            self.assertTrue(atomic.is_file() and not atomic.is_symlink())
            self.assertEqual(atomic.read_bytes(), real.read_bytes())
            self.assertFalse((out / "libc.so.6").exists())
            self.assertEqual((out / "SYSTEM-LIBS.txt").read_text().split(), ["libc.so.6"])
            notice = (out / "licenses" / "libatomic.txt").read_text()
            self.assertIn("GPL-3.0-or-later WITH GCC-exception-3.1", notice)
            self.assertIn("(distribution package libatomic1 14.2.0-4ubuntu2~24.04.1)", notice)
            self.assertEqual(binary_deps.uncovered(out), [])

    def test_the_bundle_list(self):
        self.assertTrue(collect_libs.bundled_anyway("libatomic.so.1"))
        self.assertFalse(collect_libs.bundled_anyway("libc.so.6"))
        self.assertTrue(collect_libs.bundled_anyway("libfoo.so.2", ["libfoo.so*"]))
        self.assertTrue((ROOT.parent / "tools" / "packaging" / "notices" / "libatomic.txt").is_file())


@unittest.skipUnless(FINAL4.is_dir(), "the published 0.2.14 files are only on the owner PC (C:\\rc214\\FINAL4\\UPLOAD) or in STRATA_FINAL4_DIR")
class PublishedPackages(unittest.TestCase):
    """The check on the real 0.2.14 Linux engine archives (read only): it would have refused the three AMD packages for libatomic.so.1."""

    def check(self, name):
        path = FINAL4 / name
        if not path.is_file():
            self.skipTest(f"{name} is not in {FINAL4}")
        return binary_deps.archive_uncovered(path)

    def test_the_three_amd_packages_lack_libatomic(self):
        for variant in ("rx6000", "rx9070", "rx7900"):
            self.assertEqual(self.check(f"strata-engine-0.1.60-linux-x86_64-hip-{variant}.tar.gz"), [("librocm_sysdeps_numa.so.1", "libatomic.so.1")], variant)

    def test_halo_and_cuda_need_nothing_else(self):
        for name in ("strata-engine-0.1.60-linux-x86_64-hip-halo.tar.gz", "strata-engine-0.1.60-linux-x86_64-cuda-cuda13.tar.gz",
                     "strata-engine-0.1.60-linux-x86_64-cuda-cuda12.tar.gz", "strata-engine-0.1.59-linux-aarch64-cuda-cuda13.tar.gz"):
            self.assertEqual(self.check(name), [], name)


if __name__ == "__main__":
    unittest.main()
