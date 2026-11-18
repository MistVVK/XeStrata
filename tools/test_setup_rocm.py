# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Which ROCm runs the engine, a promise a machine can silently break: AMD's own ROCm (its packages install no
ld.so.conf, so the loader alone does not say it is there) is preferred when it holds the libraries the engine links
and the GPUs' code, the distribution's is demanded when it does not, and setup's one line names the one in use.
A fake /opt tree and a substituted `ldconfig -p` stand for the machine, and the engine's linked names come from a
hand-made ELF.

    python -m unittest tools.test_setup_rocm
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# the engine's linked names today: the HIP runtime, rocBLAS, hipBLASLt (an AMD ROCm 10.1 tree has them)
LIBS = ("libamdhip64.so.7", "librocblas.so.5", "libhipblaslt.so.1")
GFX = {"gfx1200"}


def fake_elf(names: list) -> bytes:
    """A 64-bit little-endian ELF whose dynamic section needs `names` - all the parser of elf_needed reads."""
    strtab = bytearray(b"\0")
    offs = {}
    for n in names:
        offs[n] = len(strtab)
        strtab += n.encode() + b"\0"
    strv = 0x1000
    dyn = b"".join(struct.pack("<QQ", 1, offs[n]) for n in names) + struct.pack("<QQ", 5, strv) + \
        struct.pack("<QQ", 0, 0)
    dynoff = strv + len(strtab)
    size = dynoff + len(dyn)

    def phdr(t: int, off: int, filesz: int) -> bytes:
        return struct.pack("<IIQQQQQQ", t, 0, off, off, off, filesz, filesz, 0x1000)

    e = bytearray(64)
    e[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HHIQQQ", e, 16, 2, 0x3E, 1, 0, 64, 0)      # e_type, e_machine, e_version, entry, phoff
    struct.pack_into("<HHHHHH", e, 52, 64, 56, 2, 0, 0, 0)        # ehsize, phentsize, phnum
    out = bytes(e) + phdr(1, 0, size) + phdr(2, dynoff, len(dyn))
    return out + b"\0" * (strv - len(out)) + bytes(strtab) + dyn


def make_rocm_tree(opt: Path, name: str, libs=LIBS, rocblas=("gfx1200",), hipblaslt=("gfx1200",)) -> Path:
    """A ROCm tree as the shapes installed have it: lib/ with the libraries, rocBLAS's and hipBLASLt's kernel files
    named by their arch."""
    root = opt / name
    lib = root / "lib"
    lib.mkdir(parents=True)
    for n in libs:
        (lib / n).write_bytes(b"")
    for d, archs in ((lib / "rocblas/library", rocblas), (lib / "hipblaslt/library", hipblaslt)):
        d.mkdir(parents=True)
        for g in archs:
            (d / f"Kernels-000-{g}.hsaco").write_bytes(b"")
    return root


class WhichRocm(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.opt = Path(tmp.name)
        self.eng = Path(tmp.name) / "engine"
        (self.eng / "lib").mkdir(parents=True)
        (self.eng / setup.EXE).write_bytes(fake_elf(["libc.so.6", *LIBS]))
        (self.eng / "lib" / "libonemath.so").write_bytes(fake_elf([LIBS[1], LIBS[2]]))
        for p in (mock.patch.object(setup, "ldconfig_print", return_value=""),       # a machine without the loader's
                  mock.patch.object(setup, "packaged_engine_dir", return_value=self.eng)):
            p.start()
            self.addCleanup(p.stop)

    def roots(self):
        return setup.rocm_roots(self.opt)

    def test_amds_own_rocm_is_preferred_and_the_distribution_not_asked_for(self):
        root = make_rocm_tree(self.opt, "rocm/core-10.1")
        with mock.patch.object(setup, "rocm_roots", return_value=self.roots()):
            self.assertEqual(setup.amd_rocm_usable(setup.amd_needed(self.eng), GFX), root)
            self.assertTrue(setup.amd_runtime_ok(GFX))

    def test_without_amds_own_rocm_the_distribution_is_asked_for(self):
        self.assertEqual(self.roots(), [])
        self.assertFalse(setup.amd_runtime_ok(GFX))

    def test_a_short_tree_or_another_soname_leaves_the_distribution_asked_for(self):
        make_rocm_tree(self.opt, "rocm/core-10.1", libs=LIBS[:2])
        self.assertIsNone(setup.amd_rocm_usable(setup.amd_needed(self.eng), GFX, self.roots()))
        make_rocm_tree(self.opt, "rocm-6.2.4", libs=("libamdhip64.so.7", "librocblas.so.4", "libhipblaslt.so.1"))
        self.assertIsNone(setup.amd_rocm_usable(setup.amd_needed(self.eng), GFX, self.roots()))

    def test_ubuntus_old_tree_and_a_rocm_with_an_ld_so_conf_run_it_too(self):
        # /opt/rocm-<version> is the old shape (Debian's rocm, ROCm's own installers); a tree that writes an
        # ld.so.conf the loader finds instead, whatever its shape
        root = make_rocm_tree(self.opt, "rocm-6.2.4", libs=("libamdhip64.so.7", "librocblas.so.4", "libhipblaslt.so.1"))
        names = ["libamdhip64.so.7", "librocblas.so.4", "libhipblaslt.so.1"]
        with mock.patch.object(setup, "rocm_roots", return_value=self.roots()):
            self.assertEqual(setup.amd_rocm_usable(names, GFX), root)
        cache = "\tlibrocblas.so.4 (libc6,x86-64) => /opt/rocm-6.2.4/lib/librocblas.so.4\n"
        with mock.patch.object(setup, "ldconfig_print", return_value=cache):
            self.assertFalse(setup.amd_libs_loadable())     # the cache names rocBLAS alone: not enough
        more = "".join(f"\t{lib} (libc6,x86-64) => /opt/rocm-6.2.4/lib/{lib}\n"
                       for lib in ("libamdhip64.so.7", "libhipblaslt.so.1"))
        with mock.patch.object(setup, "ldconfig_print", return_value=cache + more):
            self.assertTrue(setup.amd_runtime_ok(GFX))

    def test_setups_line_names_the_rocm_in_use(self):
        self.assertIn("ROCm", setup.amd_rocm_line(make_rocm_tree(self.opt, "rocm/core-10.1")))
        self.assertIn("distribution", setup.amd_rocm_line(None))


if __name__ == "__main__":
    unittest.main()
