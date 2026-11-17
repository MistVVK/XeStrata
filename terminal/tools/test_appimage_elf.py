"""Tests for apps/desktop/scripts/appimage_elf.py (#4337: the bundled ripgrep, a static PIE, crashed in the Linux AppImage).

A static PIE relocates itself at start-up by reading its own `.dynamic` section. linuxdeploy's RPATH rewrite moves that section and fills the old
place; the program then does not start. The checks here: ELF files built in memory (run everywhere), and, on Linux with gcc and -static-pie, a real
static PIE that runs, the same file with its `.dynamic` overwritten the way the reporters found it (it must crash at start, and the scan must say so).
No downloads; nothing of the real install is touched.

    python -m unittest tools.test_appimage_elf
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "app" / "desktop" / "scripts"))
import appimage_elf as E  # noqa: E402


def make_static_pie(*, extra_tags=(), rela=True, garbage=False, filler=False, interp=False, loads=2, pie=True, entry=0x1000) -> bytes:
    """A small ELF64 ET_DYN: PT_LOADs, a PT_DYNAMIC inside the last one, optionally a PT_INTERP. Not runnable, only well formed."""
    page = 0x1000
    dyn = [(E.DT_RELA, 0x1100), (E.DT_RELASZ, 48), (E.DT_RELAENT, 24)] if rela else []
    dyn += ([(E.DT_FLAGS_1, E.DF_1_PIE)] if pie else []) + list(extra_tags) + [(E.DT_NULL, 0)]
    dyn_bytes = b"".join(struct.pack("<qQ", t, v) for t, v in dyn)
    if garbage:
        dyn_bytes = b"Z" * len(dyn_bytes)
    nph = loads + 1 + (1 if interp else 0)
    body = bytearray(page * (loads + 1))
    phdr = []
    for i in range(loads):
        phdr.append((E.PT_LOAD, 5 if i == 0 else 6, i * page, i * page, page, page, page))
    dyn_off = page * loads + 0x40
    # the last load covers the dynamic section
    phdr[-1] = (E.PT_LOAD, 6, (loads - 1) * page, (loads - 1) * page, 2 * page, 2 * page, page)
    phdr.append((E.PT_DYNAMIC, 6, dyn_off, dyn_off, len(dyn_bytes), len(dyn_bytes), 8))
    if interp:
        phdr.append((E.PT_INTERP, 4, 0x200, 0x200, 16, 16, 1))
    body[dyn_off:dyn_off + len(dyn_bytes)] = dyn_bytes
    if rela:
        body[0x1100:0x1100 + 48] = (b"Z" * 48) if filler else struct.pack("<QQqQQq", 0x2000, 8, 0x2100, 0x2008, 8, 0x2200)
    if filler:
        body[0x300:0x300 + 400] = b"X" * 400
    head = bytearray(64)
    head[:4] = b"\x7fELF"
    head[4], head[5], head[6] = 2, 1, 1
    struct.pack_into("<HHIQQQIHHHHHH", head, 16, E.ET_DYN, 62, 1, entry, 64, 0, 0, 64, 56, len(phdr), 64, 0, 0)
    out = bytearray(head)
    for p_type, flags, off, vaddr, filesz, memsz, align in phdr:
        out += struct.pack("<IIQQQQQQ", p_type, flags, off, vaddr, vaddr, filesz, memsz, align)
    data = bytes(out) + bytes(body[len(out):])
    return data


class Synthetic(unittest.TestCase):
    def test_a_sound_static_pie_has_no_problems(self):
        self.assertEqual(E.check_static_pie(make_static_pie()), [])
        self.assertEqual(E.check_static_pie(make_static_pie(loads=3)), [])

    def test_a_filled_dynamic_section_is_reported(self):
        problems = E.check_static_pie(make_static_pie(garbage=True))
        self.assertTrue(any("tags that do not exist" in p for p in problems), problems)

    def test_an_added_rpath_on_a_static_pie_is_reported(self):
        for tag in (E.DT_RPATH, E.DT_RUNPATH):
            problems = E.check_static_pie(make_static_pie(extra_tags=[(tag, 1)]))
            self.assertTrue(any("RPATH or RUNPATH" in p for p in problems), problems)

    def test_a_filled_relocation_table_and_filler_runs_are_reported(self):
        problems = E.check_static_pie(make_static_pie(filler=True))
        self.assertTrue(any("relocation table is filled" in p for p in problems), problems)
        self.assertTrue(any("filler bytes" in p for p in problems), problems)

    def test_a_dynamic_segment_outside_the_loaded_segments_is_reported(self):
        data = bytearray(make_static_pie())
        # point PT_DYNAMIC's address far outside every PT_LOAD
        info = E.read_elf(bytes(data))
        phoff = 64
        for i, p in enumerate(info["phdrs"]):
            if p["type"] == E.PT_DYNAMIC:
                struct.pack_into("<Q", data, phoff + i * 56 + 16, 0x900000)
        problems = E.check_static_pie(bytes(data))
        self.assertTrue(any("not inside any loaded segment" in p for p in problems), problems)

    def test_programs_that_are_not_static_pies_are_left_alone(self):
        self.assertEqual(E.check_static_pie(make_static_pie(garbage=True, interp=True)), [], "a dynamic program with an interpreter")
        self.assertEqual(E.check_static_pie(b"#!/bin/sh\necho hi\n"), [])
        self.assertEqual(E.check_static_pie(b"\x7fELF" + b"\x01" * 60), [], "32-bit")
        self.assertEqual(E.check_static_pie(b""), [])
        self.assertEqual(E.check_static_pie(b"\x7fELF\x02\x01" + b"\x00" * 100), [], "no program headers: nothing to check")

    def test_a_shared_library_with_no_interpreter_passes_however_it_is_named(self):
        """Pillow, numpy and sharp ship .so files with an RUNPATH of $ORIGIN, no PT_INTERP and (usually) no entry point: they were flagged as static PIEs."""
        sonames = [(E.DT_SONAME, 1), (E.DT_RUNPATH, 2)]
        for lib in (make_static_pie(entry=0, pie=False, extra_tags=sonames),            # the common case: no entry point
                    make_static_pie(entry=0, pie=False, extra_tags=[(E.DT_RUNPATH, 2)]),  # no SONAME either (a Node .node module)
                    make_static_pie(entry=0x1000, pie=False, extra_tags=sonames),         # an entry point (libc, ld.so style) but a SONAME and no PIE flag
                    make_static_pie(entry=0x1000, pie=True, extra_tags=sonames),          # even a stray PIE flag: a SONAME makes it a library
                    make_static_pie(entry=0, pie=False, garbage=True)):                   # a library with a damaged dynamic section is not this checker's business
            self.assertEqual(E.check_static_pie(lib), [])
        self.assertEqual(E.static_pie_kind(make_static_pie(entry=0, pie=False, extra_tags=sonames), E.read_elf(make_static_pie(entry=0, pie=False, extra_tags=sonames))), None)

    def test_a_static_pie_is_told_from_a_library_and_a_damaged_one_is_found(self):
        good = make_static_pie()
        self.assertEqual(E.static_pie_kind(good, E.read_elf(good)), "static-pie")
        self.assertEqual(E.check_static_pie(good), [])
        bad = make_static_pie(garbage=True)
        self.assertEqual(E.static_pie_kind(bad, E.read_elf(bad)), "damaged")
        self.assertTrue(E.check_static_pie(bad))
        # linuxdeploy's own edit: the section survives (moved) with a RUNPATH added; it is still a static PIE and the RUNPATH gives it away
        rp = make_static_pie(extra_tags=[(E.DT_RUNPATH, 1)])
        self.assertEqual(E.static_pie_kind(rp, E.read_elf(rp)), "static-pie")
        self.assertTrue(any("RPATH or RUNPATH" in p for p in E.check_static_pie(rp)))
        # no entry point, no PIE flag, not a static PIE: left alone
        self.assertEqual(E.check_static_pie(make_static_pie(pie=False)), [], "a sound dynamic section with no PIE flag and no SONAME is not a static PIE")

    def test_a_folder_of_libraries_and_one_static_pie(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                Path(tmp, f"lib{i}.so.1").write_bytes(make_static_pie(entry=0, pie=False, extra_tags=[(E.DT_SONAME, 1), (E.DT_RUNPATH, 2)]))
            Path(tmp, "something.node").write_bytes(make_static_pie(entry=0, pie=False, extra_tags=[(E.DT_RUNPATH, 2)]))
            Path(tmp, "rg").write_bytes(make_static_pie())
            self.assertEqual(E.scan(tmp), {}, "no name rules: only the static PIE is looked at, and it is sound")
            Path(tmp, "rg").write_bytes(make_static_pie(garbage=True))
            self.assertEqual(list(E.scan(tmp)), ["rg"])

    def test_a_truncated_file_is_a_report_not_a_crash(self):
        data = make_static_pie()
        for cut in (70, 200, 4100, len(data) - 10):
            E.check_static_pie(data[:cut])

    def test_folder_scan_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged, packed = Path(tmp, "staged"), Path(tmp, "packed")
            for base in (staged, packed):
                (base / "node_modules/@vscode/ripgrep-linux-x64/bin").mkdir(parents=True)
                (base / "lib").mkdir()
            good, bad = make_static_pie(), make_static_pie(garbage=True, extra_tags=[(E.DT_RUNPATH, 1)])
            rg = "node_modules/@vscode/ripgrep-linux-x64/bin/rg"
            (staged / rg).write_bytes(good)
            (packed / rg).write_bytes(bad)
            (staged / "lib/libx.so").write_bytes(good)
            (packed / "lib/libx.so").write_bytes(good)
            (staged / "readme.txt").write_text("text")
            found = E.scan(str(packed))
            self.assertEqual(list(found), [rg.replace("/", os.sep)])
            diffs = E.compare(str(staged), str(packed))
            self.assertEqual(list(diffs), [rg.replace("/", os.sep)])
            self.assertIn("bytes differ", diffs[rg.replace("/", os.sep)])
            # the repaired copy passes both
            (packed / rg).write_bytes(good)
            self.assertEqual(E.scan(str(packed)), {})
            self.assertEqual(E.compare(str(staged), str(packed)), {})
            # a missing file is reported
            (packed / "lib/libx.so").unlink()
            self.assertEqual(E.compare(str(staged), str(packed)), {os.path.join("lib", "libx.so"): ["missing"]})

    def test_the_command_line_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rg").write_bytes(make_static_pie())
            self.assertEqual(E.main(["x", "scan", tmp]), 0)
            Path(tmp, "rg").write_bytes(make_static_pie(garbage=True))
            self.assertEqual(E.main(["x", "scan", tmp]), 1)
            self.assertEqual(E.main(["x"]), 2)


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("gcc"), "needs Linux and gcc")
class RealStaticPie(unittest.TestCase):
    """A real static PIE from gcc: it runs; with its .dynamic filled the way the reports found it, it does not start, and the scan says why."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="strata-elf-")
        src = Path(cls.tmp, "t.c")
        src.write_text('#include <stdio.h>\n#include <string.h>\n#include <stdlib.h>\n'
                       'static const char *n[]={"alpha","beta","gamma"};\n'
                       'int main(int c,char**v){char b[256];memset(b,0,sizeof b);for(int i=0;i<3;i++)strcat(b,n[i]);'
                       'printf("ok %s %d\\n",b,c);void*p=malloc(100000);memcpy(p,b,200);free(p);return 0;}\n')
        cls.good = Path(cls.tmp, "good")
        built = subprocess.run(["gcc", "-O2", "-static-pie", str(src), "-o", str(cls.good)], capture_output=True)
        cls.ok = built.returncode == 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        if not self.ok:
            self.skipTest("this gcc cannot build a static PIE")

    def test_the_real_program_runs_and_passes(self):
        out = subprocess.run([str(self.good)], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0)
        self.assertIn("ok alphabetagamma", out.stdout)
        self.assertEqual(E.check_static_pie(self.good.read_bytes()), [])

    def test_a_real_shared_library_with_a_runpath_passes(self):
        src = Path(self.tmp, "lib.c")
        src.write_text("int answer(void) { return 42; }" + chr(10))
        lib = Path(self.tmp, "libanswer.so.1")
        built = subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-Wl,-soname,libanswer.so.1", "-Wl,-rpath,$ORIGIN", str(src), "-o", str(lib)], capture_output=True)
        if built.returncode != 0:
            self.skipTest("this gcc cannot build a shared library")
        data = lib.read_bytes()
        self.assertEqual(E.check_static_pie(data), [], "a real library with RUNPATH $ORIGIN and no interpreter is not a static PIE")
        self.assertEqual(E.static_pie_kind(data, E.read_elf(data)), None)
        # and the real static PIE of this class is still told apart and still found when damaged
        good = self.good.read_bytes()
        self.assertEqual(E.static_pie_kind(good, E.read_elf(good)), "static-pie")

    def test_a_filled_dynamic_section_does_not_start_and_is_found(self):
        data = bytearray(self.good.read_bytes())
        info = E.read_elf(bytes(data))
        seg = next(p for p in info["phdrs"] if p["type"] == E.PT_DYNAMIC)
        data[seg["off"]:seg["off"] + seg["filesz"]] = b"Z" * seg["filesz"]
        bad = Path(self.tmp, "bad")
        bad.write_bytes(bytes(data))
        bad.chmod(0o755)
        run = subprocess.run([str(bad)], capture_output=True)
        self.assertNotEqual(run.returncode, 0, "a static PIE whose .dynamic is overwritten must not run")
        self.assertTrue(E.check_static_pie(bytes(data)))
        self.assertEqual(E.main(["x", "scan", self.tmp]), 1)

    def test_the_repack_script_check_passes_a_sound_bundled_rg_and_stops_a_damaged_one(self):
        """The shell function of repack-appimage.sh --check, run on a fixture AppDir whose `rg` is a real static PIE."""
        if sys.platform != "linux" or os.uname().machine != "x86_64" or not shutil.which("bash") or not shutil.which("python3"):
            self.skipTest("the check runs the bundled rg: x86-64 Linux with bash and python3")
        script = ROOT.parent / "app" / "desktop" / "scripts" / "repack-appimage.sh"
        src = Path(self.tmp, "rg.c")
        src.write_text(chr(10).join(["#include <stdio.h>", "#include <string.h>",
                                     'int main(int c, char **v) { for (int i = 1; i < c; i++) if (!strcmp(v[i], "--version")) { puts("ripgrep 0"); return 0; }',
                                     '  FILE *f = fopen(v[c - 1], "r"); if (!f) return 2; char b[256]; int n = 0; while (fgets(b, 256, f)) if (strstr(b, "needle")) n++;',
                                     '  printf("%d", n); return 0; }', ""]))
        lib = Path(self.tmp, "AppDir-lib")
        rg = lib / "harness/node_modules/@vscode/ripgrep-linux-x64/bin/rg"
        rg.parent.mkdir(parents=True)
        built = subprocess.run(["gcc", "-O2", "-static-pie", str(src), "-o", str(rg)], capture_output=True)
        if built.returncode != 0:
            self.skipTest("this gcc cannot build a static PIE")
        snippet = 'source <(sed -n "/^elf_and_rg_check()/,/^}/p" "$SCRIPT"); elf_and_rg_check'
        env = dict(os.environ, SCRIPT=str(script), LIB=str(lib), HERE=str(script.parent))
        ok = subprocess.run(["bash", "-c", snippet], env=env, capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        self.assertIn("bundled rg runs: ok", ok.stdout)
        data = bytearray(rg.read_bytes())
        info = E.read_elf(bytes(data))
        seg = next(p for p in info["phdrs"] if p["type"] == E.PT_DYNAMIC)
        data[seg["off"]:seg["off"] + seg["filesz"]] = b"Z" * seg["filesz"]
        rg.write_bytes(bytes(data))
        bad = subprocess.run(["bash", "-c", snippet], env=env, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("BROKEN", bad.stdout)
        self.assertIn("does not run", bad.stderr)


if __name__ == "__main__":
    unittest.main()
