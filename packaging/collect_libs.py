#!/usr/bin/env python3
"""Collect the shared libraries an engine binary needs into one folder, ready for packaging/package_engine.py (Linux).

    python packaging/collect_libs.py --binary build/strata --search /usr/local/cuda/lib64 --out stage/ \
        [--also 'libcublasLt.so*'] [--data /opt/rocm/lib/rocblas/library=rocblas/library --data-glob '*gfx1201*']

What it does: runs `ldd` on the binary with LD_LIBRARY_PATH set to --search, copies every dependency that is NOT
supplied by the distribution (anything resolved outside /lib, /lib64, /usr/lib, /usr/lib64 and their x86_64-linux-gnu
folders) into --out under its soname (symlinks are resolved, the real file is copied once per name), repeats for each
copied library (their own dependencies), and adds libraries that are loaded with dlopen and so never show up in ldd
(--also, a glob searched in --search). `--data SRC=DEST` copies a data folder (rocBLAS / hipBLASLt kernel libraries)
to DEST under --out, optionally filtered by --data-glob.

Distribution libraries (libc, libm, libstdc++, ...) are not copied: they are listed in
`SYSTEM-LIBS.txt` beside the output so a missing one on a target PC can be named. The GPU driver's own libraries
(libcuda.so.1, libnvidia-*, libze_intel_gpu, libOpenCL ICDs) are never bundled: they belong to the driver.
A distribution library that is not on every system is copied from the build machine all the same (BUNDLE_FROM_SYSTEM, more
with --bundle-system): libatomic.so.1, which TheRock's librocm_sysdeps_numa.so.1 needs; a container with no libatomic1 package
could not start the AMD engine (boxia, 2026-11-15). Its notice goes into licenses/ (packaging/notices/).
Nothing is downloaded; this only reads files on the build machine.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys


def _copy(src, dst) -> None:
    """shutil.copy2, except onto itself: a library that an earlier binary's pass already put in --out is found there again by ldd (RUNPATH $ORIGIN)
    and is the same file (SameFileError broke the second binary's pass for the AMD packages)."""
    try:
        if os.path.exists(dst) and os.path.samefile(src, dst):
            return
    except OSError:
        pass
    shutil.copy2(src, dst)
from pathlib import Path

SYSTEM_DIRS = ("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")
DRIVER_PATTERNS = ("libcuda.so*", "libnvidia-*", "libnvcuvid*", "libze_intel_gpu*", "libigdrcl*", "libOpenCL.so*",
                   "libGLX*", "libEGL*", "libvulkan*", "libdrm_amdgpu*", "libdrm.so*", "libamdgpu*")
LDD_LINE = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(0x")
#: Distribution libraries that are bundled anyway (by soname glob): not on packaging/binary_deps.py's base list, so a PC without them
#: could not start the engine. Each one has its notice in packaging/notices/<name>.txt, which is copied to <out>/licenses/.
BUNDLE_FROM_SYSTEM = ("libatomic.so.1",)
NOTICES = Path(__file__).resolve().parent / "notices"


def is_system(path: str) -> bool:
    return path.startswith(SYSTEM_DIRS)


def bundled_anyway(name: str, extra: list[str] | tuple[str, ...] = ()) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in (*BUNDLE_FROM_SYSTEM, *extra))


def _owner_package(path: str) -> tuple[str, str] | None:
    """(package, version) of the distribution package that installed ``path`` (dpkg on Debian / Ubuntu, rpm elsewhere), or None."""
    for command, parse in ((["dpkg-query", "-S", path], lambda t: t.split(":", 1)[0].strip()),
                           (["rpm", "-qf", "--qf", "%{NAME}\n", path], lambda t: t.strip())):
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0 and done.stdout.strip():
            package = parse(done.stdout.splitlines()[0])
            ask = ["dpkg-query", "-W", "-f", "${Version}", package] if command[0] == "dpkg-query" else ["rpm", "-q", "--qf", "%{VERSION}-%{RELEASE}", package]
            try:
                version = subprocess.run(ask, capture_output=True, text=True, timeout=30).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                version = ""
            return package, version
    return None


def copy_notice(name: str, out: Path, origin: str = "") -> None:
    """For a distribution library that was bundled: packaging/notices/<soname without .so.N>.txt into <out>/licenses/, followed by where the copy
    came from (the file and the distribution package with its version), and that package's own copyright file and the GPL-3 text from the build
    machine when it has them (/usr/share/doc/<package>/copyright, /usr/share/common-licenses/GPL-3)."""
    stem = name.split(".so", 1)[0]
    source = NOTICES / f"{stem}.txt"
    if not source.is_file():
        print(f"WARNING: {name} is bundled but packaging/notices/{stem}.txt does not exist", file=sys.stderr)
        return
    folder = out / "licenses"
    folder.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    owner = (_owner_package(origin) or _owner_package(str(Path(origin).resolve())) or        # /lib vs /usr/lib on a merged-/usr system
             _owner_package(origin.replace("/lib/", "/usr/lib/", 1) if origin.startswith("/lib/") else origin)) if origin else None
    if origin:
        text += f"\nCopied from {origin}" + (f" (distribution package {owner[0]} {owner[1]})" if owner else "") + " on the build machine.\n"
    (folder / source.name).write_text(text, encoding="utf-8")
    if owner and Path(f"/usr/share/doc/{owner[0]}/copyright").is_file():
        _copy(Path(f"/usr/share/doc/{owner[0]}/copyright"), folder / f"{stem}-copyright.txt")
    if Path("/usr/share/common-licenses/GPL-3").is_file():
        _copy(Path("/usr/share/common-licenses/GPL-3"), folder / "GPL-3.txt")


def is_driver(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in DRIVER_PATTERNS)


GFX = re.compile(r"gfx[0-9a-f]+", re.I)


def arch_wanted(name: str, wanted: str) -> bool:
    """Data files for other GPU architectures are left out; shared ones (no gfx name) stay."""
    if not wanted:
        return True
    named = {m.group(0).lower() for m in GFX.finditer(name)}
    return not named or bool(named & {w.strip().lower() for w in wanted.split(",") if w.strip()})


ELF_MAGIC = bytes([0x7F]) + b"ELF"


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) == ELF_MAGIC
    except OSError:
        return False


def soname(path: Path) -> str | None:
    """The library's own name (DT_SONAME), which is the name its users ask the loader for."""
    done = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True)
    match = re.search(r"\(SONAME\)\s+Library soname: \[([^\]]+)\]", done.stdout)
    return match.group(1) if match else None


def dependencies(binary: Path, search: list[str]) -> tuple[dict[str, str], list[str]]:
    """{soname: resolved path} and the sonames ldd could not find."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(search + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    done = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, env=env, timeout=120)
    found, missing = {}, []
    for line in done.stdout.splitlines():
        if "not found" in line:
            missing.append(line.split("=>")[0].strip())
            continue
        match = LDD_LINE.match(line)
        if match:
            found[match.group(1)] = match.group(2)
    return found, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--search", action="append", default=[], help="a library folder to resolve against (repeatable)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--also", action="append", default=[], help="glob of dlopen'd libraries to add, searched in --search")
    ap.add_argument("--data", action="append", default=[], metavar="SRC=DEST")
    ap.add_argument("--data-glob", action="append", default=[], help="keep only data files matching one of these globs")
    ap.add_argument("--data-arch", default="", help="comma list (gfx1200,gfx1201): of the data files that name a gfx arch keep "
                                                    "only these; files naming none (shared tables) are kept")
    ap.add_argument("--bundle-system", action="append", default=[], metavar="GLOB",
                    help="a distribution library (soname glob) to bundle all the same, beside " + ", ".join(BUNDLE_FROM_SYSTEM))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    system: set[str] = set()
    queue = [a.binary]
    seen = set()
    while queue:
        item = queue.pop()
        if item in seen:
            continue
        seen.add(item)
        found, missing = dependencies(item, a.search)
        for name in missing:
            if not is_driver(name):
                print(f"WARNING: {item.name} needs {name}, which ldd cannot find", file=sys.stderr)
        for name, path in found.items():
            if is_driver(name):
                continue
            if is_system(path) and not bundled_anyway(name, a.bundle_system):
                system.add(name)
                continue
            if name in copied:
                continue
            real = Path(path).resolve()
            _copy(real, a.out / name)                    # the real file under its soname (a package holds no symlinks)
            copied[name] = str(real)
            if is_system(path):
                copy_notice(name, a.out, path)
            queue.append(a.out / name)
    for pattern in a.also:
        hits = [Path(d) / n for d in a.search if os.path.isdir(d) for n in sorted(os.listdir(d)) if fnmatch.fnmatch(n, pattern)]
        if not hits:
            print(f"WARNING: --also {pattern!r} matched nothing in {a.search}", file=sys.stderr)
        for hit in hits:
            if not hit.is_file() or not is_elf(hit):
                continue
            real = hit.resolve()
            if str(real) in copied.values():          # libX.so -> libX.so.1 -> libX.so.1.2.3: one file, one name (the soname)
                continue
            name = soname(real) or hit.name
            if name in copied:
                continue
            _copy(real, a.out / name)
            copied[name] = str(real)
            found, missing = dependencies(a.out / name, a.search)
            for dep in missing:
                if not is_driver(dep):
                    print(f"WARNING: {name} needs {dep}, which ldd cannot find", file=sys.stderr)
            for dep, path in found.items():
                if is_driver(dep) or dep in copied:
                    continue
                if is_system(path) and not bundled_anyway(dep, a.bundle_system):
                    system.add(dep)
                else:
                    _copy(Path(path).resolve(), a.out / dep)
                    copied[dep] = str(Path(path).resolve())
                    if is_system(path):
                        copy_notice(dep, a.out, path)
    for spec in a.data:
        src, _, dest = spec.partition("=")
        if not dest:
            ap.error(f"--data {spec!r} is not SRC=DEST")
        for path in sorted(Path(src).rglob("*")):
            if path.is_file() and (not a.data_glob or any(fnmatch.fnmatch(path.name, g) for g in a.data_glob))                     and arch_wanted(path.name, a.data_arch):
                target = a.out / dest / path.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                _copy(path.resolve(), target)
    (a.out / "SYSTEM-LIBS.txt").write_text("\n".join(sorted(system)) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in a.out.rglob("*") if p.is_file())
    print(f"copied {len(copied)} libraries ({total / (1 << 20):.1f} MiB incl. data) to {a.out}; {len(system)} system libraries left to the OS")
    for name in sorted(copied):
        print(f"  {name} <- {copied[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
