#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Whether each NVIDIA and AMD architecture of an engine got its own device code (issue #7).

    tools/package/check_device_code.py PACKAGE.deb|PACKAGE.rpm|ENGINE

Every source with device code carries a mark made when it is compiled for an architecture
(include/strata/kernels/device_arch_tag.hpp: "xestrata-device-arch:<architecture>:<source>").  The name a device image
is filed under (an AMD code object's bundle id, an NVIDIA cubin's arch) comes from the link and stays right when the
link is handed another architecture's code, so the marks are compared with it: every architecture must hold the marks
of the same sources, each one its own.  Within a family too (gfx1100 against gfx1151, sm_86 against sm_89), which the
instructions alone cannot tell apart.

Needs clang-offload-extract and clang-offload-bundler (intel/llvm's: .tools/intel-llvm-contrib/install/bin,
.tools/intel-llvm/install/bin, or XESTRATA_LLVM_BIN) and, for NVIDIA, NVIDIA's cuobjdump (a developer's tool, not a
build or test dependency).  No GPU.  Exit 0 when every architecture holds the same sources' marks, all its own.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARK = re.compile(rb"xestrata-device-arch:([A-Za-z0-9_]+):([A-Za-z0-9_]+)")
FATBIN = bytes.fromhex("50ed55ba")
BUNDLE = b"__CLANG_OFFLOAD_BUNDLE__"


def tool(name: str) -> str:
    dirs = [os.environ.get("XESTRATA_LLVM_BIN", ""), str(ROOT / ".tools/intel-llvm-contrib/install/bin"),
            str(ROOT / ".tools/intel-llvm/install/bin")]
    for d in dirs:
        if d and (Path(d) / name).is_file():
            return str(Path(d) / name)
    found = shutil.which(name)
    if not found:
        sys.exit(f"{name} not found (set XESTRATA_LLVM_BIN to intel/llvm's bin folder)")
    return found


def engine_of(path: Path, tmp: Path) -> Path:
    """The xestrata-engine executable: the path itself, or the one inside a deb or rpm."""
    if path.suffix not in (".deb", ".rpm"):
        return path
    root = tmp / "root"
    root.mkdir()
    if path.suffix == ".deb":
        subprocess.run(["dpkg-deb", "-x", str(path), str(root)], check=True)
    else:
        rpm = subprocess.run(["rpm2cpio", str(path)], check=True, capture_output=True).stdout
        subprocess.run(["cpio", "-idm", "--quiet"], input=rpm, cwd=root, check=True)
    found = [p for p in root.rglob("xestrata-engine") if p.is_file() and p.parent.name == "engine"]
    if not found:
        sys.exit(f"no engine/xestrata-engine in {path}")
    return found[0]


def amd_entries(image: Path, tmp: Path, bundler: str):
    """(architecture, code object bytes) for each hipv4 entry of an AMD image."""
    ids = subprocess.run([bundler, "-list", "-type=o", f"-input={image}"], capture_output=True, text=True).stdout
    for bid in ids.split():
        if not bid.startswith("hipv4-"):
            continue
        out = tmp / "co.o"
        subprocess.run([bundler, "-unbundle", "-type=o", f"-input={image}", f"-targets={bid}", f"-output={out}"],
                       check=True, capture_output=True)
        yield bid.rsplit("-", 1)[-1], out.read_bytes()


def nvidia_entries(image: Path, cuobjdump: str):
    """(architecture as its mark spells it, the marks) for an NVIDIA fatbin: the cubin's arch, and the marks read from
    the PTX's byte arrays (the cubin may be compressed)."""
    arch = re.findall(r"sm_(\d+)", subprocess.run([cuobjdump, "-lelf", str(image)], capture_output=True,
                                                  text=True).stdout)
    ptx = subprocess.run([cuobjdump, "-ptx", str(image)], capture_output=True, text=True).stdout
    marks = []
    for m in re.finditer(r"\.b8 [^\[]*device_arch_tag[^\[]*\[\d+\] = \{([0-9, ]+)\}", ptx):
        marks.append(bytes(int(b) for b in m.group(1).split(",")))
    for a in sorted(set(arch)):
        yield f"sm_{int(a) * 10}", marks


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    extract, bundler = tool("clang-offload-extract"), tool("clang-offload-bundler")
    cuobjdump = shutil.which("cuobjdump")
    marks: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))   # label -> source -> archs marked
    nvidia_seen = False
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        engine = engine_of(Path(sys.argv[1]), tmp)
        imgs = tmp / "img"
        imgs.mkdir()
        subprocess.run([extract, "-q", f"--stem={imgs / 't'}", str(engine)], check=True, capture_output=True)
        for image in sorted(imgs.iterdir()):
            head = image.read_bytes()[:32]
            if head.startswith(BUNDLE):
                for label, code in amd_entries(image, tmp, bundler):
                    for m in MARK.finditer(code):
                        marks[label][m.group(2).decode()].add(m.group(1).decode())
            elif head.startswith(FATBIN):
                nvidia_seen = True
                if cuobjdump is None:
                    continue
                for label, found in nvidia_entries(image, cuobjdump):
                    for raw in found:
                        hit = MARK.search(raw)
                        if hit:
                            marks[label][hit.group(2).decode()].add(hit.group(1).decode())
    if nvidia_seen and cuobjdump is None:
        print("NVIDIA: not checked (no cuobjdump)")
    bad = 0
    families = defaultdict(list)
    for label in sorted(marks):
        families["amd" if label.startswith("gfx") else "nvidia"].append(label)
    for fam, labels in families.items():
        everyone = set().union(*(set(marks[lb]) for lb in labels))
        for label in labels:
            wrong = {src: archs for src, archs in marks[label].items() if archs != {label}}
            missing = everyone - set(marks[label])
            ok = not wrong and not missing
            bad += not ok
            print(f"{label:8s} {len(marks[label]):3d} sources  {'ok' if ok else 'WRONG'}")
            for src, archs in sorted(wrong.items()):
                print(f"         {src}: compiled for {', '.join(sorted(archs))}")
            for src in sorted(missing):
                print(f"         {src}: no mark (other architectures have one)")
    if not marks:
        print("no marks found: an engine built before include/strata/kernels/device_arch_tag.hpp, or no NVIDIA "
              "or AMD code")
        return 1
    print("ALL OK" if bad == 0 else f"WRONG {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
