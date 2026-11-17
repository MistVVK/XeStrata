#!/usr/bin/env python3
"""Is every static-PIE program inside an AppImage still the program it was staged as? (#4337)

Why: the bundled ripgrep (`harness/node_modules/@vscode/ripgrep-linux-x64/bin/rg`) is a statically linked PIE. Such a program relocates itself at
start-up by reading its own `.dynamic` section through a fixed offset in its code. linuxdeploy, which Tauri runs over the AppDir, treats every ELF file
as a library to fix up and rewrites its RPATH; for a static PIE that moves `.dynamic` to a new segment and leaves the old place filled with a filler
byte, so the start-up code reads garbage. Reporters on #4337 see the same crash offset in every run (`rg + 0x3ffcf1`) and filler bytes (0x5A) where
`.rela.dyn` and `.dynamic` were. A static PIE whose `.dynamic` is filled does not start: tools/test_appimage_elf.py shows that on a real one.
apps/desktop/scripts/repack-appimage.sh puts the staged bytes back; this file is the check that says whether it did.

    appimage_elf.py scan <folder>                      every static PIE under <folder> must have a sound .dynamic (exit 1 if not)
    appimage_elf.py compare <staged-folder> <folder>   the same files, byte for byte, plus what differs in their ELF headers

What is checked: only static PIE programs. A shared library (.so, Python's and Node's native modules, Pillow, numpy, sharp) is not one, however it is
named: it has no entry point, or a DT_SONAME and no DF_1_PIE flag, and its RPATH/RUNPATH is normal. A static PIE is an ET_DYN with no interpreter, an entry
point, and DF_1_PIE in DT_FLAGS_1; one whose .dynamic cannot be read at all (overwritten) is reported as damaged, since the overwrite takes the flags too.

Standard library only; reads ELF64 little endian (x86-64 and arm64), anything else is ignored.
"""
import os
import struct
import sys

ET_DYN = 3
PT_LOAD, PT_DYNAMIC, PT_INTERP = 1, 2, 3
DT_NULL, DT_NEEDED, DT_RELA, DT_RELASZ, DT_RELAENT, DT_SONAME, DT_RPATH, DT_RUNPATH = 0, 1, 7, 8, 9, 14, 15, 29
DT_FLAGS_1, DF_1_PIE = 0x6FFFFFFB, 0x08000000
FILLERS = (0x5A, 0x58)           # 'Z' (what the reports found) and 'X' (what patchelf documents)
RUN = 128                        # that many equal filler bytes in a row is not code or data


def _known_tag(tag):
    """The dynamic tags the ELF specification and the GNU and processor ranges define: anything else is garbage."""
    return 0 <= tag <= 0x22 or 0x6000000D <= tag <= 0x6FFFFFFF or 0x70000000 <= tag <= 0x7FFFFFFF


def read_elf(data):
    """The facts about an ELF64 little-endian file, or None when it is not one."""
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return None
    e_type = struct.unpack_from("<H", data, 16)[0]
    e_entry = struct.unpack_from("<Q", data, 0x18)[0]
    phoff = struct.unpack_from("<Q", data, 0x20)[0]
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    if phentsize < 56 or phoff + phentsize * phnum > len(data):
        return {"type": e_type, "entry": e_entry, "phdrs": [], "broken": "the program header table is outside the file"}
    phdrs = []
    for i in range(phnum):
        p_type, flags, off, vaddr, _paddr, filesz, memsz, align = struct.unpack_from("<IIQQQQQQ", data, phoff + i * phentsize)
        phdrs.append({"type": p_type, "flags": flags, "off": off, "vaddr": vaddr, "filesz": filesz, "memsz": memsz, "align": align})
    return {"type": e_type, "entry": e_entry, "phdrs": phdrs, "broken": None}


def is_pie_candidate(info):
    """An ET_DYN with no interpreter, a dynamic segment and an entry point: what a static PIE looks like from outside the dynamic section.
    A shared library normally has no entry point (e_entry 0) and is not a candidate; the few that do (the C library itself, the dynamic loader)
    have a readable dynamic section and are told apart by it (static_pie_kind). The test uses nothing the damage we look for can erase."""
    return bool(info) and info["type"] == ET_DYN and not info["broken"] and info.get("entry", 0) != 0         and not any(p["type"] == PT_INTERP for p in info["phdrs"]) and any(p["type"] == PT_DYNAMIC for p in info["phdrs"])


def static_pie_kind(data, info):
    """'static-pie' for a sound static PIE (DT_FLAGS_1 has DF_1_PIE, no DT_SONAME), 'damaged' for a candidate whose dynamic section cannot be read
    (a static PIE with an overwritten .dynamic loses its flags with it, so unreadable is treated as damaged), None for everything else:
    shared libraries, which carry a DT_SONAME and no DF_1_PIE, and programs that are not static PIEs."""
    if not is_pie_candidate(info):
        return None
    entries, why = dynamic_entries(data, info)
    if why or any(not _known_tag(t) for t, _ in entries):
        return "damaged"
    tags = {t: v for t, v in entries}
    if DT_SONAME in tags:
        return None
    return "static-pie" if tags.get(DT_FLAGS_1, 0) & DF_1_PIE else None


def is_static_pie(info):
    """Kept for callers: a candidate. check_static_pie decides with static_pie_kind."""
    return is_pie_candidate(info)


def _mapped(info, vaddr, size):
    """The file offset of [vaddr, vaddr+size) when one PT_LOAD maps all of it from the file, else None."""
    for p in info["phdrs"]:
        if p["type"] == PT_LOAD and p["vaddr"] <= vaddr and vaddr + size <= p["vaddr"] + p["filesz"]:
            return p["off"] + (vaddr - p["vaddr"])
    return None


def dynamic_entries(data, info):
    """(tag, value) pairs of the PT_DYNAMIC segment up to DT_NULL, and the problem if it does not end properly."""
    seg = next(p for p in info["phdrs"] if p["type"] == PT_DYNAMIC)
    if seg["off"] + seg["filesz"] > len(data):
        return [], "the dynamic segment is outside the file"
    out = []
    for at in range(seg["off"], seg["off"] + seg["filesz"] - 15, 16):
        tag, val = struct.unpack_from("<qQ", data, at)
        if tag == DT_NULL:
            return out, None
        out.append((tag, val))
    return out, "the dynamic segment does not end with DT_NULL"


def _longest_filler(data, start, end):
    best = run = 0
    prev = None
    for b in data[start:end]:
        run = run + 1 if b == prev and b in FILLERS else (1 if b in FILLERS else 0)
        prev = b
        best = max(best, run)
    return best


def check_static_pie(data):
    """The problems of one file (a list of sentences); empty for a sound static PIE and for anything that is not one."""
    info = read_elf(data)
    if static_pie_kind(data, info) is None:
        return []
    problems = []
    seg = next(p for p in info["phdrs"] if p["type"] == PT_DYNAMIC)
    if _mapped(info, seg["vaddr"], max(seg["filesz"], 1)) is None:
        problems.append("the dynamic segment is not inside any loaded segment")
    entries, why = dynamic_entries(data, info)
    if why:
        problems.append(why)
    bad = [t for t, _ in entries if not _known_tag(t)]
    if bad:
        problems.append(f"{len(bad)} dynamic entries have tags that do not exist (first {bad[0]:#x}): the section is overwritten")
    if any(t in (DT_RPATH, DT_RUNPATH) for t, _ in entries):
        problems.append("a static PIE has an RPATH or RUNPATH: something rewrote it after it was linked")
    values = dict(entries)
    if DT_RELA in values:
        size = values.get(DT_RELASZ, 0)
        at = _mapped(info, values[DT_RELA], max(size, 1))
        if size == 0 or at is None:
            problems.append("the relocation table named by DT_RELA is empty or outside the loaded segments")
        elif _longest_filler(data, at, at + size) >= min(RUN, size):
            problems.append("the relocation table is filled with one byte")
    for p in info["phdrs"]:
        if p["type"] == PT_LOAD and _longest_filler(data, p["off"], p["off"] + p["filesz"]) >= RUN:
            problems.append(f"{RUN} or more filler bytes (0x5A or 0x58) in a row inside a loaded segment")
            break
    return problems


def elf_differences(staged, packaged):
    """What differs between two builds of one program in the parts that matter for start-up: for the report."""
    a, b = read_elf(staged), read_elf(packaged)
    if not a or not b:
        return [] if staged == packaged else ["one of the two is not an ELF64 file"]
    out = []
    ta, tb = [p["type"] for p in a["phdrs"]], [p["type"] for p in b["phdrs"]]
    if ta != tb:
        out.append(f"program headers {len(ta)} -> {len(tb)} ({ta.count(PT_LOAD)} -> {tb.count(PT_LOAD)} loaded segments)")
    if is_pie_candidate(a) and is_pie_candidate(b):
        da, _ = dynamic_entries(staged, a)
        db, _ = dynamic_entries(packaged, b)
        if da != db:
            out.append(f"dynamic section: {len(da)} -> {len(db)} entries" + ("; an RPATH/RUNPATH was added" if any(t in (DT_RPATH, DT_RUNPATH) for t, _ in db) and not any(t in (DT_RPATH, DT_RUNPATH) for t, _ in da) else ""))
    return out


def _files(folder):
    for base, dirs, names in os.walk(folder):
        dirs.sort()
        for name in sorted(names):
            path = os.path.join(base, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            yield path


def scan(folder):
    """{relative path: problems} for every static PIE under `folder` that has any."""
    found = {}
    for path in _files(folder):
        try:
            with open(path, "rb") as f:
                head = f.read(4)
                if head != b"\x7fELF":
                    continue
                data = head + f.read()
        except OSError:
            continue
        problems = check_static_pie(data)
        if problems:
            found[os.path.relpath(path, folder)] = problems
    return found


def compare(staged_folder, folder):
    """{relative path: sentences} for every staged ELF file that is missing or differs in `folder`."""
    found = {}
    for path in _files(staged_folder):
        rel = os.path.relpath(path, staged_folder)
        with open(path, "rb") as f:
            want = f.read()
        if want[:4] != b"\x7fELF":
            continue
        other = os.path.join(folder, rel)
        if not os.path.isfile(other):
            found[rel] = ["missing"]
            continue
        with open(other, "rb") as f:
            have = f.read()
        if have != want:
            found[rel] = ["bytes differ"] + elf_differences(want, have)
    return found


def main(argv):
    if len(argv) >= 3 and argv[1] == "scan":
        bad = scan(argv[2])
        for rel, problems in bad.items():
            print(f"BROKEN {rel}: " + "; ".join(problems))
        print(f"{len(bad)} static PIE file(s) with a damaged dynamic section")
        return 1 if bad else 0
    if len(argv) >= 4 and argv[1] == "compare":
        bad = compare(argv[2], argv[3])
        for rel, why in bad.items():
            print(f"DIFFERS {rel}: " + "; ".join(why))
        print(f"{len(bad)} ELF file(s) differ from what was staged")
        return 1 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
