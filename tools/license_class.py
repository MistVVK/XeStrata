# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/license_class.py [--tag TAG] [--fix] [FILE...] - check each file's SPDX header against where its code
comes from (AGENTS.md, Licenses).  Run from the repository root; without files, every tracked file.

A file is upstream's when upstream Strata's integrated tag (v + STRATA_UPSTREAM_VERSION in CMakeLists.txt, or --tag)
has it at the same path, has a file of the same content elsewhere (a rename), or the file names Strata's copyright and
upstream had it at that path before the tag (the content it last had there); not when it has none of upstream's
lines (apart from blank and short ones): XeStrata wrote it anew.  Apart from the SPDX header lines XeStrata adds, the
same content as upstream's is unchanged, any other is mixed:

  unchanged  upstream's copyright holders only, upstream's license (MIT, or the SPDX identifier upstream's file has)
  mixed      those holders and XeStrata's, LGPL-3.0-or-later AND MIT
  own        a file that is not upstream's: XeStrata's copyright, LGPL-3.0-or-later; AND that project's license when
             it also names another permissive project's copyright (code transcribed from ggml: MIT; from vLLM-Omni:
             Apache-2.0)

A file without a header of its own is checked against the last REUSE.toml annotation that covers it.  --fix rewrites
the headers that differ (only the holder and identifier lines); REUSE.toml and a file without a header are reported
for a hand edit.  Exits 1 when anything differs and is not fixed; a tag missing from the clone is skipped with a note.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

XE = "2026 MistVVK and the XeStrata contributors"
STRATA = "2026 Niko1221 and the Strata contributors"
# copyright holders whose code is under a permissive license XeStrata's files may carry (AGENTS.md, Licenses);
# all of them MIT (Expat)
PERMISSIVE = ("Niko1221 and the Strata contributors", "The ggml authors", "Debian's rocm-llvm packaging",
              "Peasant Smith and the Project Maya contributors", "Sahand Ahmadian", "Jonseed", "ZhengPeng",
              "Ostris, LLC")
# the same for the projects under another permissive license: a copyright holder's words -> its license
PERMISSIVE_OTHER = {"contributors to the vLLM-Omni project": "Apache-2.0", "The HuggingFace Inc. team": "Apache-2.0",
                    "SageAttention team": "Apache-2.0", "(comfy-kitchen)": "Apache-2.0",
                    "the comfyui_controlnet_aux contributors": "Apache-2.0",
                    "the ComfyUI-SeedVR2_VideoUpscaler contributors": "Apache-2.0",
                    "the YASAKANI authors": "Apache-2.0",
                    "Secret Labs AB": "MIT-CMU", "Fredrik Lundh and contributors": "MIT-CMU",
                    "Jeffrey A. Clark and contributors": "MIT-CMU"}
OWN = "LGPL-3.0-or-later"
MIXED = "LGPL-3.0-or-later AND MIT"
# the files the classes do not cover: other projects' folders, license texts, the notices
SKIP = re.compile(r"^(third_party/|LICENSES/|COPYING|NOTICE$)")
HEAD_LINES = 40  # an SPDX line further down is an example in the text, not the header
SPDX = re.compile(rb"^(?P<pre>.*?)SPDX-(?P<key>FileCopyrightText|License-Identifier): (?P<val>.*?)"
                  rb"(?P<post>\s*(\*/|-->)?\s*)$")


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def header(lines: list[bytes]) -> list[int]:
    """The indexes of the SPDX lines at the top of a file."""
    return [i for i, ln in enumerate(lines[:HEAD_LINES]) if SPDX.match(ln.rstrip(b"\r\n"))]


def strip(data: bytes) -> bytes:
    """The content without the SPDX lines at its top, and without the <!-- --> that held only them."""
    if b"\0" in data[:8192]:
        return data
    lines = data.splitlines(keepends=True)
    idx = set(header(lines))
    if not idx:
        return data
    out = [ln for i, ln in enumerate(lines) if i not in idx]
    first = min(idx)
    if 0 < first < len(out) and out[first - 1].strip() == b"<!--" and out[first].strip() == b"-->":
        del out[first - 1:first + 1]
    return b"".join(out)


def holders_and_id(data: bytes) -> tuple[list[str], str | None]:
    hs, ident = [], None
    for ln in data.splitlines()[:HEAD_LINES]:
        m = SPDX.match(ln)
        if not m:
            continue
        val = m["val"].decode()
        if m["key"] == b"FileCopyrightText":
            hs.append(val)
        else:
            ident = val
    return hs, ident


class Upstream:
    def __init__(self, tag: str):
        self.tag = tag
        tree = git("ls-tree", "-r", "-z", tag).split(b"\0")
        self.blob = {}
        for e in tree:
            if e:
                meta, path = e.split(b"\t", 1)
                if meta.split()[1] == b"blob":
                    self.blob[path.decode()] = meta.split()[2].decode()
        self.data = self._cat(list(self.blob.values()))
        self.by_hash: dict[str, str] = {}
        for p, b in self.blob.items():
            self.by_hash.setdefault(hashlib.sha256(strip(self.data[b])).hexdigest(), p)

    @staticmethod
    def _cat(objs: list[str]) -> dict[str, bytes]:
        out = subprocess.run(["git", "cat-file", "--batch"], input="\n".join(objs).encode() + b"\n",
                             check=True, capture_output=True).stdout
        res, pos = {}, 0
        for o in objs:
            nl = out.index(b"\n", pos)
            size = int(out[pos:nl].split()[2])
            res[o] = out[nl + 1:nl + 1 + size]
            pos = nl + 1 + size + 1
        return res

    def base(self, path: str, data: bytes) -> bytes | None:
        """upstream's content of the file, or None when the file is not upstream's."""
        if path in self.blob:
            return self.data[self.blob[path]]
        p = self.by_hash.get(hashlib.sha256(strip(data)).hexdigest())
        if p:
            return self.data[self.blob[p]]
        if STRATA not in holders_and_id(data)[0]:
            return None
        # deleted upstream before the tag: the content of the last commit that had it
        last = git("log", "-1", "--format=%H", self.tag, "--", path).decode().strip()
        for rev in (last, last + "^") if last else ():
            r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True)
            if r.returncode == 0:
                return r.stdout
        return b""  # Strata's copyright, no upstream file to compare: mixed


def shares_lines(data: bytes, base: bytes) -> bool:
    """Whether the file has a line of upstream's, apart from blank and short ones (a brace, an else)."""
    if not base or b"\0" in data[:8192]:
        return True
    old = {ln.strip() for ln in strip(base).splitlines() if len(ln.strip()) > 3}
    return any(ln.strip() in old for ln in strip(data).splitlines() if len(ln.strip()) > 3)


def expected(path: str, data: bytes, hs: list[str], up: Upstream) -> tuple[str, list[str], str]:
    """(class, holders, identifier) the file should carry, given the holders it names (its header or REUSE.toml).
    An own file keeps the holders it names, XeStrata's among them: what it transcribed is not checked here."""
    base = up.base(path, data)
    if base is not None and not shares_lines(data, base):
        base = None  # written anew at upstream's path (AGENTS.md)
    if base is None:
        perm = [h for h in hs if h != XE and any(k in h for k in PERMISSIVE)]
        other = sorted({lic for h in hs for k, lic in PERMISSIVE_OTHER.items() if k in h})
        ident = " AND ".join([OWN] + (["MIT"] if perm else []) + other)
        return ("own", hs if XE in hs else hs + [XE], ident)
    bhs, bid = holders_and_id(base)
    others = [h for h in hs if h != XE] or bhs or [STRATA]
    if base and strip(data) == strip(base):
        return ("unchanged", others, bid or "MIT")
    return ("mixed", others + [XE], MIXED)


def reuse_rules() -> list[tuple[re.Pattern[str], list[str], str]]:
    rules = []
    for a in tomllib.loads(Path("REUSE.toml").read_text()).get("annotations", []):
        paths = a["path"] if isinstance(a["path"], list) else [a["path"]]
        hs = a["SPDX-FileCopyrightText"]
        hs = hs if isinstance(hs, list) else [hs]
        for p in paths:
            rx = re.escape(p).replace(r"\*\*", "\0").replace(r"\*", "[^/]*").replace("\0", ".*")
            rules.append((re.compile(rx + "$"), hs, a["SPDX-License-Identifier"]))
    return rules


def fix(path: Path, data: bytes, hs: list[str], ident: str) -> None:
    lines = data.splitlines(keepends=True)
    idx = header(lines)
    have = holders_and_id(data)[0]
    hs = [h for h in have if h in hs] + [h for h in hs if h not in have]  # keep the order the file has
    m0 = SPDX.match(lines[idx[0]].rstrip(b"\r\n"))
    assert m0
    pre, post = m0["pre"], m0["post"]
    eol = lines[idx[0]][len(lines[idx[0]].rstrip(b"\r\n")):]
    # the keys are split so that reuse does not read these lines as this file's own header
    new = [pre + b"SPDX-" + b"FileCopyrightText: " + h.encode() + post + eol for h in hs]
    new.append(pre + b"SPDX-" + b"License-Identifier: " + ident.encode() + post + eol)
    path.write_bytes(b"".join(lines[:idx[0]] + new + lines[idx[-1] + 1:]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", help="upstream's integrated tag (default: v + STRATA_UPSTREAM_VERSION)")
    ap.add_argument("--fix", action="store_true", help="rewrite the headers that differ")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    tag = a.tag
    if not tag:
        m = re.search(r'STRATA_UPSTREAM_VERSION "([^"]+)"', Path("CMakeLists.txt").read_text())
        tag = "v" + m[1] if m else ""
    if subprocess.run(["git", "rev-parse", "-q", "--verify", f"{tag}^{{commit}}"], capture_output=True).returncode:
        print(f"license classes: skipped (upstream's tag {tag or '?'} is not in this clone: git fetch upstream --tags)")
        return 0
    files = a.files or git("ls-files", "-z").decode().split("\0")
    up = Upstream(tag)
    rules = reuse_rules()
    status = 0
    for f in files:
        p = Path(f)
        if not f or SKIP.match(f) or not p.is_file():
            continue
        data = p.read_bytes()
        hs, ident = holders_and_id(data)
        if ident is None:
            hit = [(h, i) for rx, h, i in rules if rx.match(f)]
            if not hit:
                continue  # reuse lint reports it
            hs, ident = hit[-1]
            cls, want_hs, want_id = expected(f, data, hs, up)
            if sorted(hs) != sorted(want_hs) or ident != want_id:
                print(f"{f}: {cls}, REUSE.toml gives {hs} {ident}; should be {want_hs} {want_id}")
                status = 1
            continue
        cls, want_hs, want_id = expected(f, data, hs, up)
        if sorted(hs) == sorted(want_hs) and ident == want_id:
            continue
        if a.fix:
            fix(p, data, want_hs, want_id)
            print(f"{f}: {cls}, fixed")
        else:
            print(f"{f}: {cls}, has {hs} {ident}; should be {want_hs} {want_id}")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
