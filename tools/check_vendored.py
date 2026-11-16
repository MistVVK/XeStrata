#!/usr/bin/env python3
"""Verify third_party/strata-ggml, strata-gguf-py, strata-llama and stb against their manifests (UPSTREAM.md).   python tools/check_vendored.py [--upstream <llama.cpp dir>]

The vendored ggml / ggml-cuda sources are frozen at the llama.cpp commit named in UPSTREAM.md.  The manifest lists every file with its
kind and sha256 (text files are hashed with LF line ends, so a Windows checkout agrees with a Linux one):

  upstream   identical to the file at the pinned commit
  edited     a Strata edit of an upstream file (the edit is described in UPSTREAM.md); the row also holds the upstream sha256
  added      a file that is Strata's own (not in llama.cpp)
  copy       ggml/src/ggml-cuda-tq/X: a verbatim copy of its pristine counterpart (ggml-cuda/X; ggml-cuda-tq/include/X is include/X); the row holds the
             counterpart's hash.  The trellis (Gyro) MMQ build compiles its headers from this folder, which is first on strata_mmq's include path,
             so a header that includes mmq.cuh by name has to sit beside the patched mmq.cuh
  patched    the same, with additions (checked here: every line of the counterpart is still there, in order; only lines were added)

Checks: every listed file exists and has the listed hash; no file of the tree is missing from the list; the pin equals
third_party/ggml/VERSION.txt; the LICENSE is there; every `edited` row names its upstream hash.  With --upstream <dir> the upstream
hashes are also checked against that checkout (the files at the pin, e.g. `git archive` of the commit): `upstream` rows must equal the file
there, `edited` rows must carry its hash.  Standard library only.  Exit 0 clean, 1 on any problem.
"""
import argparse
import hashlib
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TREE = os.path.join("third_party", "strata-ggml")
IDENTITY = ()      # a tree that keeps llama.cpp's layout and has no file under another name (third_party/strata-llama): an empty map, every file `upstream`
GGML_RENAMED = {}   # (no file of third_party/strata-ggml has another name upstream: the quant header is ours since 0.2.16, include/strata/quant)
# every vendored tree: (folder, maps a path in the tree to the path in the llama.cpp checkout at the pin)
GGUF_PY_MAP = (("gguf/", "gguf-py/gguf/"), ("converter/conversion/", "conversion/"), ("converter/convert_hf_to_gguf.py", "convert_hf_to_gguf.py"),
               ("converter/LICENSE", "LICENSE"), ("LICENSE", "gguf-py/LICENSE"), ("pyproject.toml", "gguf-py/pyproject.toml"))
STB_MAP = (("stb_image.h", "vendor/stb/stb_image.h"),)   # third_party/stb: the one header, from llama.cpp's vendor/stb
BEGIN, END = "<!-- vendored-files:begin -->", "<!-- vendored-files:end -->"
NOT_LISTED = {"UPSTREAM.md"}          # the manifest itself
TEXT_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".txt", ".cmake", ".in", ".md", ".inc", ".py", ".toml", ".typed"}


def sha(path):
    with open(path, "rb") as f:
        data = f.read()
    if os.path.splitext(path)[1].lower() in TEXT_SUFFIXES or os.path.basename(path) == "LICENSE":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def counterpart(rel):
    if rel.startswith("ggml/src/ggml-cuda-tq/include/"):
        return "ggml/include/" + rel[len("ggml/src/ggml-cuda-tq/include/"):]
    if rel.startswith("ggml/src/ggml-cuda-tq/") and "/template-instances/" not in rel:
        return "ggml/src/ggml-cuda/" + rel[len("ggml/src/ggml-cuda-tq/"):]
    return None


def only_added(base, patched):
    """True when `patched` is `base` with lines inserted (every base line occurs in patched, in order)."""
    a = open(base, encoding="utf-8").read().replace(chr(13) + chr(10), chr(10)).split(chr(10))
    b = open(patched, encoding="utf-8").read().replace(chr(13) + chr(10), chr(10)).split(chr(10))
    j = 0
    for line in a:
        while j < len(b) and b[j] != line:
            j += 1
        if j == len(b):
            return False
        j += 1
    return True


def parse(manifest):
    text = open(manifest, encoding="utf-8").read()
    if BEGIN not in text or END not in text:
        raise SystemExit("UPSTREAM.md: the %s block is missing" % BEGIN)
    block = text.split(BEGIN, 1)[1].split(END, 1)[0]
    rows = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("```") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise SystemExit("UPSTREAM.md: bad row: %r" % line)
        kind, h, up, rel = parts[0], parts[1], parts[2], " ".join(parts[3:])
        rows.append((kind, h, up, rel))
    m = re.search(r"^pin:\s*([0-9a-f]{40})\s*$", text, re.M)
    return rows, (m.group(1) if m else None)


def upstream_rel(rel, upmap):
    """The path of a tree file in the llama.cpp checkout (`upmap` None: the tree keeps llama.cpp's layout)."""
    if upmap is None:
        return GGML_RENAMED.get(rel, rel)
    for here, there in upmap:
        if here.endswith("/") and rel.startswith(here):
            return there + rel[len(here):]
        if rel == here:
            return there
    return rel


def check_tree(root, tree_rel, upmap, upstream, problems):
    """Check one vendored tree against its UPSTREAM.md; returns the one-line summary."""
    tree = os.path.join(root, tree_rel)
    rows, pin = parse(os.path.join(tree, "UPSTREAM.md"))
    ver = os.path.join(root, "third_party", "ggml", "VERSION.txt")
    if os.path.exists(ver):
        vpin = open(ver, encoding="utf-8").read().split()[0]
        if vpin != pin:
            problems.append("%s: pin %s in UPSTREAM.md differs from third_party/ggml/VERSION.txt (%s)" % (tree_rel, pin, vpin))
    if not os.path.exists(os.path.join(tree, "LICENSE")):
        problems.append("%s: LICENSE is missing" % tree_rel)
    listed = set()
    for kind, h, up, rel in rows:
        if kind not in ("upstream", "edited", "added", "copy", "patched"):
            problems.append("%s: unknown kind %r" % (rel, kind))
            continue
        listed.add(rel)
        p = os.path.join(tree, rel)
        if not os.path.isfile(p):
            problems.append("%s: listed but missing" % rel)
            continue
        if sha(p) != h:
            problems.append("%s: sha256 differs from the manifest (%s)" % (rel, kind))
        if kind == "upstream" and up != h:
            problems.append("%s: an `upstream` row must carry its own hash as the upstream hash" % rel)
        if kind == "edited" and (up == h or not re.fullmatch(r"[0-9a-f]{64}", up)):
            problems.append("%s: an `edited` row needs the upstream hash, different from the file's" % rel)
        if kind == "added" and up != "-":
            problems.append("%s: an `added` row has `-` as the upstream hash" % rel)
        if kind in ("copy", "patched"):
            cp = counterpart(rel)
            cpp = os.path.join(tree, cp) if cp else None
            if not cpp or not os.path.isfile(cpp):
                problems.append("%s: no counterpart in the tree for a `%s` row" % (rel, kind))
            else:
                if sha(cpp) != up:
                    problems.append("%s: its counterpart %s differs from the hash in the row" % (rel, cp))
                if kind == "copy" and up != h:
                    problems.append("%s: a `copy` must equal its counterpart" % rel)
                if kind == "patched" and (up == h or not only_added(cpp, p)):
                    problems.append("%s: a `patched` copy must differ from %s by added lines only" % (rel, cp))
        if upstream and kind in ("upstream", "edited", "copy", "patched"):
            q = os.path.join(upstream, counterpart(rel) if kind in ("copy", "patched") else upstream_rel(rel, upmap))
            if not os.path.isfile(q):
                problems.append("%s: not in the upstream tree %s" % (rel, upstream))
            elif sha(q) != up:
                problems.append("%s: upstream hash in the manifest differs from %s" % (rel, q))
    for d, dirs, files in os.walk(tree):
        dirs[:] = [x for x in dirs if x != "__pycache__"]       # bytecode written by an import of the vendored package is not a file of the tree
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), tree).replace(os.sep, "/")
            if rel in NOT_LISTED or (rel == "LICENSE" and upmap is None):
                continue
            if rel not in listed:
                problems.append("%s: in the tree but not in UPSTREAM.md" % rel)
    n = {k: sum(1 for r in rows if r[0] == k) for k in ("upstream", "edited", "added", "copy", "patched")}
    return "%s: %d files (%d upstream, %d edited, %d added, %d copy, %d patched), pin %s" % (
        tree_rel.replace(os.sep, "/"), len(rows), n["upstream"], n["edited"], n["added"], n["copy"], n["patched"], (pin or "?")[:9])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", help="a llama.cpp tree at the pinned commit (file paths as in llama.cpp; third_party/strata-gguf-py maps as in its UPSTREAM.md)")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args(argv)
    problems = []
    summary = [check_tree(a.root, TREE, None, a.upstream, problems),
               check_tree(a.root, os.path.join("third_party", "strata-gguf-py"), GGUF_PY_MAP, a.upstream, problems),
               check_tree(a.root, os.path.join("third_party", "strata-llama"), IDENTITY, a.upstream, problems),
               check_tree(a.root, os.path.join("third_party", "stb"), STB_MAP, a.upstream, problems)]
    for p in problems:
        print("check_vendored: " + p)
    if problems:
        print("check_vendored: %d problem(s)" % len(problems))
        return 1
    print("check_vendored: ok - " + "; ".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
