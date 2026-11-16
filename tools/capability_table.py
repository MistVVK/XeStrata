#!/usr/bin/env python3
"""The capability table: what each model family runs on, per kind of hardware.

    python tools/capability_table.py --check     fail when a page or the table disagrees with the data
    python tools/capability_table.py --write     rebuild docs/CAPABILITIES.md and the summaries in the other pages

Inputs: docs/data/capabilities.json (what the docs commit to) and docs/data/release.json (the packages in the release).  The
"packaged" column is read from the release's package names, never typed.  Checks:
  * a family marked "yes" or "untested" on some hardware has a package for it in the release, and one marked "no" has none;
  * every package of a family lands in a column of the table;
  * docs/CAPABILITIES.md and every `<!-- cap:... -->` block in AI_SETUP and AI_REFERENCE are what the
    generator would write (so a page cannot keep a sentence the table contradicts).
Block syntax in a page:  <!-- cap:summary:FAMILY -->text<!-- /cap -->   <!-- cap:cell:FAMILY:COLUMN -->text<!-- /cap -->
Standard library only.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doc_pages  # noqa: E402

ROOT = doc_pages.ROOT
DATA = os.path.join(ROOT, "docs", "data", "capabilities.json")
RELEASE = os.path.join(ROOT, "docs", "data", "release.json")
OUT = "docs/CAPABILITIES.md"
PAGES = ["docs/AI_SETUP.md", "docs/AI_REFERENCE.md"]

ASSET_RE = re.compile(r"^strata-(engine|dsv41)-(\d+(?:\.\d+)*)-(windows|linux|macos)-(x86_64|aarch64|arm64)-(cuda|hip|sycl|metal)(?:-(.+?))?\.(?:tar\.gz|zip)$")
COLUMN_OF = {("windows", "cuda"): "nvidia_win", ("linux", "cuda", "x86_64"): "nvidia_linux", ("linux", "cuda", "aarch64"): "nvidia_spark",
             ("windows", "hip"): "amd_win", ("linux", "hip"): "amd_linux", ("linux", "sycl"): "intel_linux", ("macos", "metal"): "mac"}
PKG_RE = re.compile(r"^strata-(control|dsv41|engine)-(\d+(?:\.\d+)*)-(.+?)\.(?:tar\.gz|zip)$")
KIND_WORDS = {"control": "Control", "dsv41": "DeepSeek engines", "engine": "engines"}
MARK_RE = re.compile(r"<!-- cap:(\S+?) -->(.*?)<!-- /cap -->", re.S)
STATUS_WORD = {"yes": "supported", "untested": "package present, untested", "no": "does not run"}


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def column_for(os_name, arch, backend):
    return COLUMN_OF.get((os_name, backend, arch)) or COLUMN_OF.get((os_name, backend))


def packages(release, family):
    """column id -> sorted list of 'variant' strings for the family's packages in the release."""
    found = {}
    wanted = family.get("backends")
    for name in release["assets"]:
        if not name.startswith(family["asset_prefix"]):
            continue
        m = ASSET_RE.match(name)
        if not m:
            continue
        kind, ver, os_name, arch, backend, variant = m.groups()
        if wanted and backend not in wanted:
            continue
        col = column_for(os_name, arch, backend)
        found.setdefault(col, []).append((backend + ("-" + variant if variant else ""), ver))
    return {c: sorted(v) for c, v in found.items()}


def validate(cap, release):
    problems = []
    ids = [c["id"] for c in cap["columns"]]
    for fam in cap["families"]:
        pk = packages(release, fam)
        for col in pk:
            if col not in ids:
                problems.append("%s: a package lands in column %r, which the table does not have" % (fam["id"], col))
        for col in ids:
            cell = fam["cells"].get(col)
            if cell is None:
                problems.append("%s: no cell for %s" % (fam["id"], col))
                continue
            has = bool(pk.get(col))
            if cell["supported"] in ("yes", "untested") and not has:
                problems.append("%s on %s is marked %s but the release has no package for it" % (fam["id"], col, cell["supported"]))
            if cell["supported"] == "no" and has:
                problems.append("%s on %s is marked 'no' but the release has a package for it (%s)" % (fam["id"], col, ", ".join(v for v, _ in pk[col])))
            if cell["supported"] not in STATUS_WORD:
                problems.append("%s on %s: unknown status %r" % (fam["id"], col, cell["supported"]))
    return problems


def carried(release):
    """version -> kind -> platforms of the packages in the asset list that are not at the release's Control / engine version
    (a package carried over from an earlier release unchanged)."""
    out = {}
    for name in release["assets"]:
        m = PKG_RE.match(name)
        if m and m.group(2) != release["control_version" if m.group(1) == "control" else "engine_version"]:
            out.setdefault(m.group(2), {}).setdefault(m.group(1), []).append(m.group(3))
    return out


def versions_line(release):
    c, e = release["control_version"], release["engine_version"]
    if c == e or c.startswith(e + "."):                     # one version (a Control-only hotfix part counts as the same build)
        s = "App and Terminal %s, Control and engines %s" % (release["product_version"], e)
    else:                                                   # 0.2.15: Control 0.1.61 with the 0.1.60 engines of v0.2.14
        s = "App and Terminal %s, Control %s, engines %s" % (release["product_version"], c, e)
    for ver, kinds in sorted(carried(release).items()):
        s += "; these packages stay at %s: %s" % (ver, "; ".join("%s %s" % (KIND_WORDS[k], ", ".join(sorted(p)))
                                                                for k, p in sorted(kinds.items())))
    return s + ". App, Control and engine versions differ on purpose."


def label(cap, col):
    return next(c["label"] for c in cap["columns"] if c["id"] == col)


def cell_text(cap, fam, col):
    c = fam["cells"][col]
    if c["supported"] == "yes":
        return "runs" + (" (%s)" % c["hardware"] if c["hardware"] else "")
    if c["supported"] == "untested":
        return "package present, not tested"
    return "does not run"


def summary(cap, fam):
    yes = ["%s%s" % (label(cap, c["id"]), " (%s)" % fam["cells"][c["id"]]["hardware"] if fam["cells"][c["id"]]["hardware"] else "")
           for c in cap["columns"] if fam["cells"][c["id"]]["supported"] == "yes"]
    untested = [label(cap, c["id"]) for c in cap["columns"] if fam["cells"][c["id"]]["supported"] == "untested"]
    no = [label(cap, c["id"]) for c in cap["columns"] if fam["cells"][c["id"]]["supported"] == "no"]
    s = "%s runs in the desktop app and in the Terminal. Hardware: %s." % (fam["name"], "; ".join(yes))
    if untested:
        s += " Package present but not tested: %s." % ", ".join(untested)
    if no:
        s += " Does not run on: %s." % ", ".join(no)
    return s + " Per-hardware limits and measurements: [CAPABILITIES.md](CAPABILITIES.md#%s)." % fam["id"]


def block(cap, release, key):
    parts = key.split(":")
    fams = {f["id"]: f for f in cap["families"]}
    if parts[0] == "summary":
        return summary(cap, fams[parts[1]])
    if parts[0] == "cell":
        return cell_text(cap, fams[parts[1]], parts[2])
    raise KeyError(key)


def page(cap, release):
    cols = cap["columns"]
    out = ["# Which model runs on which hardware", "",
           "This page is built by `tools/capability_table.py` from [docs/data/capabilities.json](data/capabilities.json) and the package names of release %s ([docs/data/release.json](data/release.json)). Do not edit it by hand." % release["tag"],
           "",
           "- **Packaged**: the engine packages in the release for that hardware. Read from the release's file names.",
           "- **Supported**: what these docs commit to. *untested* means a package exists and nothing is claimed.",
           "- **Measured**: where numbers for that combination are published. Empty means none are published; it does not mean the combination fails.",
           "",
           versions_line(release),
           "", "## At a glance", "",
           "| Model | " + " | ".join(c["label"] for c in cols) + " |",
           "| --- |" + " --- |" * len(cols)]
    for fam in cap["families"]:
        out.append("| [%s](#%s) | %s |" % (fam["name"], fam["id"], " | ".join(
            {"yes": "yes", "untested": "untested", "no": "no"}[fam["cells"][c["id"]]["supported"]] for c in cols)))
    for fam in cap["families"]:
        pk = packages(release, fam)
        out += ["", '<a id="%s"></a>' % fam["id"], "", "## " + fam["name"], "", "Model guide: [%s](%s)." % (fam["guide"].split("#")[0], fam["guide"]), "",
                "| Hardware | Packaged | Supported | Limits | Measured |", "| --- | --- | --- | --- | --- |"]
        for c in cols:
            cell = fam["cells"][c["id"]]
            packed = ", ".join("`%s`" % v for v, _ in pk.get(c["id"], [])) or "none"
            what = STATUS_WORD[cell["supported"]]
            if cell["hardware"]:
                what += ": " + cell["hardware"]
            out.append("| %s | %s | %s | %s | %s |" % (c["label"], packed, what, cell["limits"] or "-", cell["measured"] or "-"))
    return "\n".join(out) + "\n"


def read(rel_path):
    with open(os.path.join(ROOT, rel_path), encoding="utf-8", newline="") as f:
        return f.read()


def write_file(rel_path, text):
    with open(os.path.join(ROOT, rel_path), "w", encoding="utf-8", newline="") as f:
        f.write(text)


def render(cap, release, text):
    return MARK_RE.sub(lambda m: "<!-- cap:%s -->%s<!-- /cap -->" % (m.group(1), block(cap, release, m.group(1))), text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    cap, release = read_json(DATA), read_json(RELEASE)
    problems = validate(cap, release)
    if args.write and not problems:
        write_file(OUT, page(cap, release))
        for p in PAGES:
            text = read(p)
            new = render(cap, release, text)
            if new != text:
                write_file(p, new)
    if not problems:
        want = page(cap, release)
        if not os.path.exists(os.path.join(ROOT, OUT)) or read(OUT).replace("\r\n", "\n") != want:
            problems.append("%s is not what tools/capability_table.py would write (run --write)" % OUT)
        for p in PAGES:
            text = read(p)
            if not MARK_RE.search(text):
                problems.append("%s has no <!-- cap:... --> block" % p)
            elif render(cap, release, text) != text:
                problems.append("%s: a <!-- cap:... --> block is out of date (run --write)" % p)
    for p in problems:
        print("FAIL " + p)
    print("capability table: %s" % ("%d problem(s)" % len(problems) if problems else "ok"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
