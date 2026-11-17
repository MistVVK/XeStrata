#!/usr/bin/env python3
"""Local link check for the Markdown pages.

    python tools/check_links.py                  current pages: exit 1 if a local link points at a file that is not there
    python tools/check_links.py --anchors        also fail on a #fragment that matches no heading or anchor in the target page
    python tools/check_links.py --baseline OUT   write a report of the archived pages (docs/history/, release notes, docs/bench-results, ...)

Local links only: http(s), mailto and other schemes are not fetched.  Links inside code fences and inline code are skipped.
Which pages are current and which are archived: tools/doc_pages.py.  Standard library only.
"""
import argparse
import os
import re
import sys
from urllib.parse import unquote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doc_pages  # noqa: E402

ROOT = doc_pages.ROOT
FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})")
INLINE_CODE = re.compile(r"`+[^`\n]*`+")
LINK = re.compile(r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*(<[^>]*>|[^)\s]*(?:\([^)\s]*\)[^)\s]*)*)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)")
REFDEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]*>|\S+)")
HTML_ATTR = re.compile(r"""\b(?:href|src)\s*=\s*["']([^"']+)["']""", re.I)
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def strip_code(text, inline=True):
    out, fence = [], None
    for line in text.split("\n"):
        m = FENCE.match(line)
        if m:
            if fence is None:
                fence = m.group(2)[0]
            elif m.group(2)[0] == fence:
                fence = None
            out.append("")
            continue
        out.append("" if fence else INLINE_CODE.sub(lambda x: " " * len(x.group(0)), line) if inline else ("" if fence else line))
    return out


def targets(text):
    """(line number, raw target) for every link in a page."""
    found = []
    in_comment = False
    for n, line in enumerate(strip_code(text), 1):
        if in_comment:
            if "-->" in line:
                in_comment, line = False, line.split("-->", 1)[1]
            else:
                continue
        line = re.sub(r"<!--.*?-->", lambda m: " " * len(m.group(0)), line)
        if "<!--" in line:
            in_comment, line = True, line.split("<!--", 1)[0]
        for rx in (LINK, REFDEF, HTML_ATTR):
            for m in rx.finditer(line):
                t = m.group(1).strip()
                if t.startswith("<") and t.endswith(">"):
                    t = t[1:-1]
                found.append((n, t))
    return found


def slug(heading):
    h = re.sub(r"<[^>]+>", "", heading).strip().lower()
    h = re.sub(r"[`*~]", "", h)
    h = re.sub(r"[^\w\- ]", "", h, flags=re.U)
    return h.replace(" ", "-")


_anchor_cache = {}


def anchors(path):
    if path in _anchor_cache:
        return _anchor_cache[path]
    seen, out = {}, set()
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        text = ""
    for line in strip_code(text, inline=False):
        m = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
        if m:
            s = slug(m.group(1))
            k = seen.get(s, 0)
            seen[s] = k + 1
            out.add(s if k == 0 else "%s-%d" % (s, k))
    for m in re.finditer(r"""<a\s+[^>]*?(?:id|name)\s*=\s*["']([^"']+)["']""", text, re.I):
        out.add(m.group(1))
    _anchor_cache[path] = out
    return out


def resolve(page, target):
    """Absolute path a local target points at, or None when it is not a local file link."""
    if not target or target.startswith("#") or SCHEME.match(target) or target.startswith("//"):
        return None
    path = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path:
        return None
    base = ROOT if path.startswith("/") else os.path.dirname(os.path.join(ROOT, page))
    return os.path.normpath(os.path.join(base, path.lstrip("/") if path.startswith("/") else path))


def check_page(page, want_anchors=False):
    problems = []
    with open(os.path.join(ROOT, page), encoding="utf-8", errors="replace") as f:
        text = f.read()
    for n, t in targets(text):
        frag = unquote(t.split("#", 1)[1]) if "#" in t else ""
        dest = resolve(page, t) if not t.startswith("#") else os.path.join(ROOT, page)
        if dest is None:
            continue
        if not os.path.exists(dest):
            problems.append((page, n, t, "missing file"))
        elif want_anchors and frag and dest.endswith(".md") and os.path.isfile(dest):
            if frag.lower() not in {a.lower() for a in anchors(dest)} and not re.match(r"^L\d+", frag):
                problems.append((page, n, t, "no such heading or anchor"))
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--anchors", action="store_true", help="also check #fragments against the target page's headings")
    ap.add_argument("--baseline", metavar="OUT", help="write a report of the archived pages to OUT and exit 0")
    args = ap.parse_args(argv)
    if args.baseline:
        rows = []
        for page, archived in doc_pages.markdown_files():
            if archived:
                rows += check_page(page, False)
        by = {}
        for page, n, t, why in rows:
            by.setdefault(page, []).append((n, t))
        lines = ["# Link baseline for archived pages", "",
                 "Local links in archived pages (docs/history/, release notes, docs/bench-results, phaseA-tests) whose target file is missing, as of the commit that wrote this file.",
                 "Archived pages are not fixed: they keep the links they had, and evidence they pointed to that was never published stays missing.",
                 "`python tools/check_links.py --baseline <file>` rebuilds this report. %d missing targets in %d pages." % (len(rows), len(by)), "",
                 "| Page | Missing local targets |", "| --- | ---: |"]
        lines += ["| %s | %d |" % (p, len(v)) for p, v in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
        with open(args.baseline, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print("baseline: %d missing targets in %d archived pages -> %s" % (len(rows), len(by), args.baseline))
        return 0
    pages = doc_pages.current_pages()
    problems = []
    for page in pages:
        problems += check_page(page, args.anchors)
    for page, n, t, why in problems:
        print("FAIL %s:%d  (%s)  %s" % (page, n, t, why))
    print("link check: %d current pages, %s" % (len(pages), "%d broken local link(s)" % len(problems) if problems else "all local links resolve"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
