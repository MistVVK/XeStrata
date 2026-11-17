"""Path-segment scan for the 0.2.16 layout move (tools/layout/checks/path_scan.py).

    python -I path_scan.py <moved worktree> [--before REV] [--after REV] [--explained FILE] [--verbose]

The move script rewrites quoted paths, joins from a moved root and links. It cannot know a path that a program builds from separate
string segments, e.g. join(here, '..', '..', '..', 'serve', 'web', 'locales'). A test added after the edit list was written, with
such a line, reads a folder that is no longer there. This scan resolves such lines and lists the ones that broke:

  1. SEGMENT CHAINS: a run of quoted segments that holds '..' ("'..', '..', 'serve', 'web'") or a quoted "../x/y" string. It is
     resolved from the folder of the file, in BEFORE (the file's old place) and in AFTER (its new place). A chain is a HIT when its
     target exists in BEFORE and does not exist in AFTER. A chain whose first segment is not '..' is resolved from the file's own folder
     only when it is the argument list of join(here/__dirname/HERE/dirname, ...) - the common test form.
  2. VANISHED QUOTED PATHS: a quoted path such as "serve/web/ime.js" that exists from the repository root in BEFORE, exists nowhere in
     AFTER (not at the root, not under terminal/, engine/, app/, tools/, docs/), and sits on a line the move did not change.

A listed line is a miss (add an edit to move_0216_edits.py) or explained: add "<path glob> | <text in the line> | <reason>" to the
explained file. Exit status 0 only when nothing is left: the gate of the freeze-eve replay.
BEFORE and AFTER are two commits of the worktree (default HEAD~1 = the clean head, HEAD = the moved tree, as throwaway commits).
"""
import fnmatch
import posixpath
import re
import subprocess
import sys

TEXT = (".py", ".mjs", ".js", ".ts", ".tsx", ".sh", ".bat", ".ps1", ".cmake", ".yml", ".yaml", ".cpp", ".cu", ".hpp", ".h", ".txt", ".toml", ".rs", ".cjs")
SKIP_PREFIX = ("engine/third_party/", "docs/bench-results/", "docs/history/", "tools/layout/", "app/host/node_modules/")
QUOTED = r"""(?:"[^"\n]*"|'[^'\n]*')"""
CHAIN = re.compile(r"(?:%s\s*,\s*)*%s" % (QUOTED, QUOTED))
STR_UP = re.compile(r"""["'](\.\./(?:\.\./)*[^"'\n]+)["']""")
START = re.compile(r"""["']([A-Za-z_][A-Za-z0-9_-]*/[A-Za-z0-9_./-]+)["']""")
CMAKEVAR = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}/((?:\.\./)*[A-Za-z0-9_][A-Za-z0-9_./+-]*)")
HERE = re.compile(r"(here|HERE|__dirname|import\.meta\.dirname|dirname\([^)]*\)|Path\(__file__\)[^,]*|__file__)\s*,\s*$")


def git(wt, *a):
    r = subprocess.run(["git", "-C", wt] + list(a), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout


def tree(wt, rev):
    files = set(f for f in git(wt, "ls-tree", "-r", "--name-only", rev).split("\n") if f)
    dirs = set()
    for f in files:
        d = posixpath.dirname(f)
        while d and d not in dirs:
            dirs.add(d)
            d = posixpath.dirname(d)
    return files, dirs


def exists(t, p):
    return p in t[0] or p in t[1]


def resolve(base, segs):
    return posixpath.normpath(posixpath.join(base, *segs))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:]
    if not argv:
        sys.exit(__doc__)
    wt = argv[0]
    opt = lambda n, d: argv[argv.index(n) + 1] if n in argv else d
    before, after = opt("--before", "HEAD~1"), opt("--after", "HEAD")
    explained_file = opt("--explained", "")
    expl = []
    if explained_file:
        for ln in open(explained_file, encoding="utf-8"):
            ln = ln.rstrip("\n")
            if ln.strip() and not ln.startswith("#"):
                p = [x.strip() for x in ln.split("|", 2)]
                if len(p) == 3:
                    expl.append(p)
    tb, ta = tree(wt, before), tree(wt, after)
    old_of = {}
    for ln in git(wt, "diff", "-M", "--name-status", before, after).split("\n"):
        f = ln.split("\t")
        if len(f) == 3 and f[0].startswith("R"):
            old_of[f[2]] = f[1]
        elif len(f) == 2 and f[0] == "M":
            old_of[f[1]] = f[1]
    roots = ("terminal/", "engine/", "app/", "tools/", "docs/")
    files = [f for f in sorted(ta[0]) if f.endswith(TEXT) and not f.startswith(SKIP_PREFIX)]
    left, explained, checked = [], 0, 0
    for f in files:
        old = old_of.get(f, f)
        if old not in tb[0]:
            continue                                            # a file the move created: nothing to compare
        text = git(wt, "show", "%s:%s" % (after, f))
        if "'" not in text and '"' not in text:
            continue
        old_lines = set(git(wt, "show", "%s:%s" % (before, old)).split("\n"))
        for n, line in enumerate(text.split("\n"), 1):
            if len(line) > 600:
                continue
            hits = []
            for m in CHAIN.finditer(line):
                segs = [q[1:-1] for q in re.findall(QUOTED, m.group(0))]
                if ".." not in segs:
                    continue
                lead = 0
                while lead < len(segs) and segs[lead] == "..":
                    lead += 1
                if lead == 0:
                    if not HERE.search(line[:m.start()]):
                        continue
                    rel = segs
                else:
                    rel = segs
                if any(not s or "$" in s or "{" in s or "*" in s for s in segs):
                    continue
                checked += 1
                for k in range(len(rel), lead, -1):             # the longest prefix of the chain that names something
                    pb = resolve(posixpath.dirname(old), rel[:k])
                    if exists(tb, pb):
                        pa = resolve(posixpath.dirname(f), rel[:k])
                        if not exists(ta, pa):
                            hits.append("chain %s -> %s (before) is not there after (%s)" % (", ".join(repr(s) for s in segs), pb, pa))
                        break
            for m in STR_UP.finditer(line):
                rel = m.group(1)
                if any(c in rel for c in "${*"):
                    continue
                checked += 1
                pb = posixpath.normpath(posixpath.join(posixpath.dirname(old), rel))
                if exists(tb, pb):
                    pa = posixpath.normpath(posixpath.join(posixpath.dirname(f), rel))
                    if not exists(ta, pa):
                        hits.append("string %s -> %s (before) is not there after (%s)" % (rel, pb, pa))
            if f.endswith(".cmake") or f.endswith("CMakeLists.txt"):
                # 3. CMAKE VARIABLE PATHS: ${STRATA_ROOT}/ref/x.c, ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/y.cmake. The variable's value is not
                #    known, so the rest of the path is tried from the file's folder, its parent and the repository root (and engine/): it
                #    is a HIT when the path exists from one of them before the move and from none of them after it.
                for m in CMAKEVAR.finditer(line):
                    rest = m.group(1)
                    if any(c in rest for c in "${*") or "/" not in rest.strip("/"):
                        continue
                    checked += 1
                    def found(t, ff):
                        d = posixpath.dirname(ff)
                        bases = {d, posixpath.dirname(d), posixpath.dirname(posixpath.dirname(d)), ""}
                        return any(exists(t, posixpath.normpath(posixpath.join(b, rest))) for b in bases)
                    if found(tb, old) and not found(ta, f):
                        hits.append("cmake path %s resolves before the move and from no base after it" % rest)
            if line not in old_lines:
                pass
            else:
                for m in START.finditer(line):
                    p = m.group(1).rstrip("/.")
                    if any(c in p for c in "${*") or posixpath.splitext(p)[1] == "" and "/" not in p:
                        continue
                    if exists(tb, p) and not exists(ta, p) and not any(exists(ta, r + p) for r in roots):
                        hits.append("path %s exists at the old root, nowhere after the move" % p)
            if not hits:
                continue
            why = None
            for g, sub, reason in expl:
                if fnmatch.fnmatch(f, g) and sub in line:
                    why = reason
                    break
            if why:
                explained += 1
                continue
            for h in hits:
                left.append("%s:%d: %s\n    %s" % (f, n, h, line.strip()[:200]))
    for x in left:
        print(x)
    print("path scan: %d files, %d chains/strings resolved, %d explained, %d HITS LEFT" % (len(files), checked, explained, len(left)))
    sys.exit(1 if left else 0)


main()
