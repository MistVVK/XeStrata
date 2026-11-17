"""Paths and commands in the user-facing documentation that name a place the 0.2.16 layout move took away (read-only, git objects only).

    python -I docs_paths.py <repo> <rev> <rev before the move> [--fix-list FILE] [--all-contexts]

For every user-facing page (docs/*.md, docs/lang/*, README.md, AGENTS.md, .github/*.md; NOT docs/history, docs/bench-results, release notes, or a page
pinned to an old tag) it reads the code spans and code blocks, takes every path-like token, and checks it against the tree at <rev>:
  * the token exists there (from the repository root): fine;
  * it does not, but it did before the move and the file or folder it was has a new place (git's rename detection between the two revisions): a HIT,
    with the new path named;
  * it never existed (a placeholder, a run-time folder like packs/, a command word): ignored.
Also reported: commands that start a moved program from the repository root as before (`python strata.py`, `python -m serve.server`,
`.venv/bin/python tools/x.py`): the program is under terminal/ now, and the virtual environment setup.sh makes is terminal/.venv.
Prints one line per hit: page:line: token -> new path   [context]. Exit 1 while a hit is left.
--fix-list FILE writes "page<TAB>line<TAB>old<TAB>new<TAB>context" for the hits whose context is a command (a block or a span that starts like a command);
--fix-all FILE writes it for every hit.
"""
import posixpath
import re
import subprocess
import sys

repo, rev, before = sys.argv[1:4]
fix_list = sys.argv[sys.argv.index("--fix-list") + 1] if "--fix-list" in sys.argv else None
fix_all = sys.argv[sys.argv.index("--fix-all") + 1] if "--fix-all" in sys.argv else None
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def git(*a):
    return subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


tree = set(f for f in git("ls-tree", "-r", "--name-only", rev).split("\n") if f)
dirs = set()
for f in tree:
    d = posixpath.dirname(f)
    while d and d not in dirs:
        dirs.add(d)
        d = posixpath.dirname(d)
old_tree = set(f for f in git("ls-tree", "-r", "--name-only", before).split("\n") if f)
old_dirs = set()
for f in old_tree:
    d = posixpath.dirname(f)
    while d and d not in old_dirs:
        old_dirs.add(d)
        d = posixpath.dirname(d)
# old file -> new file (renames), then old folder -> new folder by the majority of its files
new_of_file = {}
for ln in git("diff", "-M", "--name-status", before, rev).split("\n"):
    p = ln.split("\t")
    if len(p) == 3 and p[0].startswith("R"):
        new_of_file[p[1]] = p[2]
new_of_dir = {}
votes = {}
for o, n in new_of_file.items():
    od, nd = posixpath.dirname(o), posixpath.dirname(n)
    while od:
        # the old folder od maps to the folder of n that has the same trailing part
        tail = o[len(od) + 1:]
        if n.endswith("/" + tail) or n == tail:
            cand = n[: len(n) - len(tail)].rstrip("/")
            votes.setdefault(od, {}).setdefault(cand, 0)
            votes[od][cand] += 1
        od = posixpath.dirname(od)
for od, c in votes.items():
    best = max(c.items(), key=lambda x: x[1])[0]
    new_of_dir[od] = best


def new_path(p):
    """The new place of an old path (file or folder), or None."""
    p = p.rstrip("/")
    if p in new_of_file:
        return new_of_file[p]
    if p in new_of_dir:
        return new_of_dir[p]
    d = p
    while "/" in d:                                   # a file inside a moved folder that git did not list (a changed file shown as modified)
        d = posixpath.dirname(d)
        if d in new_of_dir:
            cand = new_of_dir[d] + p[len(d):]
            return cand
    return None


PAGES_SKIP = ("docs/history/", "docs/bench-results/", "docs/release-notes", "docs/ORCA_Q4_K_S.md", "docs/paper/", "docs/media/")
pages = [f for f in sorted(tree) if f.endswith(".md") and (f.startswith("docs/") or f in ("README.md", "AGENTS.md") or f.startswith(".github/"))
         and not f.startswith(PAGES_SKIP) and "/ISSUE_TEMPLATE/" not in f and "PULL_REQUEST" not in f]

TOKEN = re.compile(r"(?<![\w./\\$%{}-])(?:\./)?((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.*<>-]+|[A-Za-z0-9_-]+\.(?:py|sh|bat|ps1|txt|toml|cmake|mjs|json))(?![\w/])")
CMD_START = re.compile(r"^\s*(?:[$>]\s*)?(?:sudo\s+)?(?:python3?|py\s+-3|\.venv[\\/](?:bin|Scripts)[\\/]python(?:\.exe)?|bash|sh|\./|cmake|docker|ninja|pip|node|npm|git|cd|"
                       r"START-HERE|\.\\|strata|tools[\\/]|ctest|build[\\/]|source|export|set|\$env)", re.I)
# placeholders and words that look like paths
IGNORE = re.compile(r"^(?:https?:|www\.|//|e\.g\.|i\.e\.|and/or|w/|read/|input/|on/off|yes/no|true/false|tok/s|GB/s|MB/s|km/h|\d)")


def show(p):
    return p.replace("\\", "/")


hits = []
for page in pages:
    text = git("show", "%s:%s" % (rev, page))
    in_block = False
    for n, line in enumerate(text.split("\n"), 1):
        s = line.strip()
        if s.startswith("```"):
            in_block = not in_block
            continue
        spans = []
        if in_block:
            spans.append((line, True))
        else:
            for m in re.finditer(r"`([^`\n]+)`", line):
                spans.append((m.group(1), False))
        for body, block in spans:
            is_cmd = bool(CMD_START.match(body))
            for m in TOKEN.finditer(body.replace("\\", "/") if not re.search(r"\s", body.strip()) else body):
                tok = m.group(1).rstrip(".,;:)")
                if IGNORE.match(tok) or tok.startswith(("http", "~", "/")) or "*" in tok or "<" in tok:
                    continue
                if tok in tree or tok in dirs:
                    continue
                if tok in old_tree or tok in old_dirs:
                    nw = new_path(tok)
                    if nw and nw != tok and (nw in tree or nw in dirs):
                        hits.append((page, n, tok, nw, "command" if (is_cmd or block) else "prose"))
            # the commands that start moved programs from the root
            if is_cmd:
                for pat, why in ((r"(?:python3?\s+|\.venv[/\\]bin[/\\]python\s+)(strata\.py)\b", "terminal/strata.py"),
                                 (r"-m\s+(serve\.server)\b", "run from terminal/"),
                                 (r"(?<![\w./\\-])\.venv[/\\](?:bin|Scripts)[/\\][A-Za-z0-9_.-]+", "terminal/.venv"),
                                 (r"-m venv \.venv\b", "terminal/.venv")):
                    for mm in re.finditer(pat, body):
                        hits.append((page, n, mm.group(0).strip(), why, "command"))
for h in hits:
    print("%s:%d: %s -> %s   [%s]" % h)
print("docs paths: %d pages read at %s, %d HITS (%d in commands)" % (len(pages), rev[:9], len(hits), sum(1 for h in hits if h[4] == "command")))
for target, only_cmd in ((fix_list, True), (fix_all, False)):
    if target:
        with open(target, "w", encoding="utf-8") as f:
            for h in hits:
                if h[4] == "command" or not only_cmd:
                    f.write("%s\t%d\t%s\t%s\t%s\n" % h)
sys.exit(1 if hits else 0)
