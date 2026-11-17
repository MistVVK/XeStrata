"""MAP-0216 (from MAP-0215): the 0.2.16 top-level layout move, replayable.

    python -I move_0216.py <worktree> [--plan] [--no-edits]

Run a COPY of tools/layout (it moves the tree it lives in): copy the folder out of the checkout first, then run the copy's
move_0216.py on a clean checkout.  tools/layout stays in tools/ (KEEP_IN_TOOLS).  See tools/layout/README.md.

<worktree> must be a clean checkout of the base commit (baf16363b or a later main with the same top level).
1. git mv per MOVES (a pure rename, so `git log --follow` and blame keep working);
2. the path edits:
   a. ROOT-RELATIVE files (CI workflows, Dockerfiles and their ignore files, .gitattributes, .gitignore, tools/check):
      every path token that starts with a moved entry is mapped through MOVES (longest prefix first);
   b. Markdown links and HTML href/src in every .md: resolved from the page's OLD place, mapped, written relative to
      its NEW place (links that pointed at nothing before stay as they were);
   c. GitHub URLs .../Strata/(blob|tree|raw)/main/<path>: mapped;
   d. EDITS: explicit (file, old, new, count) replacements for code that computes a root (parents[N], dirname) or names
      a moved entry in a way a token rewrite cannot know (Python joins, CMake sources outside engine/, the archive
      map of package_terminal.py).  Each must match exactly `count` times, or the script stops and says which.
Prints a summary and writes <worktree>/../map0216-edits.log (one line per edited file with the number of changes).
Nothing is committed: the caller commits on a throwaway branch.
"""
import os
import posixpath
import re
import subprocess
import sys

WT = os.path.abspath(sys.argv[1])
PLAN = "--plan" in sys.argv
NO_EDITS = "--no-edits" in sys.argv

LANGS = ["de", "es", "fr", "ja", "pl", "pt-BR", "zh-CN"]
REQS = ["requirements.txt", "requirements-build.txt", "requirements-dev.txt", "requirements-monitor.txt", "requirements-nvfp4-local.txt"]

# (old, new), applied in order.  Directory entries move whole; an entry listed before its parent moves first.
MOVES = []
# --- engine/: everything CMake builds
for e in ["src", "include", "cmake", "metal", "sycl", "spark", "third_party", "CMakeLists.txt", "CMakePresets.json"]:
    MOVES.append((e, "engine/" + e))
MOVES.append(("tests/core/test_tqp_ref.py", "terminal/tests/core/test_tqp_ref.py"))   # a pytest file among the C++ tests: it stays with py-tests (found 11-17: its 5 tests were no longer collected)
for t in ["core", "cuda", "gdn", "hip", "ngram", "platform", "prefill", "program", "quant"]:
    MOVES.append(("tests/" + t, "engine/tests/" + t))           # C++ tests (CTest)
MOVES.append(("tests/data/native_experts", "engine/tests/data/native_experts"))
for b in ["micro", "gdn_chunk_bench.cpp", "qsa_prompt_attn_sm70_bench.cpp", "gfx906_qsa_query_swizzle_probe.cpp",
          "test_gfx906_qsa_query_swizzle_guard.py"]:
    MOVES.append(("bench/" + b, "engine/bench/" + b))           # C++ benches the CMake files build
# --- docs/
for l in LANGS:
    MOVES.append(("README.%s.md" % l, "docs/lang/README.%s.md" % l))
MOVES += [("history", "docs/history"), ("bench/results", "docs/bench-results")]
# AGENTS.md stays at the root (owner 11-16, Q2: coding agents read it there by convention)
# --- .github/
MOVES += [("CONTRIBUTING.md", ".github/CONTRIBUTING.md"), ("SECURITY.md", ".github/SECURITY.md")]
# --- tools/ (the repository's own tooling; tools/check stays where CI and the docs call it)
MOVES += [("packaging", "tools/packaging"), ("docker", "tools/docker"), ("Dockerfile", "tools/docker/Dockerfile.src"),
          (".dockerignore", "tools/docker/Dockerfile.src.dockerignore"), ("research", "tools/research"),
          ("flake.nix", "tools/nix/flake.nix"), ("flake.lock", "tools/nix/flake.lock"),
          ("BENCH.bat", "tools/BENCH.bat"), ("bench.sh", "tools/bench.sh")]
# --- terminal/: the Terminal as it is installed (the archive keeps these paths), plus its tests
MOVES += [("bench/prompts", "terminal/bench/prompts")]
# crates/ (new since MAP-0215, #5961): the Terminal's native helper programs (strata_control/native_helpers.py) -> terminal/crates
for e in ["strata.py", "setup.py", "strata_cli", "strata_control", "serve", "integrations", "bin", "conftest.py", "data", "ref", "crates"] + REQS:
    MOVES.append((e, "terminal/" + e))
# the repository checks stay in tools/ (CI and the docs call them there; none is part of the Terminal): tools/check and the
# docs / vendored-code / binary checks with their tests.  Listed so the token map keeps them.
KEEP_IN_TOOLS = ["layout", "check", "check_links.py", "doc_pages.py", "release_manifest.py", "capability_table.py", "bench_results_index.py",
                 "settings_register.py", "check_vendored.py", "check_no_mmq.py", "check_cuda_spills.py", "cuda_spill_baseline.json",
                 "check_isa_guard.py", "test_check_links.py", "test_release_manifest.py", "test_capability_table.py",
                 "test_bench_results_index.py", "test_settings_register.py", "test_check_no_mmq.py", "test_check_cuda_spills.py",
                 "test_check_isa_guard.py"]
for k in KEEP_IN_TOOLS:
    MOVES.append(("tools/" + k, "tools/" + k))
MOVES.append(("tools", "terminal/tools"))                      # the Python tools (model preparation, tests, checks)
for t in ["desktop", "snapshot", "split", "sycl", "terminal", "time_skip", "tuning", "data", "fixtures"]:
    MOVES.append(("tests/" + t, "terminal/tests/" + t))
MOVES += [("bench", "tools/bench")]                            # what is left of bench/ (profile corpus, bench_vs_llama.py)
# --- app/
MOVES.append(("apps", "app"))
# --- gone (the start files people use stay: START-HERE.bat, setup.sh); see MAP-0215.md for why
REMOVE = ["SETUP.bat", "UPDATE.bat", "update.sh"]


def git(*a, check=True):
    return subprocess.run(["git", "-C", WT, *a], capture_output=True, text=True, encoding="utf-8", check=check)


TRACKED = set(git("ls-files").stdout.splitlines())


def moved(path):
    """The new place of an OLD repo-relative path (longest matching prefix of MOVES), or None if it does not move."""
    best = None
    for old, new in MOVES:
        if path == old or path.startswith(old + "/"):
            if best is None or len(old) > len(best[0]):
                best = (old, new)
    if best is None:
        return None
    return best[1] + path[len(best[0]):]


def new_of(path):
    m = moved(path)
    if path in REMOVE:
        return None
    return m if m is not None else path


# ---------------------------------------------------------------------------------------------------- 1. the moves
OLD_OF = {}                     # new path -> old path, for every tracked file
for f in TRACKED:
    OLD_OF[new_of(f) or f] = f

if PLAN:
    for old, new in MOVES:
        print("%-45s -> %s" % (old, new))
    print("removed:", " ".join(REMOVE))
    sys.exit(0)

done_moves = 0
os.makedirs(os.path.join(WT, "terminal"), exist_ok=True)
# tools/ first: the folder is reused for the repository tooling, so the Python tools leave before anything moves in
if os.path.exists(os.path.join(WT, "terminal", "tools")):
    sys.exit("terminal/tools exists already (an untracked leftover such as __pycache__?): start from a clean checkout (git clean -fdx)")
git("mv", "tools", "terminal/tools")
os.makedirs(os.path.join(WT, "tools"), exist_ok=True)
for k in KEEP_IN_TOOLS:
    if os.path.exists(os.path.join(WT, "terminal", "tools", k)):
        git("mv", "terminal/tools/" + k, "tools/" + k)
done_moves += 2
for old, new in MOVES:
    if old == "tools" or (old.startswith("tools/") and old[6:] in KEEP_IN_TOOLS):
        continue
    if old == new:
        continue
    src = os.path.join(WT, old)
    if not os.path.exists(src):
        continue
    # only what is still tracked under `old` (an earlier, longer entry may have taken part of it)
    if os.path.isdir(src) and not git("ls-files", "--", old).stdout.strip():
        continue
    os.makedirs(os.path.dirname(os.path.join(WT, new)) or WT, exist_ok=True)
    if os.path.exists(os.path.join(WT, new)) and os.path.isdir(os.path.join(WT, new)):
        # the target folder exists (a child moved there first): move the remaining children one by one
        for child in sorted(os.listdir(src)):
            if git("ls-files", "--", old + "/" + child).stdout.strip():
                git("mv", old + "/" + child, new + "/" + child)
                done_moves += 1
        continue
    git("mv", old, new)
    done_moves += 1
for r in REMOVE:
    if r in TRACKED:
        git("rm", "-q", r)
print("git mv: %d moves, %d removed" % (done_moves, len(REMOVE)))
if NO_EDITS:
    sys.exit(0)

NOW = set(git("ls-files").stdout.splitlines())
counts = {}


def read(p):
    with open(os.path.join(WT, p), "rb") as fh:
        raw = fh.read()
    return raw


def write(p, text, n, why):
    with open(os.path.join(WT, p), "wb") as fh:
        fh.write(text)
    counts.setdefault(p, {}).setdefault(why, 0)
    counts[p][why] += n


# ------------------------------------------------------------------------------------- 2a. root-relative path tokens
TOPS = sorted({o.split("/")[0] for o, _ in MOVES} | set(REMOVE), key=len, reverse=True)
LEAD = r"(/|\$\{?[A-Za-z_]+\}?[/\\]|%~dp0|%[A-Za-z_]+%[/\\])?"
TOKEN = re.compile(r"(?<![A-Za-z0-9_.\-/\\$@:%])" + LEAD + "(" + "|".join(re.escape(t) for t in TOPS) + r")((?:[/\\][A-Za-z0-9_.\-*{}$\[\]]*)*)(?![A-Za-z0-9_\-])")
DIRS = {o.split("/")[0] for o, _ in MOVES if os.path.isdir(os.path.join(WT, moved(o) or o)) or "/" in o}


STRICT = [True]
TRACKED_DIRS_ALL = set()
for _f in TRACKED:
    _ps = _f.split("/")
    for _i in range(1, len(_ps)):
        TRACKED_DIRS_ALL.add("/".join(_ps[:_i]))


MODE = {"slash_lead": False, "backslash": False, "bare_files": False}   # set per file by token_pass


# MAP-0216: a path after a variable is the repository's only when that variable is the checkout ($repo, %SRC% ...).  MAP-0215 mapped
# every $VAR/ lead, so a toolkit's $CUDA/bin became $CUDA/terminal/bin and a package stage's $STAGE/metal/kernels became
# $STAGE/engine/metal/kernels (the Mac package layout).  Other leads are kept and counted (printed at the end).
ROOT_VARS = {"repo", "REPO", "ROOT", "root", "SRC", "REPO_ROOT", "GITHUB_WORKSPACE"}
LEAD_VAR = re.compile(r"^(?:\$\{?([A-Za-z_]+)\}?|%([A-Za-z_]+)%)[/\\]$")
SKIPPED_LEADS = {}


def map_token(m):
    lead, top, rest = m.group(1) or "", m.group(2), m.group(3)
    lv = LEAD_VAR.match(lead)
    if lv and (lv.group(1) or lv.group(2)) not in ROOT_VARS:
        var = lv.group(1) or lv.group(2)
        SKIPPED_LEADS[var] = SKIPPED_LEADS.get(var, 0) + 1
        return m.group(0)
    if top in DIRS and not rest and not lead:
        return m.group(0)                     # a bare word ("tests", "data"): not a path
    if lead == "/" and not MODE["slash_lead"]:
        return m.group(0)                     # "/data", "/src/": an absolute path (a container's), not an ignore-file anchor
    suffix = ""
    if "\\" in rest and not MODE["backslash"]:
        # a backslash outside a Windows script is an escape (\"tools\": or a path's closing \"), not a separator: map what is before it
        suffix = rest[rest.index("\\"):]
        rest = rest[:rest.index("\\")]
        if top in DIRS and not rest:
            return m.group(0)
    if not rest and not lead and top not in DIRS and not MODE["bare_files"]:
        # a bare file name ("CMakeLists.txt", "setup.py"): a path when it is quoted or in a script's command line, not in prose
        s, a, b = m.string, m.start(), m.end()
        quoted = a > 0 and b < len(s) and s[a - 1] in "\"'`" and s[b] in "\"'`"
        line = s[s.rfind("\n", 0, a) + 1:a].lstrip()
        command = MODE.get("script") and not line.startswith(("#", "rem ", "REM ", "::"))
        if not (quoted or command):
            return m.group(0)
    back = "\\" in rest or lead.endswith("\\") or lead == "%~dp0"
    path = top + rest.replace("\\", "/")
    # a glob / trailing slash: map the longest concrete prefix
    core = path.rstrip("/")
    tail = path[len(core):]
    # STRICT (code and CI): a path below a top-level entry is mapped only when it exists in the old tree or is a pattern
    # ("data/capabilities.json" in a docs page's text is relative to docs/, not the repository's data/)
    if STRICT[0] and "/" in core and not any(c in core for c in "*?[{$") and core not in TRACKED and core not in TRACKED_DIRS_ALL:
        return m.group(0)
    probe = core
    while True:
        new = moved(probe)
        if probe in REMOVE or new is not None or "/" not in probe:
            break
        probe = probe.rsplit("/", 1)[0]
    if new is None:
        return m.group(0)
    out = new + core[len(probe):] + tail
    return lead + (out.replace("/", "\\") if back else out) + suffix


def token_pass(p, why="root-relative tokens"):
    MODE["slash_lead"] = p.endswith(("ignore", ".gitattributes"))
    MODE["backslash"] = p.endswith((".bat", ".cmd", ".ps1"))
    MODE["bare_files"] = p.startswith(".github/workflows/") or p == "tools/check" or MODE["slash_lead"]
    MODE["script"] = p.endswith((".sh", ".bat", ".cmd", ".ps1")) or "Dockerfile" in p.rsplit("/", 1)[-1]
    raw = read(p).decode("utf-8")
    new = TOKEN.sub(map_token, raw)
    n = sum(1 for a in TOKEN.finditer(raw) if map_token(a) != a.group(0))
    if new != raw:
        write(p, new.encode("utf-8"), n, why)


# files whose paths are read from the repository root: CI, the container builds, tools/check, the release build scripts
# (they cd to the checkout first).  NOT install.sh / install.ps1: they name paths inside the downloaded archives, which keep theirs.
ROOT_RELATIVE = [p for p in NOW if p.startswith(".github/workflows/")] + \
    [p for p in NOW if p.startswith("tools/docker/") and p.rsplit("/", 1)[-1] in
     ("Dockerfile.src", "Dockerfile.src.dockerignore", "publish.sh", "fetch-engines.py")] + \
    [p for p in NOW if p.startswith("tools/packaging/") and p.endswith((".sh", ".bat")) and not p.endswith("install.sh")] + \
    ["engine/spark/build_engine.sh", ".gitattributes"] + ["tools/" + k for k in KEEP_IN_TOOLS if k.endswith((".py", "check"))]
for p in sorted(set(q for q in ROOT_RELATIVE if q in NOW)):
    token_pass(p)

# Python joins from the repository root in the same files: ROOT / "bench" / "results", os.path.join(ROOT, "third_party", "x")
JOIN = re.compile(r"""(\b(?:ROOT|REPO|REPO_ROOT|root)\s*(,|/)\s*)((?:["'][A-Za-z0-9_.\-]+["'])(?:\s*(?:,|/)\s*["'][A-Za-z0-9_.\-]+["'])*)""")
PART = re.compile(r"""["']([A-Za-z0-9_.\-]+)["']""")


def map_join(m):
    head, sep, body = m.group(1), m.group(2), m.group(3)
    parts = PART.findall(body)
    q = body[0]
    if parts == ["tools"]:
        return m.group(0)              # tools/ is still a top-level folder (the repository checks)
    for k in range(len(parts), 0, -1):
        path = "/".join(parts[:k])
        if path in TRACKED or path in TRACKED_DIRS_ALL or path in REMOVE:
            new = moved(path)
            if new is None or new == path:
                return m.group(0)
            newparts = new.split("/") + parts[k:]
            glue = (" / " if sep == "/" else ", ")
            return head + glue.join(q + x + q for x in newparts)
    return m.group(0)


TRACKED_DIRS_ALL = set()
for f in TRACKED:
    ps = f.split("/")
    for i in range(1, len(ps)):
        TRACKED_DIRS_ALL.add("/".join(ps[:i]))
for p in sorted(set(q for q in ROOT_RELATIVE if q in NOW and (q.endswith(".py") or q == "tools/check"))):
    raw = read(p).decode("utf-8")
    new = JOIN.sub(map_join, raw)
    if new != raw:
        write(p, new.encode("utf-8"), sum(1 for a in JOIN.finditer(raw) if map_join(a) != a.group(0)), "root-relative joins")

# 2e. Python files that moved one level down WITH their root (terminal/..., engine/...): their ROOT (parents[N] of the file) was the
# repository and is now terminal/ or engine/.  A join to something that stayed inside that folder is unchanged; a join to something
# outside it becomes ROOT.parent / <new path> (os.path.dirname(ROOT) for os.path.join).  Runtime code keeps its joins to files that the
# Terminal archive carries from outside terminal/ (docs/TERMINAL_LAUNCH.md ...): the archive has them at ROOT, and those few places
# need both layouts (the explicit edits).
JOIN2 = re.compile(r"""(((?:\bself\.)?\b(?:ROOT|REPO|REPO_ROOT|_ROOT)|(?<![.\w])repo)\s*(,|/)\s*)((?:["'][A-Za-z0-9_.\-/]+["'])(?:\s*(?:,|/)\s*["'][A-Za-z0-9_.\-/]+["'])*)""")
PART2 = re.compile(r"""["']([A-Za-z0-9_.\-/]+)["']""")
ARCHIVE_FROM_OUTSIDE = ("docs/MANAGEMENT_PROTOCOL.md", "docs/TERMINAL_LAUNCH.md", "metal/tools", "third_party/strata-gguf-py", "third_party")


def map_join2(group, is_test, engine_fn=None):
    def f(m):
        head, name, sep, body = m.group(1), m.group(2), m.group(3), m.group(4)
        parts = [x for s_ in PART2.findall(body) for x in s_.strip("/").split("/") if x]
        q = body[0]
        for k in range(len(parts), 0, -1):
            path = "/".join(parts[:k])
            if path in TRACKED or path in TRACKED_DIRS_ALL or path in REMOVE:
                new = moved(path)
                if path in REMOVE:
                    return m.group(0)
                if new is None:
                    new = path              # a file or folder that stays at the repository root (README.md, START-HERE.bat, docs/...)
                if not is_test and any(path == a or path.startswith(a + "/") for a in ARCHIVE_FROM_OUTSIDE):
                    return m.group(0)
                glue = (" / " if sep == "/" else ", ")
                rest = parts[k:]
                if new == group or new.startswith(group + "/"):
                    inner = new[len(group) + 1:].split("/") if new != group else []
                    if inner == parts[:k]:
                        return m.group(0)
                    return head + glue.join(q + x + q for x in inner + rest)
                if engine_fn and sep == "/" and new.startswith("engine/"):
                    return engine_fn + " / " + glue.join(q + x + q for x in new.split("/")[1:] + rest)
                base = m.group(1)[:m.group(1).index(name) + len(name)]
                up = (base + ".parent" + " / ") if sep == "/" else ("os.path.dirname(" + base + "), ")
                return up + glue.join(q + x + q for x in new.split("/") + rest)
        return m.group(0)
    return f


# 2f. relative path strings in the app's scripts and tests ('../../../strata_control/drafting.py', new URL('../../../setup.py', ...)):
# app/ is at the depth apps/ was, so each string is resolved from the file's old folder and written for its new one
RELSTR = re.compile(r"""(['"`])((?:\.\./)+[A-Za-z0-9_.\-/]+)(['"`])""")
for p in sorted(NOW):
    if not p.startswith("app/") or not p.endswith((".mjs", ".js", ".ts", ".cjs")):
        continue
    old = OLD_OF.get(p, p)
    raw = read(p).decode("utf-8")

    def relfix(m, p=p, old=old):
        target = posixpath.normpath(posixpath.join(posixpath.dirname(old), m.group(2)))
        if target.startswith("..") or (target not in TRACKED and target not in TRACKED_DIRS_ALL):
            return m.group(0)
        new = new_of(target) or target
        rel = posixpath.relpath(new, posixpath.dirname(p))
        if m.group(2).endswith("/") and not rel.endswith("/"):
            rel += "/"
        return m.group(0) if rel == m.group(2) else m.group(1) + rel + m.group(3)
    new = RELSTR.sub(relfix, raw)
    if new != raw:
        write(p, new.encode("utf-8"), sum(1 for a in RELSTR.finditer(raw) if relfix(a) != a.group(0)), "app relative strings")

for p in sorted(NOW):
    if not p.endswith(".py"):
        continue
    group = p.split("/")[0]
    old = OLD_OF.get(p, p)
    if group not in ("terminal", "engine") or old.split("/")[0] == group:
        continue
    raw = read(p).decode("utf-8")
    is_test = "/tests/" in p or p.rsplit("/", 1)[-1].startswith("test_") or p.endswith("conftest.py")
    fn = map_join2(group, is_test, "engine_src()" if p == "terminal/setup.py" else None)
    # MAP-0216: the inline root too, `Path(__file__).resolve().parents[N] / "apps" / ...`, where parents[N] was the repository (N = the
    # file's old folder depth); after the one-level move the same expression is terminal/ or engine/, exactly like a ROOT name
    rx = re.compile(JOIN2.pattern.replace(r"|(?<![.\w])repo)", r"|(?<![.\w])repo|Path\(__file__\)(?:\.resolve\(\))?\.parents\[%d\])" % old.count("/"), 1))
    new = rx.sub(fn, raw)
    if new != raw:
        write(p, new.encode("utf-8"), sum(1 for a in rx.finditer(raw) if fn(a) != a.group(0)), "joins from a moved root")

# .gitignore: tracked paths are mapped; what setup.py writes at the install root (/.venv/, /engine/, /models/ ...) is written
# in terminal/ now, and is kept at the old place too for a clone made before the move (never /engine/: that is the engine's source)
STRICT[0] = False
MODE.update(slash_lead=True, backslash=False, bare_files=True)
gi = read(".gitignore").decode("utf-8")
out = []
for line in gi.splitlines():
    s = line.strip()
    if s.startswith("/") and not s.startswith("/.git") and s.split("/")[1] not in TOPS and s.split("/")[1] not in ("terminal",):
        first = s.split("/")[1].rstrip("*")
        out.append("/terminal" + s)
        if first not in ("engine",):
            out.append(s)
        continue
    out.append(TOKEN.sub(map_token, line))
gi2 = "\n".join(out) + "\n"
if gi2 != gi:
    write(".gitignore", gi2.encode("utf-8"), sum(1 for a, b in zip(gi.splitlines(), out) if a != b), "gitignore")

# ----------------------------------------------------------------------------------- 2b. Markdown and HTML links
LINK = re.compile(r"(!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*<?)([^)\s>]+)")
ATTR = re.compile(r"(\b(?:href|src)\s*=\s*[\"'])([^\"']+)", re.I)
REFDEF = re.compile(r"^(\s{0,3}\[[^\]]+\]:\s*<?)(\S+?)(>?)$", re.M)
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:|^#|^/|^\{|^\$")


def relink(page_new, page_old):
    def fix(m):
        target = m.group(2)
        if SCHEME.match(target):
            return m.group(0)
        path, frag = (target.split("#", 1) + [""])[:2] if "#" in target else (target, None)
        path_q, query = (path.split("?", 1) + [""])[:2] if "?" in path else (path, None)
        if not path_q:
            return m.group(0)
        old_abs = posixpath.normpath(posixpath.join(posixpath.dirname(page_old), path_q))
        if old_abs.startswith(".."):
            return m.group(0)
        if old_abs not in TRACKED and old_abs.rstrip("/") not in TRACKED_DIRS:
            return m.group(0)
        new_abs = new_of(old_abs) or old_abs
        rel = posixpath.relpath(new_abs, posixpath.dirname(page_new) or ".")
        if path_q.endswith("/") and not rel.endswith("/"):
            rel += "/"
        if rel == path_q.rstrip("/") or rel == path_q:
            return m.group(0)
        out = rel + ("?" + query if query is not None else "") + ("#" + frag if frag is not None else "")
        return m.group(1) + out
    return fix


TRACKED_DIRS = set()
for f in TRACKED:
    parts = f.split("/")
    for i in range(1, len(parts)):
        TRACKED_DIRS.add("/".join(parts[:i]))

for p in sorted(NOW):
    if not p.endswith(".md"):
        continue
    old = OLD_OF.get(p, p)
    raw = read(p)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        continue
    fixer = relink(p, old)
    n_before = text
    text2 = LINK.sub(fixer, text)
    text2 = ATTR.sub(fixer, text2)
    def fix_ref(m):
        r = fixer(m)
        return r if r == m.group(0) else r + m.group(3)
    text2 = REFDEF.sub(fix_ref, text2)
    if text2 != text:
        n = sum(1 for a, b in zip(text.splitlines(), text2.splitlines()) if a != b)
        write(p, text2.encode("utf-8"), n, "links")

# -------------------------------------------------------------------------------------------------- 2c. GitHub URLs
GH = re.compile(r"(github\.com/Niko1221/Strata/(?:blob|tree|raw)/main/|raw\.githubusercontent\.com/Niko1221/Strata/main/)([A-Za-z0-9_.\-/]+)")
for p in sorted(NOW):
    raw = read(p)
    if b"Niko1221/Strata/" not in raw and b"Niko1221/Strata/main" not in raw:
        continue
    if b"\0" in raw[:8192]:
        continue
    text = raw.decode("utf-8", "replace")
    def gh(m):
        path = m.group(2).rstrip(".")
        new = moved(path)
        if new is None or (path not in TRACKED and not any(f.startswith(path + "/") for f in TRACKED)):
            return m.group(0)
        return m.group(1) + new + m.group(2)[len(path):]
    text2, n = GH.subn(gh, text)
    if text2 != text:
        write(p, text2.encode("utf-8"), sum(1 for a in GH.finditer(text) if gh(a) != a.group(0)), "github urls")

# ------------------------------------------------------------------------------------------------- 2d. explicit edits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from move_0216_edits import EDITS, REGEX_EDITS          # noqa: E402

missing = []
for p, old_s, new_s, count in EDITS:
    if p not in NOW:
        missing.append("%s: not in the tree" % p)
        continue
    text = read(p).decode("utf-8")
    k = text.count(old_s)
    if k != count:
        missing.append("%s: %r found %d times, expected %d" % (p, old_s[:70], k, count))
        continue
    write(p, text.replace(old_s, new_s).encode("utf-8"), count, "explicit")

import fnmatch  # noqa: E402
for pat, rx, repl, need in REGEX_EDITS:
    total = 0
    for p in sorted(q for q in NOW if fnmatch.fnmatch(q, pat)):
        text = read(p).decode("utf-8")
        text2, n = re.subn(rx, repl, text)
        if n:
            write(p, text2.encode("utf-8"), n, "explicit regex")
            total += n
    if total < need:
        missing.append("regex %r on %s: %d matches, expected at least %d" % (rx, pat, total, need))

# MAP-0216: tables of repository paths in a file (a test's list of scripts it reads, ...): inside the span from `start` to `end`,
# every quoted string that names a tracked file or folder of the OLD tree is written as its new path.  The code that joins them
# to a root is changed by an explicit edit beside it (ROOT / rel -> ROOT.parent / rel).
import move_0216_edits as _E  # noqa: E402
QUOTED = re.compile(r"""(["'])([A-Za-z0-9_.\-/]+)\1""")
for p, start, end, need in getattr(_E, "PATH_SPANS", []):
    if p not in NOW:
        missing.append("%s: not in the tree (path span)" % p)
        continue
    text = read(p).decode("utf-8")
    a = text.find(start)
    b = text.find(end, a + len(start)) if a >= 0 else -1
    if a < 0 or b < 0:
        missing.append("%s: path span %r ... %r not found" % (p, start[:40], end[:40]))
        continue

    def span_fix(m):
        old = m.group(2)
        # a path (it has a /) or a tracked FILE: a bare folder name in the table ("cmake" as a program name) is a word, not a path
        if (("/" in old and (old in TRACKED or old in TRACKED_DIRS_ALL)) or old in TRACKED) and new_of(old) and new_of(old) != old:
            return m.group(1) + new_of(old) + m.group(1)
        return m.group(0)
    seg = text[a:b]
    seg2 = QUOTED.sub(span_fix, seg)
    n = sum(1 for x, y in zip(QUOTED.findall(seg), QUOTED.findall(seg2)) if x != y)
    if n < need:
        missing.append("%s: path span mapped %d strings, expected at least %d" % (p, n, need))
        continue
    if n:
        write(p, (text[:a] + seg2 + text[b:]).encode("utf-8"), n, "path tables")

# REPORT ONLY (nothing changed): repository paths named in the TEXT of current pages and in code comments/strings of the moved
# code, outside the files edited above. They read from the repository root ("python tools/x.py", "see src/foo.cpp"); some of them
# describe an installed Terminal, whose paths do not change, so each needs a reader's decision.
EDIT_SKIPS = dict(SKIPPED_LEADS)          # the edit passes only (the report below calls map_token too)
STRICT[0] = True
MODE.update(slash_lead=False, backslash=False, bare_files=False)
prose = []
for p in sorted(NOW):
    if p.startswith(("docs/history/", "docs/bench-results/", "engine/third_party/")) or "release-notes-" in p:
        continue
    if not p.endswith((".md", ".py", ".mjs", ".js", ".ts", ".cpp", ".cu", ".hpp", ".h", ".cmake", ".txt", ".sh", ".bat", ".ps1", ".rs", ".json")):
        continue
    if p in counts and any(k in counts[p] for k in ("root-relative tokens",)):
        continue
    try:
        text = read(p).decode("utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for n, line in enumerate(text.splitlines(), 1):
        for mm in TOKEN.finditer(line):
            if map_token(mm) != mm.group(0) and (mm.group(3) or mm.group(2) in REMOVE or "." in mm.group(2)):
                prose.append("%s\t%d\t%s\t%s" % (p, n, mm.group(0), map_token(mm)))
with open(os.path.join(os.path.dirname(WT), "map0216-prose.tsv"), "w", encoding="utf-8") as o:
    o.write("file\tline\tnames\twould be\n" + "\n".join(prose) + "\n")
print("report: %d repository paths named in text/comments of %d files (not edited; map0216-prose.tsv)" % (len(prose), len({x.split(chr(9))[0] for x in prose})))

log = os.path.join(os.path.dirname(WT), "map0216-edits.log")
tot = 0
with open(log, "w", encoding="utf-8") as o:
    for p in sorted(counts):
        for why, n in counts[p].items():
            o.write("%s\t%s\t%d\n" % (p, why, n))
            tot += n
by = {}
for p in counts:
    for why, n in counts[p].items():
        by.setdefault(why, [0, 0])
        by[why][0] += 1
        by[why][1] += n
for why, (nf, n) in sorted(by.items()):
    print("edits %-22s %4d files %5d changes" % (why, nf, n))
print("total %d changes in %d files; log %s" % (tot, len(counts), log))
print("kept (not the checkout): paths after " + ", ".join("$%s x%d" % kv for kv in sorted(EDIT_SKIPS.items())))
if missing:
    print("EXPLICIT EDITS THAT DID NOT APPLY (%d):" % len(missing))
    for m in missing:
        print("  " + m)
    sys.exit(3)
