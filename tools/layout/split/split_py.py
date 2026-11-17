"""MAP-0216 file split (cleanup step 5.1 / 5.2): move the names of a big Python module that can move UNCHANGED into new modules.

    python -I split_py.py <repo folder holding the module> <module file, relative> <package folder, relative> <import prefix> <patched names file> [--plan]
    e.g. python -I split_py.py <wt>/terminal setup.py strata_setup strata_setup <wt>/patched_setup.txt

A PURE move: the text of every moved definition is copied unchanged (comments directly above it travel with it), the original module
imports the names back (a re-export block after its imports), so `module.name` keeps working for every caller and test. A name moves
only when that cannot change behaviour:
  - no test replaces it (the "patched names file" from patched_py.py: setup.X = ..., patch.object(setup, "X"), patch("setup.X")),
  - nothing rebinds it with `global`, no module-level statement (a call, an if, a try) uses it, it does not use __file__ / __name__ /
    globals() / locals(),
  - everything it uses at module level is itself moved (or an import the new module repeats). A name that uses a replaced name stays,
    because the new module would call its own copy and the test's replacement would no longer reach it.
The moved names are grouped by the module's banner sections; sections that depend on each other in a cycle become one new module.
After the rewrite the script CHECKS the move: every moved definition's syntax tree equals the original's, every other top-level
statement is unchanged, and no name is lost or doubled. It exits 1 on any difference. Re-runnable: the same input gives the same output.
"""
import ast
import os
import re
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
PLAN = "--plan" in sys.argv
base, mfile, pkg, prefix, pfile = args[:5]
path = os.path.join(base, mfile)
src = open(path, encoding="utf-8", newline="").read()
if "\r" in src:
    sys.exit("CRLF in " + path)
lines = src.split("\n")
tree = ast.parse(src)
patched = {l.split()[1] for l in open(pfile, encoding="utf-8") if l.startswith("P ")}
BANNER = re.compile(r"^# -{20,} ?(.*)$")
banners = [(i + 1, BANNER.match(l).group(1).strip()) for i, l in enumerate(lines) if BANNER.match(l)]


def section_of(lineno):
    cur = "head"
    for ln, name in banners:
        if ln <= lineno:
            cur = name
    return cur


def slug(s):
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s or "part"


# ------------------------------------------------------------------------------- what is defined, imported, bound elsewhere
defs, node_of, order = {}, {}, []
for node in tree.body:
    names = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names = [node.name]
    elif isinstance(node, ast.Assign):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if len(names) != len(node.targets):
            names = []          # tuple targets, attribute targets: stays
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names = [node.target.id]
    if not names:
        continue
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    for n in names:
        defs.setdefault(n, []).append(node)
    order.append((start, node, names))
multi = {n for n, v in defs.items() if len(v) > 1}        # defined twice: stays (the later one wins; moving one changes that)

imports = {}       # bound name -> the top-level import node
for node in tree.body:
    if isinstance(node, ast.Import):
        for a in node.names:
            imports[(a.asname or a.name).split(".")[0]] = node
    elif isinstance(node, ast.ImportFrom):
        for a in node.names:
            imports[a.asname or a.name] = node
future_ann = any(isinstance(n, ast.ImportFrom) and n.module == "__future__" for n in tree.body)

other_bound, other_uses = set(), set()
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        continue
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and any(n for _s, nn, nms in order if nn is node for n in nms):
        continue
    for x in ast.walk(node):
        if isinstance(x, ast.Name):
            (other_bound if isinstance(x.ctx, ast.Store) else other_uses).add(x.id)
        elif isinstance(x, (ast.Import, ast.ImportFrom)):
            for a in x.names:
                other_bound.add((a.asname or a.name).split(".")[0])
rebound = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Global):
        rebound.update(n.names)
runtime_why = {n: "global statement" for n in rebound}


def _is_globals_call(x):
    return isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in ("globals", "vars") and (x.func.id == "globals" or not x.args)


def _module_dict(x):
    """sys.modules[...].__dict__ (a module's dictionary), not an instance's self.__dict__."""
    return (isinstance(x, ast.Attribute) and x.attr == "__dict__" and isinstance(x.value, ast.Subscript)
            and isinstance(x.value.value, ast.Attribute) and x.value.value.attr == "modules")


# (2) names written at RUN TIME through the module's dictionary: globals()["NAME"] = ..., globals().update(...), module.__dict__[...] ...
dynamic = []
for n in ast.walk(tree):
    targets = []
    if isinstance(n, (ast.Assign, ast.AugAssign, ast.Delete)):
        targets = n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]
    for t in targets:
        if isinstance(t, ast.Subscript) and (_is_globals_call(t.value) or _module_dict(t.value)):
            if isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
                runtime_why[t.slice.value] = "written through globals()/__dict__"
            else:
                dynamic.append(("a subscript write into globals() / __dict__ with a computed key", t.lineno))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("update", "setdefault", "pop", "__setitem__") \
            and (_is_globals_call(n.func.value) or _module_dict(n.func.value)):
        dynamic.append(("%s() on globals() / __dict__" % n.func.attr, n.lineno))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr" and n.args:
        a0 = n.args[0]
        if isinstance(a0, ast.Subscript) or (isinstance(a0, ast.Attribute) and a0.attr == "modules"):
            dynamic.append(("setattr() on a module looked up by name", n.lineno))

# (3) the module's own names assigned FROM OTHER non-test code: `server.NAME = x`, setattr(server, "NAME", x), serve.server.NAME = x
dotted = os.path.splitext(mfile)[0].replace("/", ".")
short = dotted.rsplit(".", 1)[-1]
parent = dotted.rsplit(".", 1)[0] if "." in dotted else ""
ext_files = 0
ext_hits = []
for root_, dirs_, files_ in os.walk(base):
    dirs_[:] = [d for d in dirs_ if d not in ("__pycache__", "node_modules", ".git", "tests", ".venv")]
    for f_ in files_:
        if not f_.endswith(".py") or f_.startswith("test_") or f_ == "conftest.py":
            continue
        p_ = os.path.join(root_, f_)
        if os.path.abspath(p_) == os.path.abspath(path):
            continue
        t_ = open(p_, encoding="utf-8", errors="replace").read()
        if short not in t_:
            continue
        ext_files += 1
        aliases = {short} if parent else set()
        aliases |= set(re.findall(r"import\s+%s\s+as\s+(\w+)" % re.escape(dotted), t_))
        if parent:
            aliases |= set(re.findall(r"from\s+%s\s+import\s+[^\n]*?\b%s\s+as\s+(\w+)" % (re.escape(parent), re.escape(short)), t_))
        pats = [r"\b(?:%s)\.(\w+)\s*(?:[-+*/|&%%]?=)(?!=)" % "|".join(map(re.escape, aliases)) if aliases else None,
                r"\b%s\.(\w+)\s*(?:[-+*/|&%%]?=)(?!=)" % re.escape(dotted),
                r"setattr\(\s*(?:%s)\s*,\s*[\"'](\w+)[\"']" % "|".join(map(re.escape, aliases | {dotted})),
                r"\b(?:%s)\.__dict__\b" % "|".join(map(re.escape, aliases | {dotted}))]
        for k_, pat in enumerate(pats):
            if not pat:
                continue
            for m_ in re.finditer(pat, t_):
                if k_ == 3:
                    dynamic.append(("%s.__dict__ in %s" % (short, os.path.relpath(p_, base)), t_[:m_.start()].count("\n") + 1))
                else:
                    runtime_why[m_.group(1)] = "assigned from %s" % os.path.relpath(p_, base).replace("\\", "/")
                    ext_hits.append((m_.group(1), os.path.relpath(p_, base).replace("\\", "/")))
if dynamic and "--allow-dynamic" not in sys.argv:
    for w_, ln_ in dynamic:
        print("DYNAMIC NAME WRITE (cannot tell which names): %s, line %d" % (w_, ln_))
    sys.exit(2)
rebound = set(runtime_why)
LOCATION = {"__file__", "__name__", "__doc__", "__spec__", "__package__", "globals", "locals", "vars", "__builtins__"}


def names_in(node):
    return {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}


deps, bad = {}, set()
for n, nodes in defs.items():
    d = set()
    for node in nodes:
        used = names_in(node)
        if used & LOCATION:
            bad.add(n)
        d |= (used & set(defs)) - {n}
        d |= used & other_bound                              # bound by a try / if block: stays, so does what needs it
        d |= used & imports.keys() & patched                 # an import name that a test replaces (subprocess, time ...)
    deps[n] = d
movable = {n for n in defs if n not in patched and n not in rebound and n not in other_uses and n not in multi and n not in bad
           and n not in other_bound}
changed = True
while changed:
    changed = False
    for n in list(movable):
        if any(d not in movable for d in deps[n]):
            movable.discard(n)
            changed = True

# ------------------------------------------------------------------------------- group by banner section, merge cycles
sec_of = {n: section_of(min([defs[n][0].lineno] + [d.lineno for d in getattr(defs[n][0], "decorator_list", [])])) for n in defs}
secs = {}
for n in movable:
    secs.setdefault(sec_of[n], set()).add(n)
sec_dep = {s: {sec_of[d] for n in ns for d in deps[n] if d in movable and sec_of[d] != s} for s, ns in secs.items()}
# strongly connected components over sections (small graph: iterate to a fixpoint on reachability)
reach = {s: set(v) for s, v in sec_dep.items()}
for _ in range(len(secs) + 1):
    for s in reach:
        for t in list(reach[s]):
            reach[s] |= reach.get(t, set())
group = {}
for s in secs:
    cyc = sorted({t for t in reach[s] if s in reach.get(t, ())} | {s})
    group[s] = cyc[0]
modules = {}
for s, ns in secs.items():
    modules.setdefault(group[s], set()).update(ns)
NICE = {"head": "base", "http": "http_api", "openai": "openai_api", "anthropic": "anthropic_api"}     # no module named like a stdlib one
modname = {g: NICE.get(slug(g), slug(g)) for g in modules}
# a module named like another one gets a number
seen = {}
for g in sorted(modname):
    k = modname[g]
    seen[k] = seen.get(k, 0) + 1
    if seen[k] > 1:
        modname[g] = "%s_%d" % (k, seen[k])
mod_of = {n: modname[group[sec_of[n]]] for n in movable}


def seg_range(node):
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    i = start - 1
    while i > 0 and lines[i - 1].startswith("#") and not BANNER.match(lines[i - 1]):
        i -= 1                                            # comment lines directly above travel with the definition
    return i + 1, node.end_lineno


# ------------------------------------------------------------------------------- build the new modules
moved_nodes = []                                          # (module, start, end, names, node)
for start, node, names in order:
    if all(n in movable for n in names):
        s, e = seg_range(node)
        moved_nodes.append((mod_of[names[0]], s, e, names, node))
by_mod = {}
for m, s, e, names, node in moved_nodes:
    by_mod.setdefault(m, []).append((s, e, names, node))


def import_text(node, keep):
    """The import statement `node` restricted to the bound names in `keep` (alias forms kept)."""
    parts = []
    for a in node.names:
        bound = (a.asname or a.name).split(".")[0]
        if bound in keep:
            parts.append(a.name + (" as " + a.asname if a.asname else ""))
    if not parts:
        return None
    if isinstance(node, ast.Import):
        return "import " + ", ".join(parts)
    return "from %s%s import %s" % ("." * node.level, node.module or "", ", ".join(parts))


out_files = {}
for m, items in sorted(by_mod.items()):
    used = set()
    names_here = set()
    for s, e, names, node in items:
        used |= names_in(node)
        names_here |= set(names)
    head = ['"""Moved unchanged out of %s (file split, cleanup step 5): the %s part. `%s` imports these names back,' % (mfile, [g for g in modname if modname[g] == m][0], os.path.splitext(mfile)[0].replace("/", ".")),
            'so every caller and test still reaches them as `%s.<name>`."""' % os.path.splitext(mfile)[0].replace("/", ".")]
    if future_ann:
        head.append("from __future__ import annotations")
    head.append("")
    imp_lines = []
    done = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            if id(node) in done:
                continue
            done.add(id(node))
            t = import_text(node, used - names_here)
            if t:
                imp_lines.append(t)
    cross = {}
    for n in used & movable:
        if mod_of[n] != m:
            cross.setdefault(mod_of[n], set()).add(n)
    cross_lines = ["from %s.%s import %s" % (prefix, k, ", ".join(sorted(v))) for k, v in sorted(cross.items())]
    body = []
    for s, e, names, node in items:
        body.append("\n".join(lines[s - 1:e]))
    text = "\n".join(head + imp_lines + ([""] + cross_lines if cross_lines else [])) + "\n\n\n" + "\n\n\n".join(body) + "\n"
    out_files[m] = text

# ------------------------------------------------------------------------------- rewrite the original
cut = set()
for m, s, e, names, node in moved_nodes:
    cut.update(range(s, e + 1))
last_import = max((n.end_lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and n.lineno < (order[0][0] if order else 10 ** 9)), default=0)
reexp = []
for m in sorted(by_mod):
    ns = sorted({n for s, e, names, node in by_mod[m] for n in names})
    reexp.append("from %s.%s import (  # noqa: F401  (moved unchanged, file split)\n%s)" % (prefix, m, "".join("    %s,\n" % n for n in ns)))
new_lines = []
for i, l in enumerate(lines, 1):
    if i in cut:
        continue
    new_lines.append(l)
    if i == last_import:
        new_lines.append("")
        new_lines.extend("\n".join(reexp).split("\n"))
text2 = "\n".join(new_lines)
text2 = re.sub(r"\n{4,}", "\n\n\n", text2)

# ------------------------------------------------------------------------------- the check
def dump(n):
    return ast.dump(n, include_attributes=False)


orig = {}
for start, node, names in order:
    for n in names:
        orig[n] = dump(node)
errs = []
new_tree = ast.parse(text2)
remain = {}
for node in new_tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        remain[node.name] = dump(node)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                remain[t.id] = dump(node)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        remain[node.target.id] = dump(node)
moved_dump = {}
for m, text in out_files.items():
    t = ast.parse(text)
    for node in t.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            moved_dump[node.name] = dump(node)
        elif isinstance(node, ast.Assign):
            for x in node.targets:
                if isinstance(x, ast.Name):
                    moved_dump[x.id] = dump(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            moved_dump[node.target.id] = dump(node)
for n in movable:
    if moved_dump.get(n) != orig[n]:
        errs.append("moved name differs or is missing: " + n)
    if n in remain:
        errs.append("moved name still in the original: " + n)
for n in set(orig) - movable:
    if remain.get(n) != orig[n]:
        errs.append("kept name changed or lost: " + n)
rest_before = [dump(n) for n in tree.body if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)) and any(x in movable for x in [getattr(n, "name", None)] + [t.id for t in getattr(n, "targets", []) if isinstance(t, ast.Name)] + ([n.target.id] if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) else [])))]
rest_after = [dump(n) for n in new_tree.body if not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith(prefix + "."))]
if rest_before != rest_after:
    errs.append("the remaining statements of the original are not the same as before (%d vs %d)" % (len(rest_before), len(rest_after)))

tot_moved = sum(e - s + 1 for m, s, e, names, node in moved_nodes)
print("%s: %d lines, %d top-level names, %d movable (%d lines) into %d modules under %s/" % (mfile, len(lines), len(defs), len(movable), tot_moved, len(out_files), pkg))
print("  %.1f%% of the file moves. Run-time rebinding found: %d names (global statements %d, globals()/__dict__ writes %d, assigned from other non-test code %d in %d files read: %s)" % (
    100.0 * tot_moved / len(lines), len([n for n in runtime_why if n in defs]), len([n for n, w in runtime_why.items() if w == "global statement" and n in defs]),
    len([n for n, w in runtime_why.items() if w.startswith("written")]), len({n for n, w in runtime_why.items() if w.startswith("assigned")}), ext_files,
    ", ".join(sorted({n for n, w in runtime_why.items() if w != "global statement"}))[:200] or "none"))
for m, items in sorted(by_mod.items()):
    print("  %-28s %4d names, %5d lines" % (m + ".py", sum(len(i[2]) for i in items), sum(i[1] - i[0] + 1 for i in items)))
for e in errs[:20]:
    print("CHECK FAILED:", e)
if errs:
    sys.exit(1)
print("check: moved definitions identical, other statements unchanged, no name lost or doubled")
if PLAN:
    sys.exit(0)
pkgdir = os.path.join(base, pkg)
os.makedirs(pkgdir, exist_ok=True)
init = os.path.join(pkgdir, "__init__.py")
if not os.path.exists(init):
    open(init, "w", encoding="utf-8", newline="").write('"""Parts moved unchanged out of %s (file split, cleanup step 5)."""\n' % mfile)
for m, text in out_files.items():
    open(os.path.join(pkgdir, m + ".py"), "w", encoding="utf-8", newline="").write(text)
open(path, "w", encoding="utf-8", newline="").write(text2)
print("written")
