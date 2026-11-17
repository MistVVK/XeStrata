"""Which top-level names of a module can move out unchanged (pure move)?

    python -I closure_py.py <module file> <patched names file (patched_py.py output)>

A name can move when it is not replaced by any test, is never rebound by a `global` statement, and everything it uses at module
level (other top-level names, imported names) is itself movable or an import that the new module can repeat. A name that uses a
replaced name, or a name that stays, stays. The rest of the module keeps working through a re-export (from new_module import name).
Prints the movable set grouped by banner section, with line counts, and the names that block the biggest groups.
"""
import ast
import re
import sys

path, pfile = sys.argv[1], sys.argv[2]
src = open(path, encoding="utf-8").read()
lines = src.split("\n")
tree = ast.parse(src)
patched = {l.split()[1] for l in open(pfile, encoding="utf-8") if l.startswith("P ")}
BANNER = re.compile(r"^# -{20,} ?(.*)$")
banners = [(i + 1, BANNER.match(l).group(1).strip()) for i, l in enumerate(lines) if BANNER.match(l)]


def section_of(lineno):
    cur = "(head)"
    for ln, name in banners:
        if ln <= lineno:
            cur = name
    return cur


defs = {}
node_of = {}
for node in tree.body:
    names = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names = [node.name]
    elif isinstance(node, ast.Assign):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names = [node.target.id]
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) if names else 0
    for n in names:
        defs[n] = (start, node.end_lineno, section_of(start))
        node_of[n] = node

imports = set()
for node in tree.body:
    if isinstance(node, ast.Import):
        for a in node.names:
            imports.add((a.asname or a.name).split(".")[0])
    elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
        for a in node.names:
            imports.add(a.asname or a.name)

# names rebound with `global` anywhere
rebound = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Global):
        rebound.update(n.names)

# module-level statements that are not definitions (calls, ifs, try blocks) use names too: those names must stay put
other_uses = set()
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        continue
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        continue
    for x in ast.walk(node):
        if isinstance(x, ast.Name):
            other_uses.add(x.id)


def free_names(node):
    return {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}


deps = {}
for n, node in node_of.items():
    deps[n] = (free_names(node) & set(defs)) - {n}
    deps[n] |= (free_names(node) & imports & patched)       # a patched import name (subprocess, time ...) is a dependency that cannot be repeated

movable = {n for n in defs if n not in patched and n not in rebound and n not in other_uses}
changed = True
while changed:
    changed = False
    for n in list(movable):
        for d in deps[n]:
            if d not in movable:
                movable.discard(n)
                changed = True
                break
print("module %s: %d top-level names; patched %d; rebound by global %d; used by module-level statements %d; MOVABLE %d" % (
    path, len(defs), len(patched & set(defs)), len(rebound & set(defs)), len(other_uses & set(defs)), len(movable)))
tot = 0
secs = {}
for n in movable:
    s, e, sec = defs[n]
    secs.setdefault(sec, []).append((s, e, n))
for sec, items in sorted(secs.items(), key=lambda x: -sum(e - s + 1 for s, e, _ in x[1])):
    ln = sum(e - s + 1 for s, e, _ in items)
    tot += ln
    allsec = [n for n in defs if defs[n][2] == sec]
    print("  %-46s movable %3d of %3d names, %5d lines" % (sec[:46], len(items), len(allsec), ln))
print("movable lines: %d of %d" % (tot, len(lines)))
# the biggest blockers: names not movable that other not-movable names (in big sections) wait for
block = {}
for n in defs:
    if n in movable:
        continue
    for d in deps[n]:
        if d not in movable:
            block[d] = block.get(d, 0) + 1
print("top blockers (name: how many unmovable names wait for it; reason)")
for d, c in sorted(block.items(), key=lambda x: -x[1])[:25]:
    why = "patched" if d in patched else "rebound" if d in rebound else "module-level use" if d in other_uses else "waits for others"
    print("  %-34s %3d  %s" % (d, c, why))
