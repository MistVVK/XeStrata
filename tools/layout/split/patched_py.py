"""Which names of a module do tests REPLACE (patch / assign / setattr / monkeypatch) rather than only read?

    python -I patched_py.py <module import name> <tests dir> [<tests dir> ...]
A pure move that re-exports a name keeps every read working; a replaced name is only seen by code that looks it up in the original
module's namespace, so code that moves away from the module and calls the name would stop seeing the replacement.
"""
import os
import re
import sys

mod = sys.argv[1]
dirs = sys.argv[2:]
M = re.escape(mod)
PATTERNS = [
    re.compile(r"""patch\(\s*["'](?:\w+\.)*%s\.([A-Za-z_]\w*)["']""" % M),
    re.compile(r"""patch\.object\(\s*(?:\w+\.)*%s\s*,\s*["']([A-Za-z_]\w*)["']""" % M),
    re.compile(r"""\b%s\.([A-Za-z_]\w*)\s*=[^=]""" % M),
    re.compile(r"""setattr\(\s*(?:\w+\.)*%s\s*,\s*["']([A-Za-z_]\w*)["']""" % M),
    re.compile(r"""monkeypatch\.setattr\(\s*["'](?:\w+\.)*%s\.([A-Za-z_]\w*)["']""" % M),
    re.compile(r"""@mock\.patch\(\s*["'](?:\w+\.)*%s\.([A-Za-z_]\w*)["']""" % M),
    re.compile(r"""\b(?:\w+\.)+%s\.([A-Za-z_]\w*)\s*=[^=]""" % M),      # serve.server.NAME = value
]
# names a test module binds with `from <mod> import a, b` and then replaces are not seen here; they are listed as "imported by"
IMP = re.compile(r"\bfrom\s+%s\s+import\s+\(?([^)\n]+)" % M)
patched = {}
imported = {}
for d in dirs:
    for root, _x, files in os.walk(d):
        if "node_modules" in root or "__pycache__" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            try:
                t = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if mod not in t:
                continue
            for rx in PATTERNS:
                for m in rx.finditer(t):
                    patched.setdefault(m.group(1), set()).add(f)
            for m in IMP.finditer(t):
                for n in m.group(1).split(","):
                    n = n.strip().split(" as ")[0].strip()
                    if n:
                        imported.setdefault(n, set()).add(f)
print("patched/assigned names: %d" % len(patched))
for n in sorted(patched):
    print("P %s %d" % (n, len(patched[n])))
print("from-imported names: %d" % len(imported))
for n in sorted(imported):
    print("I %s %d" % (n, len(imported[n])))
