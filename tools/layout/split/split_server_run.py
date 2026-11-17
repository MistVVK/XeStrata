"""Run the whole 5.2 split of serve/server.py on a checkout, start to end (re-runnable on any head; nothing is committed).

    python -I split_server_run.py <terminal folder of a MOVED checkout> <work folder for the names file>

  1. the names tests replace: patched_py.py for `server` and for every alias the tests import it under (S, _server, srv ...);
  2. split_py.py: the closure of what can move unchanged -> serve/server_parts/, server.py re-exports it (exit 1 on any check failure,
     exit 2 when a name is written through globals() with a computed key);
  3. the one test table that counts process-start sites per file (tests/terminal/test_no_debugger_children.py): the counts are read
     from the test's own spawn_sites() on the split tree and written into STRATA_PROGRAMS (post_split_server.py).
Then run the gates: py-tests, py-serve, py-tools, docs (Linux), the path scan, names_of.py before / after.
"""
import importlib.util
import os
import re
import subprocess
import sys

root, work = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(work, exist_ok=True)
py = [sys.executable, "-I"]


def run(args, **kw):
    r = subprocess.run(py + args, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
    return r


aliases = {"server"}
for d in ("serve", "tests", "tools", "strata_cli", "strata_control"):
    for dp, dn, fn in os.walk(os.path.join(root, d)):
        dn[:] = [x for x in dn if x not in ("__pycache__", "node_modules")]
        for f in fn:
            if f.endswith(".py"):
                t = open(os.path.join(dp, f), encoding="utf-8", errors="replace").read()
                aliases |= set(re.findall(r"import\s+serve\.server\s+as\s+(\w+)", t))
                aliases |= set(re.findall(r"from\s+serve\s+import\s+[^\n]*?\bserver\s+as\s+(\w+)", t))
names_file = os.path.join(work, "patched_server.txt")
lines = []
for a in sorted(aliases):
    r = run([os.path.join(here, "patched_py.py"), a, "serve", "tests", "tools", "strata_cli", "strata_control"], cwd=root)
    lines += [l for l in r.stdout.splitlines() if l.startswith("P ")]
open(names_file, "w", encoding="utf-8").write("patched names (aliases: %s)\n" % ", ".join(sorted(aliases)) + "\n".join(sorted(set(lines))) + "\n")
print("replaced names: %d (module aliases read: %s)" % (len(set(lines)), ", ".join(sorted(aliases))))
r = run([os.path.join(here, "split_py.py"), root, "serve/server.py", "serve/server_parts", "serve.server_parts", names_file])
print(r.stdout.strip())
if r.returncode:
    print(r.stderr.strip())
    sys.exit(r.returncode)
# the spawn-site table, from the test's own count on the split tree
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("nodebugger", os.path.join(root, "tests", "terminal", "test_no_debugger_children.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sites = mod.spawn_sites()
left, moved = sites.get("serve/server.py", 0), sites.get("serve/server_parts/engines.py", 0)
others = {k: v for k, v in sites.items() if k.startswith("serve/server_parts/") and k != "serve/server_parts/engines.py"}
if others:
    print("spawn sites in other new modules, list them by hand in STRATA_PROGRAMS:", others)
    sys.exit(3)
r = run([os.path.join(here, "post_split_server.py"), root, str(left), str(moved)])
print(r.stdout.strip() or r.stderr.strip())
sys.exit(r.returncode)
