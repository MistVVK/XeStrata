"""MAP-0216 (copy of MAP-0215): the Terminal archive's member list before and after the move (no engine, no wheels: --skip-deps).

python -I archive_lists.py <before.tar> <after.tar> <work dir> <target> [<python cache dir>]
Unpacks each git archive into <work>/<arm>/src, runs its package_terminal.py (packaging/ before, tools/packaging/ after) for <target>
with --skip-deps, and writes <work>/<arm>-members.txt (archive member names without the top folder, sorted) and
<work>/compare.txt (only-before / only-after lines).  The work dir must not exist.
"""
import os
import subprocess
import sys
import tarfile
import zipfile

before, after, work, target = sys.argv[1:5]
cache = sys.argv[5] if len(sys.argv) > 5 else None
if os.path.exists(work):
    sys.exit("work dir exists: " + work)
os.makedirs(work)
lists = {}
for arm, tar, script in (("before", before, "packaging/package_terminal.py"), ("after", after, "tools/packaging/package_terminal.py")):
    src = os.path.join(work, arm, "src")
    os.makedirs(src)
    with tarfile.open(tar) as tf:
        tf.extractall(src)
    out = os.path.join(work, arm, "out")
    cmd = [sys.executable, os.path.join(src, script), "--out", out, "--target", target, "--skip-deps", "--version", "0.0.0-map0216",
           "--commit", "0" * 40]
    if cache:
        cmd += ["--cache", cache]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("%s: exit %d\n%s%s" % (arm, r.returncode, r.stdout[-1500:], r.stderr[-3000:]))
    if r.returncode:
        continue
    names = []
    for f in os.listdir(out):
        p = os.path.join(out, f)
        if f.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
        elif f.endswith(".tar.gz"):
            with tarfile.open(p) as t:
                names = [m.name + ("/" if m.isdir() else "") for m in t.getmembers()]
    names = sorted({n.split("/", 1)[1] for n in names if "/" in n and n.split("/", 1)[1] and not n.endswith("/")})
    # the interpreter is the same pinned archive in both: keep the application part only
    app = [n for n in names if not n.startswith("python/")]
    lists[arm] = app
    with open(os.path.join(work, arm + "-members.txt"), "w", encoding="utf-8") as o:
        o.write("\n".join(app) + "\n")
    print("%s: %d members (%d outside python/)" % (arm, len(names), len(app)))
if len(lists) == 2:
    b, a = set(lists["before"]), set(lists["after"])
    with open(os.path.join(work, "compare.txt"), "w", encoding="utf-8") as o:
        for n in sorted(b - a):
            o.write("only-before " + n + "\n")
        for n in sorted(a - b):
            o.write("only-after  " + n + "\n")
    print("compare: %d only before, %d only after, %d in both" % (len(b - a), len(a - b), len(a & b)))
    for n in sorted(b - a)[:30]:
        print("  only-before", n)
    for n in sorted(a - b)[:30]:
        print("  only-after ", n)
