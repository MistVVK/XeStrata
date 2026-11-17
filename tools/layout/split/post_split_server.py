"""After split_py.py moved serve/server.py's names into serve/server_parts/: the one test table that counts process-start sites per file
(tests/terminal/test_no_debugger_children.py STRATA_PROGRAMS) lists the moved ones under their new file. Exact text edit, asserted.

    python -I post_split_server.py <terminal folder> <spawn sites left in serve/server.py> <spawn sites now in serve/server_parts/engines.py>
(the counts are what the test itself reports: run it once, read "{'serve/server_parts/engines.py': N}", the rest stays in server.py)
"""
import os
import sys

root, left, moved = sys.argv[1], sys.argv[2], sys.argv[3]
path = os.path.join(root, "tests", "terminal", "test_no_debugger_children.py")
text = open(path, encoding="utf-8", newline="").read()
old = '    "serve/server.py": 7,              # the engine, the picture reader (env from child_env / vision_env), the strata-device probe, before_load commands\n'
new = ('    "serve/server.py": %s,              # the engine, the picture reader (env from child_env / vision_env), the strata-device probe, before_load commands\n'
       '    "serve/server_parts/engines.py": %s,   # (moved out of serve/server.py unchanged, file split)\n') % (left, moved)
if text.count(old) != 1:
    sys.exit("the STRATA_PROGRAMS line of serve/server.py was not found exactly once")
open(path, "w", encoding="utf-8", newline="").write(text.replace(old, new))
print("edited", path)
