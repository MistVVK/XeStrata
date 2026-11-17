"""MAP-0216 on Windows (v3): the start-file / Windows-path tests and every Windows-only test file, before or after the move,
EACH FILE IN ITS OWN pytest PROCESS.

python -I win_start_tests_0216c.py <before worktree> <after worktree> <out dir> <before|after>

Why per file (11-16 20:4x): in one combined pytest run, serve/winjob.py was first imported while an earlier test had replaced
subprocess.Popen, so winjob._REAL_POPEN (taken at import) was that stand-in and 7 tests failed with "isinstance() arg 2 must be a
type" in BOTH trees. INT's own per-shard runs pass them. Running each file alone gives every file a fresh interpreter, as CI does.
"""
import os
import re
import subprocess
import sys
import time

TESTS = ["tests/terminal/test_bench_community_amd.py", "tests/terminal/test_bench_standard.py", "tests/terminal/test_download_dedup.py",
         "tests/terminal/test_no_acquisition.py", "tests/terminal/test_packaging_pins.py", "tests/terminal/test_bundle_imports.py",
         "tests/terminal/test_bundle_tools_paths.py", "tools/test_cyrillic_paths.py", "tools/test_setup_amd.py", "tools/test_setup_choices.py",
         "tools/test_setup_cuda_repo.py", "tools/test_setup_download.py", "tools/test_setup_feed_engine.py", "tools/test_setup_find_tool.py",
         "tools/test_setup_older_gpus.py", "tools/test_setup_prompts.py", "tools/test_setup_rerun_size.py", "tools/test_setup_update.py",
         "tools/test_setup_update_gpu.py", "tools/test_windows_amd_gpu.py", "tools/test_build_cuda_windows.py",
         "serve/test_winjob.py", "serve/test_winjob_early_child.py", "tests/desktop/test_control_install.py", "tests/desktop/test_fetch_node.py",
         "tests/terminal/test_component_start.py", "tests/terminal/test_default_object_access.py", "tests/terminal/test_drafting_vocab_language.py",
         "tests/terminal/test_engine_deps.py", "tests/terminal/test_issues_022.py", "tests/terminal/test_lifecycle_matrix.py",
         "tests/terminal/test_move_dir.py", "tests/terminal/test_restart_teardown_budget.py", "tests/terminal/test_runtime.py",
         "tests/terminal/test_runtime_state_pending.py", "tests/terminal/test_runtime_state_pending_wait.py",
         "tests/terminal/test_tui_stress_screens.py", "tests/time_skip/test_time_skip.py", "tools/test_setup_oldcpu.py"]
SUMMARY = re.compile(r"=+ (.*?(passed|failed|error|skipped).*?) in [0-9.]+s")

before, after, out, only = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
for arm, tree, prefix in (("before", before, ""), ("after", after, "terminal/")):
    if arm != only:
        continue
    cwd = os.path.join(tree, prefix) if prefix else tree       # the moved tree's Python steps run from terminal/, as tools/check runs them
    t0 = time.time()
    totals = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    fails = []
    with open(os.path.join(out, arm + ".log"), "w", encoding="utf-8") as fh:
        for t in TESTS:
            path = os.path.join(tree, prefix + t)
            if not os.path.exists(path):
                fails.append("MISSING " + t)
                continue
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", path], cwd=cwd, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=600,
                               env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
            fh.write("##### %s (exit %d)\n%s\n" % (t, r.returncode, r.stdout[-20000:]))
            fh.flush()
            m = SUMMARY.search(r.stdout)
            for part in (m.group(1) if m else "").split(","):
                part = part.strip()
                for k in totals:
                    if part.endswith(" " + k) or part.endswith(" " + k + "s"):
                        totals[k] += int(part.split()[0])
            fails += [line.strip() for line in r.stdout.splitlines() if line.startswith(("FAILED", "ERROR"))]
    print("== %s: %d files, %d s, %s" % (arm, len(TESTS), time.time() - t0, totals))
    for f in fails:
        print("  " + f[:220])
