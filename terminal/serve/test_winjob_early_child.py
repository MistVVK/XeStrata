"""serve/winjob.py + serve/lifeline.py: a process that a contained child starts in its very first moments is in the
kill-on-close job too.  lifeline.popen used to assign the child to the job only after Popen returned, so whatever the
child started before that (the `npx` / `uvx` wrappers of MCP servers start their real server at once) stayed outside the
job and survived a hard kill of the server.  Now the child is created suspended, assigned, then resumed.

The window is made wide on purpose: the assignment is delayed by 1.5 s (a busy machine can delay it too), during which a
child that already runs has long started its own child.  Then the job is ended (winjob.terminate, what the OS does when
the server dies) and the processes must be gone.  They are held by handle, so a reused PID cannot pass.  Beside that:
the caller's creation flags, pipes, cwd and env pass through; a refused assignment and an error before the resume leave
no child suspended; and a server that itself runs inside a job (no breakaway) still starts and resumes its child.

Nothing of the real install is touched: the per-user state folders point at a temp folder and no launch record is
written (isolated()).

    python -m unittest serve.test_winjob_early_child      (Windows only; skipped elsewhere)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# the child: starts a grandchild at once, says its pid, then waits
CHILD = ("import subprocess, sys, time; g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
         "print(g.pid, flush=True); time.sleep(120)")
# the echo child: one stdin line back in capitals, its cwd, one variable of its env, a line on stderr
ECHO = ("import os, sys; line = sys.stdin.readline().strip(); print(line.upper()); print(os.getcwd()); "
        "print(os.environ.get('STRATA_T_ECHO', '-')); print('to stderr', file=sys.stderr)")

_PROCESS_TERMINATE, _SYNCHRONIZE, _QUERY = 0x0001, 0x00100000, 0x1000

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _k32.WaitForSingleObject.restype = wintypes.DWORD
    _k32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    _k32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _k32.CloseHandle.argtypes = (wintypes.HANDLE,)


class Held:
    def __init__(self, pid: int, label: str):
        self.label = label
        self.handle = _k32.OpenProcess(_SYNCHRONIZE | _QUERY | _PROCESS_TERMINATE, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def exited(self, wait_ms: int = 0) -> bool:
        return _k32.WaitForSingleObject(self.handle, wait_ms) == 0

    def end(self):
        if self.handle:
            if not self.exited():
                _k32.TerminateProcess(self.handle, 1)
            _k32.CloseHandle(self.handle)
            self.handle = None


class isolated:
    """The per-user state folders in a temp folder and no launch record: nothing of the real install is touched."""

    def __enter__(self):
        from strata_control import procs
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [mock.patch.dict(os.environ, {"LOCALAPPDATA": self.tmp.name, "APPDATA": self.tmp.name,
                                                    "XDG_STATE_HOME": self.tmp.name}),
                        mock.patch.object(procs, "write_record", lambda *a, **k: None)]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()
        return False


def late(real_contain):
    def slow_contain(proc):
        time.sleep(1.5)                           # a late assignment: the child has started its own child by now
        return real_contain(proc)
    return slow_contain


@unittest.skipUnless(os.name == "nt", "job objects are Windows-only")
class EarlyGrandchild(unittest.TestCase):
    def test_a_process_started_before_the_assignment_would_have_run_is_in_the_job(self):
        from serve import lifeline, winjob
        held = []
        with isolated(), mock.patch.object(winjob, "contain", late(winjob.contain)):
            self.assertEqual(lifeline.mechanism(), "job")
            child = lifeline.popen([sys.executable, "-c", CHILD], "engine", stdout=subprocess.PIPE, text=True)
            try:
                gpid = int(child.stdout.readline().strip())
                held = [Held(child.pid, "the child"), Held(gpid, "the grandchild")]
                self.assertFalse(any(h.exited() for h in held))
                self.assertTrue(winjob.terminate(), "no job to end")   # what the OS does when the server dies
                left = [h.label for h in held if not h.exited(10000)]
                self.assertEqual(left, [], "outside the job, so it survived the end of the job")
            finally:
                for h in held:
                    h.end()
                if child.poll() is None:
                    child.kill()
                child.wait(10)

    def test_a_cmd_shim_that_starts_node_is_ended_with_its_node(self):
        """Close to the report: an MCP server started through a .cmd wrapper (npx / uvx are such shims) whose real program
        is a long-running node.  With the assignment late, node used to be outside the job and survive."""
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not on PATH here")
        from serve import lifeline, winjob
        held = []
        with isolated() as iso, mock.patch.object(winjob, "contain", late(winjob.contain)):
            shim = Path(iso.tmp.name) / "fake-npx.cmd"
            shim.write_text('@echo off\r\n"%s" -e "console.log(process.pid); setInterval(function () {}, 1000)"\r\n' % node,
                            encoding="ascii")
            child = lifeline.popen([str(shim)], "mcp", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                npid = int(child.stdout.readline().strip())
                held = [Held(child.pid, "cmd.exe"), Held(npid, "node")]
                self.assertTrue(winjob.terminate(), "no job to end")
                self.assertEqual([h.label for h in held if not h.exited(10000)], [], "survived the end of the job")
            finally:
                for h in held:
                    h.end()
                if child.poll() is None:
                    child.kill()
                child.wait(10)


@unittest.skipUnless(os.name == "nt", "job objects are Windows-only")
class FlagsPipesAndRefusals(unittest.TestCase):
    def start(self, role, **kw):
        """lifeline.popen of the echo child; returns (the kwargs that reached subprocess.Popen, stdout lines, stderr, exit
        code, the cwd given, the env given)."""
        from serve import lifeline
        seen = {}
        real_popen = subprocess.Popen

        def recording(args, **k):
            seen.update(k)
            return real_popen(args, **k)
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "STRATA_T_ECHO": "through"}
            with mock.patch.object(subprocess, "Popen", recording):
                child = lifeline.popen([sys.executable, "-c", ECHO], role, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, cwd=tmp, env=env, **kw)
            out, err = child.communicate("hello\n", timeout=60)
            return seen, out.splitlines(), err.strip(), child.returncode, tmp, env

    def test_all_three_pipes_cwd_and_env_pass_through_and_the_child_runs(self):
        from serve import winjob
        with isolated():
            seen, out, err, rc, tmp, env = self.start("mcp", creationflags=subprocess.CREATE_NO_WINDOW)
        self.assertEqual(rc, 0)
        self.assertEqual(out[0], "HELLO")                         # resumed, and stdin / stdout work
        self.assertEqual(os.path.normcase(os.path.realpath(out[1])), os.path.normcase(os.path.realpath(tmp)))
        self.assertEqual(out[2], "through")
        self.assertEqual(err, "to stderr")
        self.assertIs(seen["env"], env)
        self.assertEqual((seen["stdin"], seen["stdout"], seen["stderr"]), (subprocess.PIPE,) * 3)
        self.assertEqual(seen["creationflags"], subprocess.CREATE_NO_WINDOW | winjob.CREATE_SUSPENDED)

    def test_the_callers_flags_are_kept_beside_the_suspended_start(self):
        from serve import winjob
        with isolated():
            seen, out, _, rc, _, _ = self.start("engine", creationflags=subprocess.CREATE_NO_WINDOW)
        self.assertEqual((rc, out[0]), (0, "HELLO"))
        self.assertEqual(seen["creationflags"], subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP |
                         winjob.CREATE_SUSPENDED)                  # the engine role's own group flag too

    def test_a_refused_assignment_still_starts_and_resumes_the_child(self):
        from serve import winjob
        with isolated(), mock.patch.object(winjob._k32, "AssignProcessToJobObject", lambda *a: 0):
            _, out, _, rc, _, _ = self.start("mcp")
        self.assertEqual((rc, out[0]), (0, "HELLO"))

    def test_an_error_between_the_start_and_the_resume_leaves_no_child(self):
        from serve import lifeline, winjob
        made = []
        real_popen = subprocess.Popen

        def recording(args, **k):
            made.append(real_popen(args, **k))
            return made[-1]

        def broken(proc):
            raise RuntimeError("assignment exploded")
        with isolated(), mock.patch.object(subprocess, "Popen", recording), mock.patch.object(winjob, "contain", broken):
            with self.assertRaises(RuntimeError):
                lifeline.popen([sys.executable, "-c", "import time; time.sleep(60)"], "mcp")
        self.assertEqual(len(made), 1)
        self.assertIsNotNone(made[0].poll(), "the child was left behind (suspended or running)")


# the server itself inside a job of its own (a terminal, a service wrapper, the app, a CI runner): kill-on-close and no
# breakaway; it starts the echo child through lifeline.popen and reports what came back
INSIDE_A_JOB = r'''
import ctypes, os, subprocess, sys
from ctypes import wintypes
sys.path.insert(0, ROOT)
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.CreateJobObjectW.restype = wintypes.HANDLE
k.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
k.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
k.GetCurrentProcess.restype = wintypes.HANDLE
from serve import winjob
job = k.CreateJobObjectW(None, None)
info = winjob._ExtendedLimits()
info.BasicLimitInformation.LimitFlags = 0x2000                 # kill on close; no BREAKAWAY_OK, no SILENT_BREAKAWAY_OK
assert k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
assert k.AssignProcessToJobObject(job, k.GetCurrentProcess()), ctypes.get_last_error()
from strata_control import procs
procs.write_record = lambda *a, **kw: None
from serve import lifeline
child = lifeline.popen([sys.executable, "-c", ECHO], "mcp", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
out, err = child.communicate("nested\n", timeout=60)
print("RESULT", child.returncode, out.splitlines()[0], bool(winjob._job))
'''


@unittest.skipUnless(os.name == "nt", "job objects are Windows-only")
class ServerInsideAJob(unittest.TestCase):
    def test_a_server_that_runs_in_a_job_of_its_own_still_starts_and_resumes_its_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "LOCALAPPDATA": tmp, "APPDATA": tmp, "XDG_STATE_HOME": tmp}
            code = "ROOT = %r\nECHO = %r\n" % (str(ROOT), ECHO) + INSIDE_A_JOB
            done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        result = [line for line in done.stdout.splitlines() if line.startswith("RESULT")]
        self.assertEqual(result[:1], ["RESULT 0 NESTED True"], done.stdout + done.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
