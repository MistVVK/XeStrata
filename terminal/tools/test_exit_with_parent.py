"""tools/test_exit_with_parent.py: the model packer ends when setup (its parent) ends (the app quit during "Prepare and load").

    python -m unittest tools/test_exit_with_parent.py -v
"""
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _alive(pid: int) -> bool:
    if os.name == "nt":
        return str(pid) in subprocess.run(["tasklist", "/FI", "PID eq %d" % pid], capture_output=True, text=True).stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                                # a zombie has ended
        return Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


class ExitWithParent(unittest.TestCase):
    def test_only_when_setup_asks_for_it(self):
        sys.path.insert(0, str(HERE))
        import exit_with_parent as E
        self.assertFalse(E.exit_with_parent(env={}))
        self.assertFalse(E.exit_with_parent(env={"STRATA_EXIT_WITH_PARENT": "0"}))

    def test_a_packer_ends_with_the_process_that_started_it(self):
        """setup.py killed: iq_pack, its child, must not go on writing (a grandchild whose parent is gone)."""
        child_code = ("import sys,time; sys.path.insert(0, %r); import exit_with_parent as E; E.exit_with_parent(poll=0.2); "
                      "print('up', flush=True); time.sleep(60)" % str(HERE))
        parent_code = ("import subprocess,sys,os; env=dict(os.environ, STRATA_EXIT_WITH_PARENT='1'); "
                       "p=subprocess.Popen([sys.executable,'-c',%r], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True); "
                       "print(p.pid, flush=True); p.stdout.readline(); print('ready', flush=True); sys.stdout.flush(); os._exit(0)" % child_code)
        r = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True)
        pid = int(r.stdout.readline())
        r.stdout.readline()
        r.wait(30)
        end = time.time() + 15
        alive = True
        while time.time() < end and alive:
            time.sleep(0.3)
            alive = _alive(pid)
        if alive:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
            else:
                os.kill(pid, 9)
        self.assertFalse(alive, "the packer kept running after setup ended")

    def test_the_control_cli_ends_with_the_host_that_started_it(self):
        """strata_control/parentwatch.py (strata.py calls it first): a `strata drafting fetch` the host asked for ends when the host is gone."""
        root = HERE.parent
        child_code = ("import sys,time; sys.path.insert(0, %r); from strata_control.parentwatch import exit_with_parent as w; w(poll=0.2); "
                      "print('up', flush=True); time.sleep(60)" % str(root))
        parent_code = ("import subprocess,sys,os; env=dict(os.environ, STRATA_EXIT_WITH_PARENT='1'); "
                       "p=subprocess.Popen([sys.executable,'-c',%r], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True); "
                       "print(p.pid, flush=True); p.stdout.readline(); print('ready', flush=True); sys.stdout.flush(); os._exit(0)" % child_code)
        r = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True)
        pid = int(r.stdout.readline())
        r.stdout.readline()
        r.wait(30)
        end = time.time() + 15
        alive = True
        while time.time() < end and alive:
            time.sleep(0.3)
            alive = _alive(pid)
        if alive:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
            else:
                os.kill(pid, 9)
        self.assertFalse(alive, "the Control program kept running after its host ended")
        text = (root / "strata.py").read_text(encoding="utf-8")
        self.assertLess(text.index("exit_with_parent()"), text.index("refuse_old_updater(sys.argv"))

    def test_without_the_variable_nothing_watches(self):
        sys.path.insert(0, str(HERE.parent))
        from strata_control import parentwatch
        self.assertFalse(parentwatch.exit_with_parent(env={}))

    def test_iq_pack_asks_for_it(self):
        """iq_pack.main calls it before it opens the model (a source check: iq_pack needs numpy and a model to run)."""
        text = (HERE / "iq_pack.py").read_text(encoding="utf-8")
        self.assertIn("from exit_with_parent import exit_with_parent", text)
        self.assertLess(text.index("exit_with_parent()"), text.index("G.GGUFFile(src)"))


if __name__ == "__main__":
    unittest.main()
