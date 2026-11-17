// End a program this host started together with everything that program started (#4529 #4570).
//
// Windows: `taskkill /T` walks the tree. Linux and macOS have no such call, and signalling the child alone leaves its own children running (they are
// re-parented to init). A child started with `detached: ownGroup()` leads a process group of its own, so a signal to the negative pid reaches the
// whole group and nothing else: the group is the ownership record, no other process (a same-named one included) is ever signalled by name.
import { spawnSync } from 'node:child_process';

/** The spawn option that makes a child lead its own process group on Linux and macOS (Windows needs none: taskkill /T). */
export const ownGroup = () => process.platform !== 'win32';

/** Whether the group the child leads still has a member. */
export function groupAlive(child) {
  if (process.platform === 'win32' || !child?.pid) return false;
  try { process.kill(-child.pid, 0); return true; } catch (error) { return error?.code === 'EPERM'; }
}

/**
 * Signal the child's whole tree once. A child that does not lead its own group (it was not started with ownGroup()) is signalled alone, which is what
 * every caller did before.
 */
export function endTree(child, signal = 'SIGTERM') {
  if (!child) return;
  try {
    if (process.platform === 'win32') {
      if (child.pid) spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore', timeout: 10000 });
      return;
    }
    if (child.pid) {
      try { process.kill(-child.pid, signal); return; } catch { /* no such group: it did not lead one, or it is gone */ }
    }
    // A child that never started has no pid. On macOS, kill() on it right after a failed spawn signals the caller's own process group (pid 0), so
    // it is left alone: there is nothing to end.
    if (child.pid) child.kill(signal);
  } catch { /* already gone */ }
}

/** SIGTERM to the tree, then SIGKILL to whatever of its group is still there after `graceMs`. Returns the timer so a caller may clear it. */
export function endTreeThenKill(child, graceMs = 2000) {
  endTree(child, 'SIGTERM');
  if (process.platform === 'win32' || !child?.pid) return null;
  const timer = setTimeout(() => { if (groupAlive(child)) endTree(child, 'SIGKILL'); }, graceMs);
  timer.unref?.();
  return timer;
}
