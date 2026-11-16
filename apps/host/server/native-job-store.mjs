/** Best-effort job history: a failed status save must not stop inference. */
import * as fs from 'node:fs';
import { randomUUID } from 'node:crypto';

export function createJobStore(path, {
  io = fs, warn = message => console.warn(message), now = Date.now,
  setTimer = setTimeout, clearTimer = clearTimeout,
} = {}) {
  let saved = null, pending = null, timer = null, retries = 0;
  let retryAfter = 0, warningAt = -Infinity, failed = false, closed = false;
  const delays = [250, 1000, 3000];
  const report = message => { try { warn(message); } catch { /* logging cannot stop the host */ } };

  function attempt() {
    timer = null;
    if (pending === null) return;
    const text = pending, temporary = `${path}.${process.pid}.${randomUUID()}.tmp`;
    try {
      // Write beside the destination, then replace it: preserve the last good JSON on failure.
      io.writeFileSync(temporary, text, { mode: 0o600 });
      io.renameSync(temporary, path);
      saved = text;
      pending = null;
      retries = 0;
      retryAfter = 0;
      if (failed) report('[strata host] Job history persistence recovered.');
      failed = false;
    } catch (error) {
      if (!failed || now() - warningAt >= 60000) {
        warningAt = now();
        report(`[strata host] Job history save failed (${error.code ?? 'UNKNOWN'}, errno ${error.errno ?? '?'}, ${error.syscall ?? 'write'}): ${error.message}. Keeping in-memory state; inference continues.`);
      }
      failed = true;
      if (!closed && retries < delays.length) {
        timer = setTimer(attempt, delays[retries++]);
        timer.unref?.();
      } else {
        retries = 0;
        retryAfter = now() + 30000;
      }
    } finally {
      // Only our uniquely named temporary file is eligible for cleanup.
      try { io.rmSync(temporary, { force: true }); } catch { /* try no other files */ }
    }
  }

  return {
    save(jobs) {
      if (closed) return;
      const text = JSON.stringify(jobs);
      if (text === saved) {
        pending = null;
        if (timer !== null) clearTimer(timer);
        timer = null;
        return;
      }
      pending = text; // Retries always write the newest state, never an older snapshot.
      if (timer === null && now() >= retryAfter) attempt();
    },
    close() {
      if (closed) return;
      closed = true;
      if (timer !== null) clearTimer(timer);
      timer = null;
      attempt(); // One final best-effort save, with no retry timer after shutdown.
    },
  };
}
