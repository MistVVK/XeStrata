/**
 * One long-lived PowerShell that prints a line per reading, in a loop. The monitor used to start a cold PowerShell for every sample;
 * its WMI start cost more than the probe timeout, so every query failed and WmiPrvSE burned a core (#3939).
 * The loop gives up after `maxStarts` early exits in a row, so a counter that cannot be read stays quiet.
 */
import { spawn } from 'node:child_process';

export function createPowerShellLoop({ script, onLine, spawnFn = spawn, maxStarts = 3 }) {
  let child = null, starts = 0, buffer = '';
  function start() {
    if (child || starts >= maxStarts) return;
    starts++;
    let proc;
    try { proc = spawnFn('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script], { windowsHide: true, stdio: ['ignore', 'pipe', 'ignore'] }); }
    catch { return; }
    child = proc;
    buffer = '';
    proc.on?.('error', () => { if (child === proc) child = null; });
    proc.on?.('exit', () => { if (child === proc) child = null; });
    proc.stdout?.on('data', chunk => {
      buffer += chunk;
      let i;
      while ((i = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, i);
        buffer = buffer.slice(i + 1);
        if (onLine(line)) starts = 0;
      }
    });
    proc.unref?.();
    proc.stdout?.unref?.();
  }
  return {
    start,
    stop() { try { child?.kill(); } catch { /* gone */ } child = null; },
    get starts() { return starts; },
  };
}

let stoppers = null;
/** Every loop is ended with the host. */
export function endWithHost(loop) {
  if (!stoppers) { stoppers = new Set(); process.once('exit', () => stoppers.forEach(l => l.stop())); }
  stoppers.add(loop);
  return loop;
}
