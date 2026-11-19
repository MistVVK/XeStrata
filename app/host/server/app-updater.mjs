/**
 * The App Host's client of the desktop shell's update bridge.
 *
 * The desktop app updates itself with Tauri's signed updater, which lives in the Rust shell (app/desktop/src-tauri). The
 * UI is served by this host, so it cannot call the shell directly; the shell opens a loopback listener, protected by a
 * random token it hands only to this host (STRATA_SHELL_BRIDGE, STRATA_SHELL_TOKEN), and this module is its client.
 * Outside the desktop app (a browser, a test, a source checkout) there is no bridge and `createAppUpdater` returns null.
 *
 * The shell does the checking against the signed feed (its own minisign key) and the installing; nothing here downloads
 * anything. These calls happen only for an explicit "Update" or "Check for updates", never when the host starts.
 */

/** The shell did not answer at all (its address is closed, reset or silent), as opposed to answering with an error. */
export function connectionLost(error) {
  const text = [error?.name, error?.message, error?.cause?.code, error?.cause?.message].filter(x => typeof x === 'string').join(' ');
  return /fetch failed|ECONNREFUSED|ECONNRESET|EPIPE|socket hang up|terminated|TimeoutError|AbortError|other side closed/i.test(text);
}

export function createAppUpdater({ env = process.env, fetcher = fetch, sleep = ms => new Promise(r => setTimeout(r, ms)) } = {}) {
  const base = env.STRATA_SHELL_BRIDGE;
  const token = env.STRATA_SHELL_TOKEN;
  if (!base || !token || !/^http:\/\/127\.0\.0\.1:\d+$/.test(base)) return null;
  async function call(path, method = 'GET', timeoutMs = 120_000) {
    const response = await fetcher(`${base}${path}`, { method, headers: { authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(timeoutMs) });
    let body = null;
    try { body = await response.json(); } catch { /* no body */ }
    if (!response.ok) throw new Error(body?.error ?? `The Strata app could not check for updates (${response.status}).`);
    return body;
  }
  return {
    /** { available, version, current, notes } from the shell's updater (signature-verified metadata). */
    check: () => call('/app-update/check'),
    /**
     * Download, verify and install the app update. Resolves when the shell reports it finished; on Windows the installer ends
     * the app first, so the host may be stopped before this returns.
     */
    async install({ onProgress = () => {}, signal = null, pollMs = 500, restartGraceMs = 20_000, now = () => Date.now() } = {}) {
      await call('/app-update/install', 'POST');
      let lostAt = null;
      for (;;) {
        if (signal?.aborted) throw new Error('The app update was stopped.');
        let status;
        try {
          status = await call('/app-update/status', 'GET', 15_000);
          lostAt = null;
        } catch (error) {
          /* The installer ends the app when it starts (the shell, then this host): the shell's address stops answering while the update goes through. That is how
             a successful update ends here, not a failure ("update-strata failed: fetch failed" in the log after every successful app update, 0.2.16 N-A2). The host
             waits for its own end; only when it is still here after restartGraceMs is the lost connection the error. An answer from the shell (an error status) is
             never taken for this. */
          if (!connectionLost(error)) throw error;
          lostAt ??= now();
          if (now() - lostAt >= restartGraceMs) throw error;
          await sleep(pollMs);
          continue;
        }
        onProgress(status);
        if (status.state === 'error') throw new Error(status.error ?? 'The app update failed.');
        if (status.state === 'done') return status;
        await sleep(pollMs);
      }
    },
  };
}
