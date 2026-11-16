/**
 * A session the Harness starts a turn on by itself, after our own turn has ended: a background command finished, or a helper reported late.
 * Such a turn has no request of ours behind it: the Harness's job plugin hands the agent a completion notice (`source.kind` `tool-jobs`, no
 * `source.rpcId`) and the idle agent opens a turn for it, so nothing was following the session and the reply never reached the page. This watches an idle session and says once when a new turn begins.
 */

/** The events that open a turn on the session. */
export const WAKE_EVENTS = new Set(['turn/start', 'user/message', 'step/start']);

/**
 * @param {object} options
 * @param {{ openStream: Function }} options.gw      the Harness gateway
 * @param {string} options.sessionId
 * @param {(seq:number) => void} options.onWake      called once, with the sequence number of the event that opened the new turn
 * @param {number} [options.maxMs]                   the watch ends by itself after this long (a chat nobody comes back to)
 * @returns {{ close: () => void }}
 */
export function watchForWake({ gw, sessionId, onWake, maxMs = 2 * 60 * 60_000 }) {
  let cursor = -1;
  let done = false;
  let ready = false;
  let stream = null;
  let timer = null;
  const close = () => {
    if (done) return;
    done = true;
    clearTimeout(timer);
    try { stream?.close(); } catch { /* already closed */ }
  };
  const see = event => {
    if (done || !ready || !event || event.seq <= cursor) return;
    cursor = event.seq;
    if (WAKE_EVENTS.has(event.type)) { close(); onWake(event.seq); }
  };
  stream = gw.openStream('session/follow', { request: { address: { kind: 'session', sessionId }, assistantStream: true } }, {
    onItem: item => {
      if (done) return;
      // A re-opened stream (the socket came back) starts with a fresh snapshot: a turn that began while it was away is in its records.
      if (item.type === 'snapshot') {
        if (!ready) { ready = true; cursor = item.cursor ?? -1; return; }
        for (const r of item.records ?? []) if (r?.type === 'event') see(r.event);
        return;
      }
      if (item.type === 'event') see(item.event);
    },
    onError: () => close(),
  });
  timer = setTimeout(close, maxMs);
  timer.unref?.();
  return { close };
}
