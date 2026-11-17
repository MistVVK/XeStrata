/**
 * A quiet NDJSON answer stays open (#4878). A chat turn can say nothing for minutes (the model reads a long prompt, a tool
 * runs, an approval waits), and the desktop's web view gives up on a request that has been silent for 60 seconds ("Load failed")
 * while the turn goes on in the background. A beat frame, the same `{"type":"beat"}` the Time-skip stream uses and the chat
 * consumer ignores, goes out every 15 seconds. It changes nothing else: a client that goes away still cancels (or detaches
 * from) its turn through the response's own close, exactly as before, and the timer ends on finish, on close and on stop().
 */
export const BEAT_MS = 15_000;
const BEAT_LINE = `${JSON.stringify({ type: 'beat' })}\n`;

/** The interval (a tester may shorten it: STRATA_STREAM_BEAT_MS); anything that is not a sensible number is the default. */
export function beatInterval(env = process.env) {
  const n = Number(env.STRATA_STREAM_BEAT_MS);
  return Number.isFinite(n) && n >= 20 && n <= 60_000 ? n : BEAT_MS;
}

/** Start beating on `res`. Returns stop(), which is safe to call more than once. */
export function keepStreamAlive(res, { ms = beatInterval() } = {}) {
  let timer = setInterval(() => {
    if (res.writableEnded || res.destroyed) { stop(); return; }
    try { res.write(BEAT_LINE); } catch { stop(); }
  }, ms);
  timer.unref?.();
  function stop() {
    if (timer === null) return;
    clearInterval(timer);
    timer = null;
    res.off?.('finish', stop);
    res.off?.('close', stop);
    res.off?.('error', stop);
  }
  res.once('finish', stop);
  res.once('close', stop);
  res.once('error', stop);
  return stop;
}
