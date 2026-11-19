/**
 * Leaving a streamed answer must close the connection to the model, whatever fetch does with the signal.
 *
 * `fetch(url, { signal })` is meant to tear the connection down when the signal aborts, and almost always does. Node's fetch keeps the link from the signal to the
 * request through a weak reference, though: when the garbage collector runs between the request and the abort (a busy computer, a long quiet stream), the abort is
 * heard by nobody. The page's connection closed, the host's signal aborted, and the request to the model went on (on a loaded Windows PC the "client that leaves cancels
 * the model request" test saw 0 of 2 cancelled for 30 s, about 4 runs in 10; the host's own listener on the same signal fired every time). The answer stream is held by
 * the code that reads it, so the abort is passed to it directly: cancelling the reader closes the connection.
 */

/** Cancel `reader` when `signal` aborts. Returns a function that stops listening (call it when the stream is done). */
export function cancelOnAbort(reader, signal) {
  if (!signal || !reader) return () => {};
  const stop = () => { try { void Promise.resolve(reader.cancel(signal.reason)).catch(() => {}); } catch { /* already closed */ } };
  if (signal.aborted) { stop(); return () => {}; }
  signal.addEventListener('abort', stop, { once: true });
  return () => signal.removeEventListener('abort', stop);
}
