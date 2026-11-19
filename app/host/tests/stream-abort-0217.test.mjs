// 0217 lane APP, the Windows host-test flake (chat-heartbeat "two chats at once ... exactly the one request was cancelled", 0.2.16 INT runs).
// On a loaded PC about 4 runs in 10 saw 0 of 2 model requests cancelled for the whole 30 s after a client left: the host's own listener on the answer's abort signal fired
// (the host log said so), the stream's reader never came back. Node's fetch holds the link from the signal to its request weakly; when the garbage collector runs in between,
// the abort is heard by nobody. The code that reads the answer now cancels its reader itself when the signal aborts, which closes the connection (stream-abort.mjs).
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createModelProviders } from '../server/model-providers.mjs';

const streamAbort = await import('../server/stream-abort.mjs').catch(() => null);   // (so that a host without it fails the tests below, not the loading of the file)
const cancelOnAbort = (...args) => { assert.ok(streamAbort?.cancelOnAbort, 'server/stream-abort.mjs exists'); return streamAbort.cancelOnAbort(...args); };

const within = (promise, ms, what) => Promise.race([promise, new Promise((_, reject) => setTimeout(() => reject(new Error(`${what}: still waiting after ${ms} ms`)), ms))]);

test('cancelOnAbort: a reader that waits for the next piece ends, and the connection behind it is cancelled, when the signal aborts', async () => {
  let cancelled = null;
  const body = new ReadableStream({ pull: () => new Promise(() => {}), cancel: reason => { cancelled = reason ?? 'cancelled'; } });
  const reader = body.getReader();
  const controller = new AbortController();
  const stop = cancelOnAbort(reader, controller.signal);
  const pending = reader.read();
  controller.abort(new DOMException('Cancelled by the user', 'AbortError'));
  assert.deepEqual(await within(pending, 2000, 'the read'), { done: true, value: undefined });
  assert.ok(cancelled, 'the stream behind the reader was told to close');
  stop();
});

test('cancelOnAbort: a signal that is already aborted cancels at once; no signal does nothing; the listener can be taken away', async () => {
  let cancelled = 0;
  const make = () => new ReadableStream({ pull: () => new Promise(() => {}), cancel: () => { cancelled++; } }).getReader();
  const done = new AbortController(); done.abort();
  cancelOnAbort(make(), done.signal);
  await new Promise(r => setTimeout(r, 20));
  assert.equal(cancelled, 1);
  cancelOnAbort(make(), undefined)();
  const live = new AbortController();
  const stop = cancelOnAbort(make(), live.signal);
  stop();
  live.abort();
  await new Promise(r => setTimeout(r, 20));
  assert.equal(cancelled, 1, 'a finished stream is not cancelled by a later abort');
});

test('a model API answer that is being streamed ends when the person leaves, even when fetch itself does not hear the abort', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'strata-stream-abort-'));
  try {
    let cancelled = false;
    const enc = new TextEncoder();
    // a model server that went quiet after one piece, and a fetch that ignores the signal (the case Node's weak link leaves)
    const fetchImpl = async () => new Response(new ReadableStream({
      start(controller) { controller.enqueue(enc.encode('data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n')); },
      pull: () => new Promise(() => {}),
      cancel: () => { cancelled = true; },
    }), { status: 200, headers: { 'content-type': 'text/event-stream' } });
    const service = createModelProviders({ directory, fetchImpl });
    const connection = service.save({ kind: 'custom', name: 'Quiet', baseUrl: 'http://127.0.0.1:9/v1', protocol: 'openai-chat', apiKey: 'k', models: [{ id: 'quiet', reasoning: false }] }).connections[0];
    const controller = new AbortController();
    const frames = [];
    const streaming = service.stream({ selection: { connectionId: connection.id, modelId: 'quiet' }, message: 'hi', history: [], signal: controller.signal, onFrame: frame => { frames.push(frame); if (frame.type === 'delta') controller.abort(new DOMException('Cancelled by the user', 'AbortError')); } });
    await assert.rejects(within(streaming, 3000, 'the stream after the person left'), error => { assert.equal(error.name, 'AbortError'); return true; });
    assert.ok(frames.some(f => f.type === 'delta'), 'the piece that came before was shown');
    assert.equal(cancelled, true, 'the connection to the model was closed');
  } finally { rmSync(directory, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 }); }
});
