// 0217 lane ABORT: the sweep of the other `fetch(..., { signal })` call sites after the chat-heartbeat finding (stream-abort.mjs).
// A fake model server that ignores the signal is the case Node's weak link leaves: the abort fired and nobody heard it. What must still happen is that the code
// that reads the answer cancels it (closes the connection). Every wait below is for an event; the 10 s in `within` only turns a hang into a failure message.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { createHash } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { nativeGateway } from '../server/native-gateway.mjs';
import { createImageModels, IMAGE_CATALOG } from '../server/image-models.mjs';

const within = (promise, ms, what) => Promise.race([promise, new Promise((_, reject) => setTimeout(() => reject(new Error(`${what}: still waiting after ${ms} ms`)), ms))]);
const listen = server => new Promise(r => server.listen(0, '127.0.0.1', () => r(server.address().port)));
const close = server => new Promise(r => { server.closeAllConnections(); server.close(r); });
const enc = new TextEncoder();

/** A model server's answer that said one piece and went quiet; `cancelled` resolves when the code reading it closes it. */
function quietAnswer(first) {
  let cancel; const cancelled = new Promise(r => { cancel = r; });
  const response = () => new Response(new ReadableStream({
    start(controller) { controller.enqueue(enc.encode(first)); },
    pull: () => new Promise(() => {}),
    cancel: reason => cancel(reason ?? 'cancelled'),
  }), { status: 200, headers: { 'content-type': 'text/event-stream' } });
  return { response, cancelled };
}

test('native gateway: a client that leaves a streamed answer closes the request to the engine, even when fetch does not hear the abort', { timeout: 30000 }, async () => {
  const quiet = quietAnswer('data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n');
  const records = [];
  const serving = {
    open: value => { const record = { id: records.length, state: 'queued', ...value }; records.push(record); return record; },
    mark: (id, value) => Object.assign(records[id], value),
    finish: (id, value) => Object.assign(records[id], { state: value.error ? 'error' : 'done', ...value }),
  };
  const control = { resolveInstance: () => ({ id: 'x', apiModelId: 'x', port: 9, key: 'k', gpus: [0] }) };
  let sent; const requested = new Promise(r => { sent = r; });
  const fetcher = async () => { sent(); return quiet.response(); };   // ignores the signal
  const gateway = http.createServer(async (req, res) => { let text = ''; for await (const chunk of req) text += chunk; await nativeGateway({ req, res, path: req.url, body: JSON.parse(text), control, serving, fetcher }); });
  const port = await listen(gateway);
  try {
    const result = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'x', stream: true, messages: [] }) });
    await requested;
    const reader = result.body.getReader();
    const first = await within(reader.read(), 10000, 'the first piece');
    assert.match(new TextDecoder().decode(first.value), /Hel/);
    await reader.cancel();   // the client leaves
    await within(quiet.cancelled, 10000, 'the request to the engine after the client left');
    for (let n = 0; n < 500 && !['error', 'done'].includes(records[0].state); n++) await new Promise(r => setTimeout(r, 10));   // the log entry closes right after the cancel (a ceiling, not a delay)
    assert.equal(records[0].state, 'error', 'the request log has it as cancelled');
    assert.equal(records[0].error, 'Request cancelled.');
  } finally { await close(gateway); }
});

test('image model download: a cancel closes the download when fetch does not hear the abort (the pipeline listens to the signal itself)', { timeout: 30000 }, async t => {
  const home = mkdtempSync(join(tmpdir(), 'strata-image-abort-'));
  t.after(() => rmSync(home, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 }));
  const data = Buffer.from('GGUFtest-image-weights');
  const catalog = { ...IMAGE_CATALOG, variants: [{ ...IMAGE_CATALOG.variants[0], bytes: data.length * 4, sha256: createHash('sha256').update(data).digest('hex') }] };
  const quiet = quietAnswer('GGUFtest');
  let sent; const requested = new Promise(r => { sent = r; });
  const service = createImageModels({ dataDir: join(home, 'data'), modelsDir: join(home, 'models'), real: true, catalog, fetcher: async () => { sent(); return quiet.response(); } });
  service.select('q4'); service.install('q4');
  await within(requested, 10000, 'the download request');
  await service.cancel('q4');
  await within(quiet.cancelled, 10000, 'the download after the cancel');
  assert.equal(service.library().variants[0].state, 'available');
});
