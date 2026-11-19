// 0217 lane FIX-APP, #5677: the Responses API answered 404 on the shared (LAN) port while the loopback port answered. The shared listener has an allow-list of
// paths; POST /v1/responses is on it when the host asks for it (allowResponses), and the host does. This keeps both facts.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createLocalShare } from '../server/local-share.mjs';

async function ask(allowResponses, method, path) {
  const seen = [];
  const share = createLocalShare({ handle: async (req, res, p, m) => { seen.push(`${m} ${p}`); res.writeHead(200, { 'content-type': 'application/json' }); res.end('{}'); },
    hasKey: () => true, host: '127.0.0.1', port: 0, allowResponses });
  const state = await share.start();
  try {
    const port = Number(new URL(state.url).port);
    const res = await fetch(`http://127.0.0.1:${port}${path}`, { method, ...(method === 'POST' ? { body: '{}', headers: { 'content-type': 'application/json' } } : {}) });
    return { status: res.status, seen };
  } finally { await share.stop(); }
}

test('the shared port passes POST /v1/responses to the host when responses are allowed', async () => {
  const r = await ask(true, 'POST', '/v1/responses');
  assert.equal(r.status, 200);
  assert.deepEqual(r.seen, ['POST /v1/responses']);
});

test('the shared port still refuses the Responses API when it is not allowed, and every other path it never served', async () => {
  assert.equal((await ask(false, 'POST', '/v1/responses')).status, 404);
  assert.equal((await ask(true, 'GET', '/v1/responses')).status, 404);
  assert.equal((await ask(true, 'POST', '/api/native/models')).status, 404);
});

test('the host starts its shared listener with the Responses API allowed', () => {
  const source = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.match(source, /createLocalShare\(\{[^}]*allowResponses: true/);
});
