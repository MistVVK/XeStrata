/**
 * #4535: the shared listener and the tunnel origin compare the API key in constant time (as the engine does since #213).
 * Both sides are hashed, so a wrong length is as cheap and as safe as a wrong value; a missing or non-string header
 * never matches; no key set still means open access (0.2.2).
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { sameSecret } from '../server/util.mjs';
import { createLocalShare } from '../server/local-share.mjs';

const home = mkdtempSync(join(tmpdir(), 'strata-serving-compare-'));
process.env.STRATA_HOME = home;
const serving = await import('../server/serving.mjs');
test.after(() => { try { rmSync(home, { recursive: true, force: true }); } catch { /* temp folder */ } });

const KEY = 'sk-strata-0123456789abcdef0123456789abcdef';
serving.init({ send() {} }, { port: 8080, host: '127.0.0.1' });
serving.patch({ apiKey: KEY });
const asks = headers => serving.authorized({ headers });

test('sameSecret: equal strings match; different, shorter, longer, empty and non-string never do', () => {
  assert.equal(sameSecret(KEY, KEY), true);
  assert.equal(sameSecret(KEY.slice(0, -1), KEY), false, 'a prefix of the key');
  assert.equal(sameSecret(KEY + 'x', KEY), false, 'the key plus one character');
  assert.equal(sameSecret('x'.repeat(KEY.length), KEY), false, 'same length, other content');
  assert.equal(sameSecret('x', KEY), false);
  assert.equal(sameSecret('x'.repeat(100000), KEY), false, 'a very long value is safe');
  assert.equal(sameSecret('', KEY), false);
  assert.equal(sameSecret('', ''), false, 'an empty secret is "not set", never equal to an empty header');
  for (const odd of [undefined, null, 5, {}, [KEY], true, Buffer.from(KEY)]) assert.equal(sameSecret(odd, KEY), false, String(typeof odd));
  assert.equal(sameSecret(KEY, undefined), false);
  assert.equal(sameSecret('kéy-中文', 'kéy-中文'), true, 'non-ASCII keys compare by their characters');
  assert.equal(sameSecret('kéy', 'key'), false);
});

test('authorized: Bearer and x-api-key pass with the key; wrong lengths, invalid tokens and malformed headers fail', () => {
  assert.equal(asks({ authorization: `Bearer ${KEY}` }), true);
  assert.equal(asks({ 'x-api-key': KEY }), true);
  assert.equal(asks({ authorization: 'Bearer wrong', 'x-api-key': KEY }), true, 'a wrong Bearer with the right x-api-key passes, as before');
  assert.equal(asks({ authorization: `Bearer ${KEY}`, 'x-api-key': 'wrong' }), true);
  for (const headers of [
    {},
    { authorization: '' },
    { authorization: 'Bearer ' },
    { authorization: 'Bearer' },
    { authorization: 'Bearer wrong' },
    { authorization: `Bearer ${KEY.slice(0, -1)}` },
    { authorization: `Bearer ${KEY}x` },
    { authorization: `Bearer ${'a'.repeat(5000)}` },
    { authorization: KEY },
    { authorization: `bearer ${KEY}` },
    { authorization: `Basic ${KEY}` },
    { 'x-api-key': '' },
    { 'x-api-key': KEY.slice(1) },
    { 'x-api-key': `${KEY} ` },
    { 'x-api-key': [KEY] },
    { 'x-api-key': 12345 },
    { authorization: ['Bearer ' + KEY] },
  ]) assert.equal(asks(headers), false, JSON.stringify(headers).slice(0, 80));
  assert.equal(serving.authorized({}), false, 'a request without headers at all');
});

test('authorized: with no key set the listener stays open (the 0.2.2 decision)', () => {
  serving.patch({ apiKey: null });
  assert.equal(asks({}), true);
  assert.equal(asks({ authorization: 'Bearer anything' }), true);
  serving.patch({ apiKey: KEY });
  assert.equal(asks({}), false);
});

test('a real share request: the right key reaches the model API, a wrong or missing one is refused with 401', async t => {
  serving.patch({ apiKey: KEY });
  const seen = [];
  const handle = async (req, res, path) => {
    if (!serving.authorized(req)) { res.writeHead(401, { 'content-type': 'application/json' }); res.end('{"error":"Invalid API key"}'); return; }
    seen.push(path);
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end('{"data":[]}');
  };
  // find a free port the way the host does
  const probe = http.createServer(); await new Promise(r => probe.listen(0, '127.0.0.1', r)); const port = probe.address().port; await new Promise(r => probe.close(r));
  const real = createLocalShare({ handle, hasKey: () => true, host: '127.0.0.1', port });
  await real.start();
  t.after(() => real.stop());
  const call = headers => fetch(`http://127.0.0.1:${port}/v1/models`, { headers });
  assert.equal((await call({ authorization: `Bearer ${KEY}` })).status, 200);
  assert.equal((await call({ 'x-api-key': KEY })).status, 200);
  assert.equal((await call({ authorization: `Bearer ${KEY}0` })).status, 401);
  assert.equal((await call({ authorization: 'Bearer short' })).status, 401);
  assert.equal((await call({})).status, 401);
  assert.deepEqual(seen, ['/v1/models', '/v1/models']);
});
