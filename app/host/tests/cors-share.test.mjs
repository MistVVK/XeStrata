// #3208: the local-network share listener answers browser CORS the way the app's own API does - when an API key is set.
// The reporter's chatbot page (a browser on another machine) could not get past the preflight: OPTIONS carries no key
// (the spec says so), openai() checks the key first, so every preflight got 401 and no answer had a CORS header.
// Without a key the listener sends no CORS headers, so it stays "a device on this network" (SECURITY.md).
import test from 'node:test';
import assert from 'node:assert/strict';
import { createLocalShare } from '../server/local-share.mjs';

/** openai() as index.mjs runs it: the key check first (no key set means every client passes), then path and method. */
const openaiLike = key => (req, res, path, method) => {
  if (key && req.headers.authorization !== 'Bearer test-only-key') {
    res.writeHead(401, { 'content-type': 'application/json' });
    res.end('{"error":"Invalid API key"}');
    return;
  }
  if (path === '/v1/models' && method === 'GET') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end('{"object":"list","data":[]}');
    return;
  }
  res.writeHead(404, { 'content-type': 'application/json' });
  res.end('{"error":"No route"}');
};

const preflight = (root, origin, extra = {}) => fetch(`${root}/v1/models`, {
  method: 'OPTIONS',
  headers: { origin, 'access-control-request-method': 'GET', ...extra },
});

test('with a key, the shared listener answers the preflight without one and allows any origin', async () => {
  const share = createLocalShare({ handle: openaiLike(true), hasKey: () => true, host: '127.0.0.1', port: 0 });
  try {
    const { url } = await share.start();
    const root = url.replace(/\/v1$/, '');
    const res = await preflight(root, 'http://192.168.1.20:9999', { 'access-control-request-headers': 'authorization,content-type' });
    assert.equal(res.status, 204);
    assert.equal(res.headers.get('access-control-allow-origin'), '*');
    assert.match(res.headers.get('access-control-allow-methods'), /OPTIONS/);
    assert.equal(res.headers.get('access-control-allow-headers'), 'authorization,content-type');
    assert.equal(res.headers.get('access-control-max-age'), '600');
    // a preflight that asks for nothing still names the headers the SDKs and Anthropic browser clients send
    const plain = await preflight(root, 'http://example.org');
    assert.match(plain.headers.get('access-control-allow-headers'), /authorization/);
    assert.match(plain.headers.get('access-control-allow-headers'), /anthropic-version/);
  } finally { await share.stop(); }
});

test('with a key, real answers carry Access-Control-Allow-Origin too - the 401 for a missing key as well', async () => {
  const share = createLocalShare({ handle: openaiLike(true), hasKey: () => true, host: '127.0.0.1', port: 0 });
  try {
    const { url } = await share.start();
    const ok = await fetch(url + '/models', { headers: { origin: 'http://example.org', authorization: 'Bearer test-only-key' } });
    assert.equal(ok.status, 200);
    assert.equal(ok.headers.get('access-control-allow-origin'), '*');
    const denied = await fetch(url + '/models', { headers: { origin: 'http://example.org' } });
    assert.equal(denied.status, 401);
    assert.equal(denied.headers.get('access-control-allow-origin'), '*', 'the browser must see the real 401, not a CORS wall');
  } finally { await share.stop(); }
});

test('a preflight grants nothing: the shared listener still answers no host control, only inference', async () => {
  const share = createLocalShare({ handle: openaiLike(true), hasKey: () => true, host: '127.0.0.1', port: 0 });
  try {
    const { url } = await share.start();
    const root = url.replace(/\/v1$/, '');
    assert.equal((await fetch(root + '/api/tasks', { method: 'OPTIONS', headers: { origin: 'http://example.org' } })).status, 204);
    assert.equal((await fetch(root + '/api/tasks', { headers: { authorization: 'Bearer test-only-key' } })).status, 404);
  } finally { await share.stop(); }
});

test('without a key nothing changes: no CORS headers, and the preflight falls through as before', async () => {
  const share = createLocalShare({ handle: openaiLike(false), hasKey: () => false, host: '127.0.0.1', port: 0 });
  try {
    const { url } = await share.start();
    const root = url.replace(/\/v1$/, '');
    const pre = await preflight(root, 'http://example.org', { 'access-control-request-headers': 'authorization,content-type' });
    assert.equal(pre.status, 404, 'OPTIONS /v1/models reaches the route check, as before the fix');
    assert.equal(pre.headers.get('access-control-allow-origin'), null);
    const get = await fetch(url + '/models', { headers: { origin: 'http://example.org' } });
    assert.equal(get.status, 200, 'a device on the network still uses the keyless share');
    assert.equal(get.headers.get('access-control-allow-origin'), null, 'a keyless share never lets a web page of another site read the model');
  } finally { await share.stop(); }
});
