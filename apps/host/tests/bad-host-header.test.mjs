import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import net from 'node:net';
import { fileURLToPath } from 'node:url';

/**
 * #4397: one request with an empty or malformed Host header (or a request target that is not a path) used to end the
 * whole host: the URL was built before the routing try/catch, the async handler rejected, and Node ends the process on
 * an unhandled rejection. A bad request now gets a bounded 400 and the host keeps serving.
 */
const root = fileURLToPath(new URL('..', import.meta.url));
const delay = ms => new Promise(r => setTimeout(r, ms));
async function freePort() { const s = net.createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

async function startHost() {
  const home = await mkdtemp(join(tmpdir(), 'strata-host-badhost-'));
  const port = await freePort(), sharePort = await freePort();
  const env = {
    ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port),
    STRATA_SHARE_PORT: String(sharePort), STRATA_TUNNEL_PORT: String(await freePort()), STRATA_REAL_ENGINE: '1',
    STRATA_SETTINGS_FILE: join(home, 'app-settings.json'), STRATA_COMPONENTS_DIR: join(home, 'components'),
    LOCALAPPDATA: join(home, 'Local'), APPDATA: join(home, 'Roaming'), XDG_DATA_HOME: join(home, 'xdg-data'),
    XDG_STATE_HOME: join(home, 'xdg-state'), XDG_CONFIG_HOME: join(home, 'xdg-config'),
  };
  const host = spawn(process.execPath, ['server/index.mjs'], { cwd: root, stdio: 'ignore', env });
  const exit = new Promise(r => host.once('exit', (code, signal) => r({ code, signal })));
  let up = false;
  for (let i = 0; i < 150 && !up; i++) { try { up = (await fetch(`http://127.0.0.1:${port}/api/health`)).ok; } catch { await delay(100); } }
  assert.ok(up, 'host started');
  return { host, port, sharePort, exit, done: async () => { host.kill('SIGKILL'); await exit; await rm(home, { recursive: true, force: true }).catch(() => {}); } };
}

/** One raw HTTP request on its own socket: the bytes are sent as written and the first response is returned. */
function raw(port, bytes) {
  return new Promise((resolve, reject) => {
    const socket = net.connect(port, '127.0.0.1');
    let data = '';
    const finish = () => { socket.destroy(); resolve(data); };
    const timer = setTimeout(finish, 4000);
    socket.on('connect', () => socket.write(bytes));
    socket.on('data', chunk => { data += chunk; if (/\r\n\r\n/.test(data) && /content-length: *\d+/i.test(data)) { const body = data.split('\r\n\r\n')[1] ?? ''; const len = Number(/content-length: *(\d+)/i.exec(data)[1]); if (Buffer.byteLength(body) >= len) { clearTimeout(timer); finish(); } } });
    socket.on('close', () => { clearTimeout(timer); resolve(data); });
    socket.on('error', e => { clearTimeout(timer); data ? resolve(data) : reject(e); });
  });
}
const status = response => Number(/^HTTP\/1\.[01] (\d{3})/.exec(response)?.[1] ?? 0);

test('requestUrl: a good Host (name, port, IPv6) or none gives the URL; an empty or malformed Host or target gives null', async () => {
  const { requestUrl } = await import('../server/util.mjs');
  const req = (url, host) => ({ url, headers: host === undefined ? {} : { host } });
  assert.equal(requestUrl(req('/api/health?x=1', 'localhost:8791')).pathname, '/api/health');
  assert.equal(requestUrl(req('/api/health?x=1', 'localhost:8791')).searchParams.get('x'), '1');
  assert.equal(requestUrl(req('/', '[::1]:8791')).pathname, '/');
  assert.equal(requestUrl(req('/', '127.0.0.1')).pathname, '/');
  assert.equal(requestUrl(req('/')).pathname, '/', 'no Host (HTTP/1.0) is read as localhost');
  assert.equal(requestUrl({ headers: {} }).pathname, '/', 'no url at all is the root');
  for (const bad of ['', ' ', 'a b', 'exa mple.com', 'a/b', 'a?b', 'a#b', 'u@h', 'h\\x', '[::1', 'http://x', '%zz', 'a:99999']) {
    assert.equal(requestUrl(req('/', bad)), null, `Host ${JSON.stringify(bad)}`);
  }
  assert.equal(requestUrl(req('//', 'localhost')), null, 'a target of // has no host to read');
  assert.equal(requestUrl(req('http://', 'localhost')), null);
});

test('App Host: an empty, missing, malformed or IPv6 Host never ends the process; a healthy request on another socket still works', async t => {
  const h = await startHost();
  t.after(h.done);
  let ended = null;
  h.exit.then(r => { ended = r; });

  const empty = await raw(h.port, 'GET /api/health HTTP/1.1\r\nHost: \r\nConnection: close\r\n\r\n');
  assert.equal(status(empty), 400, `empty Host: ${empty.slice(0, 80)}`);
  assert.match(empty, /bad_host/);
  assert.equal(ended, null, 'the host survived an empty Host');
  const alive = await fetch(`http://127.0.0.1:${h.port}/api/health`);
  assert.equal(alive.status, 200, 'a healthy request on a separate socket after the empty Host');

  const spaced = await raw(h.port, 'GET /api/health HTTP/1.1\r\nHost: a b\r\nConnection: close\r\n\r\n');
  assert.equal(status(spaced), 400, 'malformed Host');
  assert.ok(spaced.length < 1000, 'the 400 is bounded');
  for (const odd of ['a/b', 'u@h', '[::1', 'x y z', '%zz']) {
    const r = await raw(h.port, `GET /api/health HTTP/1.1\r\nHost: ${odd}\r\nConnection: close\r\n\r\n`);
    assert.equal(status(r), 400, `Host ${odd}`);
  }
  assert.equal(ended, null, 'the host survived the malformed Hosts');

  const none = await raw(h.port, 'GET /api/health HTTP/1.0\r\n\r\n');
  assert.equal(status(none), 200, 'HTTP/1.0 without a Host is served');

  const ipv6 = await raw(h.port, `GET /api/health HTTP/1.1\r\nHost: [::1]:${h.port}\r\nConnection: close\r\n\r\n`);
  assert.equal(status(ipv6), 200, 'a valid IPv6 Host is served');

  const target = await raw(h.port, 'GET // HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n');
  assert.equal(status(target), 400, 'a request target that is not a path');

  assert.equal(ended, null, 'the host is still running');
  assert.equal((await fetch(`http://127.0.0.1:${h.port}/api/health`)).status, 200, 'and still answers');
});

test('The share listener also survives a request target that is not a path', async t => {
  const h = await startHost();
  t.after(h.done);
  // the share listener is off until sharing is started: start it the way the app does
  const started = await fetch(`http://127.0.0.1:${h.port}/api/serving/share`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ enabled: true }) }).catch(() => null);
  if (!started || !started.ok) return t.skip('could not start the share listener in this environment');
  const bad = await raw(h.sharePort, 'GET // HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n');
  assert.equal(status(bad), 400);
  assert.equal((await fetch(`http://127.0.0.1:${h.port}/api/health`)).status, 200);
});

test('malformed JSON on inference routes returns 400 without changing route or key checks', async t => {
  const h = await startHost();
  t.after(h.done);
  const host = `http://127.0.0.1:${h.port}`;
  const json = { 'content-type': 'application/json' };
  const key = 'test-only-key';
  const settings = await fetch(host + '/api/serving', { method: 'PATCH', headers: json, body: JSON.stringify({ apiKey: key }) });
  assert.equal(settings.status, 200);
  const started = await fetch(host + '/api/serving/share', { method: 'POST', headers: json, body: '{"enabled":true}' });
  assert.equal(started.status, 200);
  const share = new URL((await started.json()).sharing.url);
  share.hostname = '127.0.0.1';
  const headers = { ...json, authorization: `Bearer ${key}` };
  const post = (base, path, body, auth = headers) => fetch(base + path, { method: 'POST', headers: auth, body });
  const invalid = { type: 'invalid_request_error', message: 'Invalid JSON in request body' };

  for (const base of [host, share.origin]) {
    const chat = await post(base, '/v1/chat/completions', '{');
    assert.equal(chat.status, 400);
    assert.deepEqual(await chat.json(), { error: { ...invalid, param: null, code: 'invalid_json' } });
    const messages = await post(base, '/v1/messages', '{');
    assert.equal(messages.status, 400);
    assert.deepEqual(await messages.json(), { type: 'error', error: invalid });
    assert.equal((await post(base, '/v1/missing', '{')).status, 404);
    assert.equal((await post(base, '/v1/chat/completions', '{', json)).status, 401);
    const semantic = await post(base, '/v1/chat/completions', '{}');
    assert.equal(semantic.status, 400);
    assert.deepEqual(await semantic.json(), { error: 'No user message found' });
  }
});
