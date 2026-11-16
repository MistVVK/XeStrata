// The /v1 foreign-Origin rule (owner 11-16), the same contract as strata serve's /v1: while no API key is set, a /v1 request from a web
// page of another site is refused with 403 (any site can make a visitor's browser POST to this computer). A request without an Origin
// (curl, SDKs, editors, server-side apps), this app's own page and a trusted origin pass; "null", an empty Origin and an unreadable one
// are another site. With a key set the key is the check: the right key passes from any page, a wrong key is 401. Run on a real host.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import net from 'node:net';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const delay = ms => new Promise(r => setTimeout(r, ms));
async function freePort() { const s = net.createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

async function startHost() {
  const home = await mkdtemp(join(tmpdir(), 'strata-host-v1origin-'));
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
  return { port, sharePort, done: async () => { host.kill('SIGKILL'); await exit; await rm(home, { recursive: true, force: true }).catch(() => {}); } };
}

/** One HTTP/1.1 request on its own socket, headers exactly as given (an empty Origin included). Resolves { status, body }. */
function request(port, method, path, headers = {}, body = '') {
  return new Promise((resolve, reject) => {
    const socket = net.connect(port, '127.0.0.1');
    let data = '';
    const lines = [`${method} ${path} HTTP/1.1`, `Host: 127.0.0.1:${port}`, 'Connection: close',
      ...Object.entries(headers).map(([k, v]) => `${k}: ${v}`), ...(body ? ['Content-Type: application/json', `Content-Length: ${Buffer.byteLength(body)}`] : [])];
    socket.on('connect', () => socket.write(`${lines.join('\r\n')}\r\n\r\n${body}`));
    socket.on('data', c => { data += c; });
    socket.on('end', () => resolve({ status: Number(/^HTTP\/1\.[01] (\d{3})/.exec(data)?.[1] ?? 0), body: data.split('\r\n\r\n').slice(1).join('\r\n\r\n') }));
    socket.on('error', reject);
    setTimeout(() => { socket.destroy(); resolve({ status: Number(/^HTTP\/1\.[01] (\d{3})/.exec(data)?.[1] ?? 0), body: data }); }, 8000);
  });
}

test('no API key: another site\'s page is refused (403); no Origin, the own page and a trusted origin pass; a key makes the key the check', { timeout: 60000 }, async () => {
  const h = await startHost();
  try {
    const models = headers => request(h.port, 'GET', '/v1/models', headers);
    const chat = headers => request(h.port, 'POST', '/v1/chat/completions', headers, JSON.stringify({ model: 'x', messages: [{ role: 'user', content: 'hi' }] }));
    // no key + foreign Origin -> 403, on GET and POST; "null", empty and unreadable origins too
    for (const origin of ['https://evil.example', 'http://192.168.1.50:8080', 'null', '', 'not a url']) {
      const r = await models({ Origin: origin });
      assert.equal(r.status, 403, `GET Origin ${JSON.stringify(origin)}`);
      assert.match(r.body, /Other websites can't use this API while no API key is set. Set an API key in Serving to allow it./);
    }
    assert.equal((await chat({ Origin: 'https://evil.example' })).status, 403, 'POST from another site');
    // no key + no Origin -> passes (curl, SDKs)
    assert.equal((await models({})).status, 200);
    assert.notEqual((await chat({})).status, 403, 'a client that is no browser is not refused for its origin');
    // no key + its own page -> passes
    assert.equal((await models({ Origin: `http://127.0.0.1:${h.port}` })).status, 200);
    assert.equal((await models({ Origin: 'http://localhost:5173' })).status, 200, 'a page on this computer');
    assert.equal((await models({ Origin: 'chrome-extension://abcdefghijklmnop' })).status, 200, 'a browser extension (no web site can send it)');
    // no key + a trusted origin -> passes
    const patch = await request(h.port, 'PATCH', '/api/serving', {}, JSON.stringify({ trustedOrigins: ['https://app.example/', 'javascript:alert(1)', 'garbage'] }));
    assert.equal(patch.status, 200);
    assert.deepEqual(JSON.parse(patch.body).trustedOrigins, ['https://app.example'], 'only exact http(s) origins are kept');
    assert.equal((await models({ Origin: 'https://app.example' })).status, 200);
    assert.equal((await models({ Origin: 'https://app.example.evil.example' })).status, 403);
    // key set: foreign Origin + right key passes; a wrong key is 401
    const key = 'sk-strata-origin-test';
    assert.equal((await request(h.port, 'PATCH', '/api/serving', {}, JSON.stringify({ apiKey: key }))).status, 200);
    assert.equal((await models({ Origin: 'https://evil.example', Authorization: `Bearer ${key}` })).status, 200);
    assert.equal((await models({ Origin: 'https://evil.example', Authorization: 'Bearer sk-strata-wrong' })).status, 401);
    assert.equal((await models({ Authorization: 'Bearer sk-strata-wrong' })).status, 401);
  } finally { await h.done(); }
});

test('the LAN sharing listener follows the same rule', { timeout: 60000 }, async () => {
  const h = await startHost();
  try {
    const on = await request(h.port, 'POST', '/api/serving/share', {}, JSON.stringify({ enabled: true }));
    assert.equal(on.status, 200, on.body);
    let r = { status: 0 };
    for (let i = 0; i < 20 && !r.status; i++) { r = await request(h.sharePort, 'GET', '/v1/models', { Origin: 'https://evil.example' }).catch(() => ({ status: 0 })); if (!r.status) await delay(100); }
    assert.equal(r.status, 403);
    assert.equal((await request(h.sharePort, 'GET', '/v1/models', {})).status, 200);
  } finally { await h.done(); }
});

test('the request\'s own Host counts as the own page on the app\'s listener only; LAN sharing and the tunnel take loopback pages only', async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-v1origin-unit-'));
  process.env.STRATA_HOME = home;
  try {
    const serving = await import('../server/serving.mjs');
    serving.patch({ apiKey: null });
    const req = (origin, host) => ({ headers: { origin, host } });
    assert.equal(serving.v1OriginRefused(req('http://192.168.1.5:8791', '192.168.1.5:8791'), 'app'), false, 'the app\'s page on a network address');
    for (const via of ['share', 'tunnel']) {
      assert.equal(serving.v1OriginRefused(req('http://192.168.1.5:8792', '192.168.1.5:8792'), via), true, via);
      assert.equal(serving.v1OriginRefused(req('http://localhost:3000', '192.168.1.5:8792'), via), false, `${via}: a page on this computer`);
    }
    assert.equal(serving.v1OriginRefused({ headers: { host: '192.168.1.5:8792' } }, 'share'), false, 'no Origin');
  } finally { await rm(home, { recursive: true, force: true }).catch(() => {}); }
});
