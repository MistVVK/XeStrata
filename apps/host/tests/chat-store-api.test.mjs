/** The chat routes of the real host: files in the profile's data folder, the page named by a header, other sites refused, a 12 MB chat, two hosts one after the other, an event for the other window. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, rm, readdir, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import net from 'node:net';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const delay = ms => new Promise(r => setTimeout(r, ms));
async function freePort() { const s = net.createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

async function startHost(home) {
  const port = await freePort();
  const env = {
    ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port),
    STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()), STRATA_REAL_ENGINE: '1',
    STRATA_SETTINGS_FILE: join(home, 'app-settings.json'), STRATA_COMPONENTS_DIR: join(home, 'components'),
    LOCALAPPDATA: join(home, 'Local'), APPDATA: join(home, 'Roaming'), XDG_DATA_HOME: join(home, 'xdg-data'),
    XDG_STATE_HOME: join(home, 'xdg-state'), XDG_CONFIG_HOME: join(home, 'xdg-config'),
  };
  const host = spawn(process.execPath, ['server/index.mjs'], { cwd: root, stdio: 'ignore', env });
  const exit = new Promise(r => host.once('exit', (code, signal) => r({ code, signal })));
  let up = false;
  for (let i = 0; i < 150 && !up; i++) { try { up = (await fetch(`http://127.0.0.1:${port}/api/health`)).ok; } catch { await delay(100); } }
  assert.ok(up, 'host started');
  const base = `http://127.0.0.1:${port}`;
  return { base, stop: async () => { host.kill('SIGKILL'); await exit; } };
}
const named = (window, extra = {}) => ({ 'x-strata-client': 'ui', 'x-strata-window': window, ...extra });
const chat = (id, n, len) => ({ id, title: `Chat ${id}`, createdAt: '2026-11-04T10:00:00.000Z', updatedAt: '2026-11-04T10:00:01.000Z', messages: Array.from({ length: n }, (_, i) => ({ id: `${id}-${i}`, role: i % 2 ? 'assistant' : 'user', content: String.fromCharCode(97 + (i % 26)).repeat(len) })) });

test('the chat routes: named page only, other sites refused, save / read / list / delete, a 12 MB chat, and two hosts one after the other see the same files', async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-chats-api-'));
  let h = await startHost(home);
  try {
    // The page must name itself; a bare request (what another site's page can send without asking first) is refused.
    assert.equal((await fetch(`${h.base}/api/chats`)).status, 403);
    assert.equal((await fetch(`${h.base}/api/chats`, { headers: { 'x-strata-client': 'ui', origin: 'https://example.com' } })).status, 403, 'a page from another site is refused');
    const empty = await (await fetch(`${h.base}/api/chats`, { headers: named('w1') })).json();
    assert.deepEqual([empty.v, empty.chats], [1, []]);

    const big = chat('s_big', 12, 1_000_000);   // 12 MB
    const body = JSON.stringify(big);
    assert.ok(body.length > 12_000_000);
    const put = await fetch(`${h.base}/api/chats/s_big`, { method: 'PUT', headers: named('w1', { 'content-type': 'application/json' }), body });
    assert.equal(put.status, 200);
    assert.deepEqual(Object.keys(await put.json()).sort(), ['bytes', 'conflict', 'id', 'rev']);
    const got = await (await fetch(`${h.base}/api/chats/s_big`, { headers: named('w1') })).json();
    assert.equal(JSON.stringify(got.session), body, 'byte for byte');
    assert.equal(got.rev, 1);

    // a chat that is not a chat, and an id that climbs out of the folder
    assert.equal((await fetch(`${h.base}/api/chats/s_x`, { method: 'PUT', headers: named('w1'), body: '{nope' })).status, 400);
    assert.equal((await fetch(`${h.base}/api/chats/s_x`, { method: 'PUT', headers: named('w1'), body: JSON.stringify(chat('s_other', 1, 1)) })).status, 400);
    assert.equal((await fetch(`${h.base}/api/chats/..%2Fevil`, { method: 'PUT', headers: named('w1'), body: JSON.stringify(chat('../evil', 1, 1)) })).status, 400);
    assert.equal((await fetch(`${h.base}/api/chats/%E0%A4%A`, { headers: named('w1') })).status, 400, 'a malformed escape is a 400, not a crash');
    assert.equal((await fetch(`${h.base}/api/chats/s_missing`, { headers: named('w1') })).status, 404);

    await fetch(`${h.base}/api/chats/s_small`, { method: 'PUT', headers: named('w2'), body: JSON.stringify(chat('s_small', 2, 10)) });
    await h.stop();
    h = await startHost(home);   // the host restarted
    const list = await (await fetch(`${h.base}/api/chats`, { headers: named('w1') })).json();
    assert.deepEqual(list.chats.map(c => c.id).sort(), ['s_big', 's_small']);
    assert.equal(JSON.stringify((await (await fetch(`${h.base}/api/chats/s_big`, { headers: named('w1') })).json()).session), body);

    const dataFiles = (await readdir(join(home, 'data', 'chats'))).sort();
    assert.deepEqual(dataFiles, ['index.json', 's_big.json', 's_small.json']);
    if (process.platform !== 'win32') assert.equal((await stat(join(home, 'data', 'chats', 's_big.json'))).mode & 0o777, 0o600);

    assert.deepEqual(await (await fetch(`${h.base}/api/chats/s_small`, { method: 'DELETE', headers: named('w2') })).json(), { id: 's_small', removed: true });
    assert.equal((await (await fetch(`${h.base}/api/chats`, { headers: named('w1') })).json()).chats.length, 1);

    // import: the browser's chats, the host keeps what it has
    const imp = await (await fetch(`${h.base}/api/chats/import`, { method: 'POST', headers: named('w1'), body: JSON.stringify({ sessions: [chat('s_big', 1, 1), chat('s_old', 3, 10)] }) })).json();
    assert.deepEqual([imp.imported, imp.skipped], [['s_old'], ['s_big']]);
  } finally { await h.stop(); await rm(home, { recursive: true, force: true }).catch(() => {}); }
});

test('a save in one window is announced to the others on the event stream, with the name of the window that made it', async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-chats-events-'));
  const h = await startHost(home);
  try {
    const controller = new AbortController();
    const res = await fetch(`${h.base}/api/events`, { signal: controller.signal });
    const reader = res.body.getReader();
    let seen = '';
    const read = (async () => { const dec = new TextDecoder(); try { for (;;) { const { value, done } = await reader.read(); if (done) return; seen += dec.decode(value); } } catch { /* aborted */ } })();
    await delay(200);
    await fetch(`${h.base}/api/chats/s_ev`, { method: 'PUT', headers: named('window-A'), body: JSON.stringify(chat('s_ev', 1, 5)) });
    for (let i = 0; i < 50 && !seen.includes('event: chats'); i++) await delay(50);
    controller.abort(); await read;
    const frame = /event: chats\ndata: (.*)\n/.exec(seen);
    assert.ok(frame, 'a chats event arrived');
    assert.deepEqual(JSON.parse(frame[1]), { id: 's_ev', rev: 1, from: 'window-A' });
  } finally { await h.stop(); await rm(home, { recursive: true, force: true }).catch(() => {}); }
});

test('a page that comes back asks which chats have a turn running; with none, the answer is an empty list and "nothing to take over"', async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-chats-active-'));
  const h = await startHost(home);
  try {
    assert.deepEqual(await (await fetch(`${h.base}/api/chat/active`)).json(), { turns: [], woken: [] });
    const none = await fetch(`${h.base}/api/chat/attach`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ sessionId: 's_nothing', requestId: 'req-1' }) });
    assert.equal(none.status, 404);
    assert.equal((await none.json()).code, 'no-turn');
    assert.equal((await fetch(`${h.base}/api/chat/attach`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' })).status, 400);
  } finally { await h.stop(); await rm(home, { recursive: true, force: true }).catch(() => {}); }
});
