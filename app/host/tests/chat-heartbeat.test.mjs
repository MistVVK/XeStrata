/**
 * #4878: a quiet desktop chat stream stays open. The chat answer is NDJSON; during a long silence (prompt processing, a tool,
 * an approval wait) a `{"type":"beat"}` frame goes out every 15 seconds so the web view does not drop the request at 60 s.
 *  - the helper, with fake timers: a 60-second tool wait gives the beats, and the timer ends on finish, on close, on error and
 *    on stop(), with no interval left behind;
 *  - the real route, against a real host and a scripted API model that goes silent: beats arrive, a client that goes away
 *    cancels the model request before the next beat, the next message works, and two chats stay independent.
 * The frontend consumer (src/lib/store.ts) has no branch for an unknown frame type, so `beat` changes nothing there; the last
 * test pins that.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { BEAT_MS, beatInterval, keepStreamAlive } from '../server/stream-beat.mjs';

/** A response stand-in: records writes, emits finish/close like http.ServerResponse. */
function fakeResponse() {
  const res = new EventEmitter();
  res.writes = [];
  res.writableEnded = false;
  res.destroyed = false;
  res.write = chunk => { res.writes.push(String(chunk)); return true; };
  res.end = () => { res.writableEnded = true; res.emit('finish'); res.emit('close'); };
  return res;
}
const beats = res => res.writes.filter(w => w === '{"type":"beat"}\n').length;
const listeners = res => ['finish', 'close', 'error'].reduce((n, e) => n + res.listenerCount(e), 0);

test('the interval is 15 s; a tester can shorten it; nonsense is the default', () => {
  assert.equal(BEAT_MS, 15_000);
  assert.equal(beatInterval({}), 15_000);
  assert.equal(beatInterval({ STRATA_STREAM_BEAT_MS: '200' }), 200);
  for (const bad of ['0', '-5', 'abc', '', '5', '999999', 'NaN']) assert.equal(beatInterval({ STRATA_STREAM_BEAT_MS: bad }), 15_000, bad);
});

test('a 60-second silent tool wait carries a beat every 15 s, each a whole NDJSON line', t => {
  t.mock.timers.enable({ apis: ['setInterval'] });
  const res = fakeResponse();
  const stop = keepStreamAlive(res);
  t.mock.timers.tick(14_999);
  assert.equal(beats(res), 0, 'nothing before 15 s');
  t.mock.timers.tick(1);
  assert.equal(beats(res), 1);
  t.mock.timers.tick(45_000);
  assert.equal(beats(res), 4, 'four beats in 60 s');
  for (const w of res.writes) assert.deepEqual(JSON.parse(w), { type: 'beat' });
  assert.ok(res.writes.every(w => w.endsWith('\n') && w.indexOf('\n') === w.length - 1));
  stop();
});

test('the timer is cleared on finish, on close, on error and by stop(): nothing beats afterwards, no listener is left', t => {
  t.mock.timers.enable({ apis: ['setInterval'] });
  for (const how of ['finish', 'close', 'error', 'stop', 'end']) {
    const res = fakeResponse();
    const stop = keepStreamAlive(res);
    t.mock.timers.tick(15_000);
    assert.equal(beats(res), 1, how);
    if (how === 'stop') stop(); else if (how === 'end') res.end(); else res.emit(how, new Error('x'));
    const after = res.writes.length;
    t.mock.timers.tick(120_000);
    assert.equal(res.writes.length, after, `${how}: no beat after the end`);
    assert.equal(listeners(res), 0, `${how}: no listener left on the response`);
    stop();   // calling it again is harmless
  }
});

test('a response that was destroyed or ended without an event is not written to, and its timer goes at the next tick', t => {
  t.mock.timers.enable({ apis: ['setInterval'] });
  const a = fakeResponse(); keepStreamAlive(a); a.destroyed = true;
  const b = fakeResponse(); keepStreamAlive(b); b.writableEnded = true;
  const c = fakeResponse(); keepStreamAlive(c); c.write = () => { throw new Error('socket closed'); };
  t.mock.timers.tick(15_000);
  assert.equal(a.writes.length + b.writes.length, 0);
  t.mock.timers.tick(120_000);
  assert.equal(a.writes.length + b.writes.length, 0);
  assert.equal(listeners(a) + listeners(b) + listeners(c), 0);
});

test('two responses beat independently: ending one leaves the other beating', t => {
  t.mock.timers.enable({ apis: ['setInterval'] });
  const one = fakeResponse(), two = fakeResponse();
  keepStreamAlive(one); keepStreamAlive(two);
  t.mock.timers.tick(30_000);
  assert.equal(beats(one), 2); assert.equal(beats(two), 2);
  one.emit('close');
  t.mock.timers.tick(30_000);
  assert.equal(beats(one), 2, 'the closed chat stops');
  assert.equal(beats(two), 4, 'the other keeps going');
});

/* ---------------------------------------------------------------- the real route --- */

const HOST = resolve(import.meta.dirname, '..', 'server', 'index.mjs');
const freePort = () => new Promise(r => { const s = createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => r(p)); }); });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } };

async function setup(t, beatMs) {
  const worker = await startScriptedWorker({ model: 'cloud-model' });
  const home = mkdtempSync(join(tmpdir(), 'strata-beat-'));
  const ws = join(home, 'workspace'); mkdirSync(ws, { recursive: true });
  const port = await freePort();
  const env = {
    ...process.env, STRATA_HOME: home, STRATA_PORT: String(port), STRATA_WORKSPACE: ws, STRATA_REAL_ENGINE: '0',
    STRATA_COMPONENTS_DIR: join(home, 'components'), STRATA_STREAM_BEAT_MS: String(beatMs),
    STRATA_SETTINGS_FILE: join(home, 'app-settings.json'), LOCALAPPDATA: join(home, 'Local'), APPDATA: join(home, 'Roaming'),
    XDG_DATA_HOME: join(home, 'xdg-data'), XDG_STATE_HOME: join(home, 'xdg-state'), XDG_CONFIG_HOME: join(home, 'xdg-config'),
    STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()),
  };
  delete env.STRATA_HARNESS_DIR;
  const child = spawn(process.execPath, [HOST], { env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  let log = '';
  child.stdout.on('data', d => { log += d; }); child.stderr.on('data', d => { log += d; });
  t.after(async () => {
    if (child.exitCode === null) {
      if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true });
      else child.kill('SIGTERM');
      await new Promise(r => child.once('exit', r));
    }
    await worker.close(); cleanup(home);
  });
  const base = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 100; i++) {
    try { if ((await fetch(`${base}/api/health`)).ok) break; } catch { /* not up yet */ }
    if (child.exitCode !== null) throw new Error(`the host exited: ${log.slice(-500)}`);
    await sleep(100);
  }
  const saved = await fetch(`${base}/api/model-providers`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ kind: 'custom', name: 'Scripted API', baseUrl: worker.url, protocol: 'openai-chat', apiKey: 'sk-host-test-0123456789', models: [{ id: 'cloud-model', name: 'Cloud model', contextMax: 64000 }] }) });
  assert.equal(saved.status, 200);
  const connectionId = (await saved.json()).connections[0].id;
  const remoteModel = { connectionId, modelId: 'cloud-model' };
  /** Open a chat stream; frames are handed to onFrame as they arrive; abort() drops the connection like a client that left. */
  const chat = (sessionId, message, onFrame = () => {}) => {
    const controller = new AbortController();
    const frames = [];
    const done = (async () => {
      const res = await fetch(`${base}/api/chat/stream`, { method: 'POST', headers: { 'content-type': 'application/json' }, signal: controller.signal, body: JSON.stringify({ sessionId, message, remoteModel, history: [], agent: false }) });
      assert.equal(res.status, 200);
      let buffer = '';
      const decoder = new TextDecoder();
      for await (const chunk of res.body) {
        buffer += decoder.decode(chunk, { stream: true });
        let i;
        while ((i = buffer.indexOf('\n')) >= 0) { const line = buffer.slice(0, i).trim(); buffer = buffer.slice(i + 1); if (line) { const f = JSON.parse(line); frames.push(f); onFrame(f); } }
      }
      return frames;
    })();
    return { frames, done, abort: () => controller.abort() };
  };
  return { worker, chat, alive: () => child.exitCode === null, log: () => log };
}
const until = async (fn, ms = 8000) => { const end = Date.now() + ms; while (Date.now() < end) { if (fn()) return true; await sleep(25); } return false; };

test('route: a silent API model gets beats; a client that leaves cancels the model request before the next beat', { timeout: 40_000 }, async t => {
  const s = await setup(t, 300);
  let first = null;
  const quiet = s.chat('chat-beat-a', 'SCRIPT:hang', f => { if (f.type === 'beat' && first === null) first = Date.now(); });
  assert.ok(await until(() => quiet.frames.filter(f => f.type === 'beat').length >= 2), 'two beats while the model says nothing');
  const framesBefore = quiet.frames.length;
  quiet.abort();
  await assert.rejects(quiet.done, { name: 'AbortError' });
  // the next beat is 300 ms away: the model request must already be cancelled well before it
  assert.ok(await until(() => s.worker.aborted >= 1, 1500), 'disconnect cancelled the direct model request');
  assert.ok(framesBefore >= 2);
  const next = s.chat('chat-beat-a', 'SCRIPT:echo still works');
  const frames = await next.done;
  assert.equal(frames.filter(f => f.type === 'delta').map(f => f.text).join(''), 'still works');
  assert.equal(frames.at(-1).type, 'done');
  assert.ok(s.alive(), 'the host is alive');
});

test('route: two chats at once are independent; leaving one does not touch the other', { timeout: 40_000 }, async t => {
  const s = await setup(t, 250);
  const a = s.chat('chat-beat-1', 'SCRIPT:hang');
  const b = s.chat('chat-beat-2', 'SCRIPT:hang');
  const count = c => c.frames.filter(f => f.type === 'beat').length;
  assert.ok(await until(() => count(a) >= 2 && count(b) >= 2), 'both beat');
  a.abort();
  await assert.rejects(a.done, { name: 'AbortError' });
  assert.ok(await until(() => s.worker.aborted === 1, 1500), 'exactly the one request was cancelled');
  const frozen = count(a);
  const bBefore = count(b);
  assert.ok(await until(() => count(b) >= bBefore + 2), 'the other chat keeps getting beats');
  assert.equal(count(a), frozen, 'the chat that left gets nothing more');
  assert.equal(s.worker.aborted, 1, 'the other model request was not cancelled');
  b.abort();
  await assert.rejects(b.done, { name: 'AbortError' });
  assert.ok(await until(() => s.worker.aborted === 2, 1500));
  assert.ok(s.alive());
});

test('route: a normal answer is unchanged (beats only fill silence) and the stream still ends', { timeout: 40_000 }, async t => {
  const s = await setup(t, 20_000);   // longer than the test: no beat should appear in a fast answer
  const frames = await s.chat('chat-beat-fast', 'SCRIPT:echo hello there').done;
  assert.equal(frames.filter(f => f.type === 'delta').map(f => f.text).join(''), 'hello there');
  assert.equal(frames.filter(f => f.type === 'beat').length, 0);
  assert.equal(frames.at(-1).type, 'done');
});

test('the desktop consumer ignores a frame type it does not know (so beat changes nothing there)', () => {
  const store = readFileSync(resolve(import.meta.dirname, '..', 'src', 'lib', 'store.ts'), 'utf8');
  const start = store.indexOf("if (type === 'delta' || type === 'reasoning') { queueText");
  const doneAt = store.indexOf("} else if (type === 'done') {", start);
  assert.ok(start > 0 && doneAt > start, 'found the frame handler');
  // the chain is if / else if only and its last branch is `done`: no final `else` follows it, so an unknown type falls through silently
  assert.match(store.slice(doneAt, doneAt + 2500), /flushSessions\(\);\s*\}\s*\},/);
  assert.ok(!store.slice(start, doneAt).includes("type === 'beat'"), 'no special case is needed');
  const client = readFileSync(resolve(import.meta.dirname, '..', 'src', 'lib', 'client.ts'), 'utf8');
  assert.match(client, /onFrame\(JSON\.parse\(line\)/);
});
