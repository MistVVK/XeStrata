// #6070 review: two gaps in following the replies the agent starts by itself (a background command finished after its turn ended).
// 1. The idle watch took its first snapshot as the past, so a turn that began between the end of ours and that snapshot (a quick command)
//    was never followed. The watch now starts after the last event our turn saw.
// 2. The bridge kept one finished turn per chat (`last`), so of two such replies that ended while no page was open only the second reached
//    the page. Every unseen one is kept (up to WOKEN_KEPT), listed oldest first and read once each; the page takes them in order.
import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { watchForWake } from '../server/agent/wake-watch.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';

function fakeGateway() {
  const g = { handlers: null, closed: 0 };
  g.openStream = (_m, _p, handlers) => { g.handlers = handlers; return { close: () => { g.closed++; } }; };
  g.snapshot = (cursor, events = []) => g.handlers.onItem({ type: 'snapshot', cursor, records: events.map(([seq, type]) => ({ type: 'event', event: { seq, type, data: {} } })) });
  g.event = (seq, type) => g.handlers.onItem({ type: 'event', event: { seq, type, data: {} } });
  return g;
}

test('watch: a turn that began after our last event but before the first snapshot wakes the chat, once', () => {
  const gw = fakeGateway(), woke = [];
  watchForWake({ gw, sessionId: 's', after: 40, onWake: s => woke.push(s) });
  gw.snapshot(46, [[39, 'turn/end'], [40, 'turn/end'], [41, 'turn/start'], [42, 'user/message'], [43, 'step/start']]);
  assert.deepEqual(woke, [41]);
  assert.equal(gw.closed, 1);
});

test('watch: with `after`, our own turn\'s events in the first snapshot never wake it; a later turn does', () => {
  const gw = fakeGateway(), woke = [];
  watchForWake({ gw, sessionId: 's', after: 40, onWake: s => woke.push(s) });
  gw.snapshot(40, [[38, 'step/start'], [39, 'step/end'], [40, 'turn/end']]);
  assert.deepEqual(woke, []);
  gw.event(40, 'turn/start');
  assert.deepEqual(woke, [], 'an event at or before the cursor');
  gw.event(41, 'turn/start');
  assert.deepEqual(woke, [41]);
});

test('watch: without `after` the first snapshot is the past, as before', () => {
  const gw = fakeGateway(), woke = [];
  watchForWake({ gw, sessionId: 's', onWake: s => woke.push(s) });
  gw.snapshot(10, [[9, 'turn/start']]);
  assert.deepEqual(woke, []);
  gw.event(11, 'turn/start');
  assert.deepEqual(woke, [11]);
});

test('the bridge hands the watch the last event its turn saw, and keeps every unseen woken reply', () => {
  const bridge = readFileSync(new URL('../server/agent/agent-bridge.mjs', import.meta.url), 'utf8');
  assert.match(bridge, /armWake\(c, runtime, sessionId, turn\.lastSeq\)/);
  assert.match(bridge, /watchForWake\(\{ gw: runtime\.gw, sessionId, after,/);
  const store = readFileSync(new URL('../src/lib/store.ts', import.meta.url), 'utf8');
  assert.match(store, /requestIds\.reduce<Promise<unknown>>\(\(before, requestId\) => before\.then\(\(\) => sendMessage\('', \{ reattach: chatId, requestId, wake: true \}\)\), Promise\.resolve\(\)\)/,
    'the page takes several woken replies of one chat one after the other');
});

function fakeHarnessDir(root) {
  const dir = join(root, 'harness');
  mkdirSync(join(dir, 'lib'), { recursive: true });
  mkdirSync(join(dir, 'node_modules'), { recursive: true });
  writeFileSync(join(dir, 'package.json'), JSON.stringify({ name: 'fake-harness', version: '0.0.0-test', type: 'module' }));
  copyFileSync(new URL('./helpers/fake-harness.mjs', import.meta.url), join(dir, 'lib', 'bin.js'));
  return dir;
}
async function rig(t, wakeMs) {
  const root = mkdtempSync(join(tmpdir(), 'strata-wake-gap-'));
  const ws = join(root, 'ws'); mkdirSync(ws, { recursive: true });
  const wakes = [];
  const bridge = createAgentBridge({
    dataDir: join(root, 'data'),
    localInstances: () => [{ id: 'inst', label: 'Fake', baseUrl: 'http://127.0.0.1:9', modelId: 'fake-model', context: 32768 }],
    connections: () => [],
    env: { ...process.env, STRATA_HARNESS_DIR: fakeHarnessDir(root), FAKE_HARNESS_WAKE_MS: String(wakeMs) },
    startTimeoutMs: 20_000,
    onWake: (chatId, requestId) => wakes.push({ chatId, requestId }),
  });
  t.after(async () => { await bridge.close(); try { rmSync(root, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  return { bridge, ws, wakes };
}
const until = async (check, ms) => { const end = Date.now() + ms; while (!check()) { if (Date.now() > end) return false; await new Promise(r => setTimeout(r, 25)); } return true; };
function shown(frames) { let text = ''; for (const f of frames) { if (f.type === 'resync') text = f.content; if (f.type === 'delta') text += f.text; } return text; }

test('bridge: a background command that finishes at once after the turn still reaches the chat', { timeout: 60_000 }, async t => {
  const r = await rig(t, 0);
  await r.bridge.runTurn({ chatId: 'quick', message: 'SCRIPT:bg', workspacePath: r.ws, mode: 'bypass', onFrame: () => {} });
  assert.ok(await until(() => r.wakes.length === 1, 10_000), 'the turn the agent started at once was followed');
  assert.ok(await until(() => !r.bridge.isRunning('quick'), 10_000));
  const frames = [];
  assert.ok(r.bridge.attach({ chatId: 'quick', requestId: r.wakes[0].requestId, onFrame: f => frames.push(f) }));
  assert.equal(shown(frames), 'The background job finished and printed 42. ');
});

test('bridge: two replies the agent started by itself while no page was open are both kept, oldest first, each read once', { timeout: 60_000 }, async t => {
  const r = await rig(t, 300);
  await r.bridge.runTurn({ chatId: 'two', message: 'SCRIPT:bg2', workspacePath: r.ws, mode: 'bypass', onFrame: () => {} });
  assert.ok(await until(() => r.wakes.length === 2 && !r.bridge.isRunning('two'), 20_000), `wakes: ${JSON.stringify(r.wakes)}`);
  assert.deepEqual(r.bridge.wokenTurns(), r.wakes.map(w => ({ chatId: 'two', requestId: w.requestId })));
  const texts = r.wakes.map(w => { const frames = []; assert.ok(r.bridge.attach({ chatId: 'two', requestId: w.requestId, onFrame: f => frames.push(f) })); return shown(frames); });
  assert.deepEqual(texts, ['The background job finished and printed 42. ', 'The second job finished and printed 43. ']);
  assert.deepEqual(r.bridge.wokenTurns(), [], 'read once');
});
