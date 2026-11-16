// A reply the agent writes by itself after its turn ended (a background command finished) reaches the page: the real bridge, its gateway
// client and session follows, against a loopback stand-in for the Harness gateway (tests/helpers/fake-harness.mjs). No model, no network.
import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';

const WAKE_TEXT = 'The background job finished and printed 42. ';

function fakeHarnessDir(root) {
  const dir = join(root, 'harness');
  mkdirSync(join(dir, 'lib'), { recursive: true });
  mkdirSync(join(dir, 'node_modules'), { recursive: true });
  writeFileSync(join(dir, 'package.json'), JSON.stringify({ name: 'fake-harness', version: '0.0.0-test', type: 'module' }));
  copyFileSync(new URL('./helpers/fake-harness.mjs', import.meta.url), join(dir, 'lib', 'bin.js'));
  return dir;
}

async function rig(t) {
  const root = mkdtempSync(join(tmpdir(), 'strata-wake-'));
  const ws = join(root, 'ws'); mkdirSync(ws, { recursive: true });
  const wakes = [];
  const logs = [];
  let onWake = () => {};
  const bridge = createAgentBridge({
    dataDir: join(root, 'data'),
    localInstances: () => [{ id: 'inst', label: 'Fake', baseUrl: 'http://127.0.0.1:9', modelId: 'fake-model', context: 32768 }],
    connections: () => [],
    env: { ...process.env, STRATA_HARNESS_DIR: fakeHarnessDir(root) },
    startTimeoutMs: 20_000,
    log: line => logs.push(line),
    onWake: (chatId, requestId) => { wakes.push({ chatId, requestId }); onWake(chatId, requestId); },
  });
  t.after(async () => { await bridge.close(); try { rmSync(root, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const send = async (chatId, message) => {
    const frames = [];
    await bridge.runTurn({ chatId, message, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) });
    return frames;
  };
  const nextWake = () => new Promise(resolve => { onWake = (chatId, requestId) => { onWake = () => {}; resolve({ chatId, requestId }); }; });
  return { bridge, send, nextWake, wakes, logs, ws };
}

/** What the page shows for a taken-over turn: the catch-up text, then every live piece. */
function shown(frames) {
  let text = '';
  for (const f of frames) {
    if (f.type === 'resync') text = f.content;
    if (f.type === 'delta') text += f.text;
  }
  return text;
}

async function watchWake(r, chatId) {
  const wake = r.nextWake();
  const first = await r.send(chatId, 'SCRIPT:bg');
  assert.match(shown(first), /Started the job in the background/);
  assert.equal(first.at(-1).type, 'done');
  const w = await wake;
  assert.equal(w.chatId, chatId);
  assert.match(w.requestId, /^wake-/);
  return w;
}

test('a reply committed before anyone followed it is shown whole on the page that takes the turn over', async t => {
  const r = await rig(t);
  const wake = r.nextWake();
  const firstTurn = r.send('chat-committed', 'SCRIPT:bg');
  const w = await wake;
  await firstTurn;
  // A page that loads now finds the turn running and marked as one the agent started by itself, and the host log says so.
  assert.deepEqual(r.bridge.runningTurns().map(({ chatId, requestId, wake }) => ({ chatId, requestId, wake })), [{ chatId: w.chatId, requestId: w.requestId, wake: true }]);
  assert.ok(r.logs.some(l => l.includes(w.chatId) && l.includes(w.requestId) && /started a turn by itself/.test(l)), r.logs.join('\n'));
  const frames = [];
  const attached = r.bridge.attach({ chatId: w.chatId, requestId: w.requestId, onFrame: f => frames.push(f) });
  assert.ok(attached, 'the turn the agent started by itself can be taken over');
  await attached.ended;
  assert.equal(shown(frames), WAKE_TEXT);
  assert.equal(frames.at(-1).type, 'done');
});

test('a reply that ended while no page was there is listed once for the page that comes back', async t => {
  const r = await rig(t);
  const w = await watchWake(r, 'chat-committed');
  for (let i = 0; i < 100 && r.bridge.isRunning(w.chatId); i++) await new Promise(res => setTimeout(res, 50));
  assert.deepEqual(r.bridge.wokenTurns(), [{ chatId: w.chatId, requestId: w.requestId }]);
  const frames = [];
  assert.ok(r.bridge.attach({ chatId: w.chatId, requestId: w.requestId, onFrame: f => frames.push(f) }));
  assert.equal(shown(frames), WAKE_TEXT);
  assert.equal(frames.at(-1).type, 'done');
  assert.deepEqual(r.bridge.wokenTurns(), [], 'read once');
});

test('the chat takes the next message after the reply it wrote by itself, and wakes again after that', async t => {
  const r = await rig(t);
  const w = await watchWake(r, 'chat-committed');
  for (let i = 0; i < 100 && r.bridge.isRunning(w.chatId); i++) await new Promise(res => setTimeout(res, 50));
  const frames = [];
  let marked;
  await r.bridge.runTurn({ chatId: w.chatId, message: 'hello', workspacePath: r.ws, mode: 'bypass', onFrame: f => { frames.push(f); if (f.type === 'start') marked = r.bridge.runningTurns().find(x => x.chatId === w.chatId)?.wake; } });
  assert.equal(marked, false, 'a turn the person started is not marked');
  assert.match(shown(frames), /Echo: hello/);
  const again = await watchWake(r, 'chat-committed');
  assert.notEqual(again.requestId, w.requestId);
  assert.equal(r.wakes.length, 2);
});
