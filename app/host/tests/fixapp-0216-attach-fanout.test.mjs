// FIX-APP-0216 A6: the same chat open in two windows. A page that took a running turn over used to REPLACE the page that had it (one subscriber per turn), so the other window got
// nothing but the heartbeat until the turn ended (soak 11-18: the driver's page check attached to turns and starved the stream that started them; the owner's twin: two windows).
// Now every page that follows a turn is a subscriber: all get every live frame, the catch-up goes to the newcomer only, and a page that leaves drops out alone.
// The real bridge, its gateway client and session follows, against the loopback stand-in for the Harness (tests/helpers/fake-harness.mjs, `SCRIPT:slow`). No model, no network.
import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';

const sleep = ms => new Promise(r => setTimeout(r, ms));
const SLOW_TEXT = 'alpha bravo charlie delta echo foxtrot golf hotel ';

function fakeHarnessDir(root) {
  const dir = join(root, 'harness');
  mkdirSync(join(dir, 'lib'), { recursive: true });
  mkdirSync(join(dir, 'node_modules'), { recursive: true });
  writeFileSync(join(dir, 'package.json'), JSON.stringify({ name: 'fake-harness', version: '0.0.0-test', type: 'module' }));
  copyFileSync(new URL('./helpers/fake-harness.mjs', import.meta.url), join(dir, 'lib', 'bin.js'));
  return dir;
}

async function rig(t) {
  const root = mkdtempSync(join(tmpdir(), 'strata-fanout-'));
  const ws = join(root, 'ws'); mkdirSync(ws, { recursive: true });
  const bridge = createAgentBridge({
    dataDir: join(root, 'data'),
    localInstances: () => [{ id: 'inst', label: 'Fake', baseUrl: 'http://127.0.0.1:9', modelId: 'fake-model', context: 32768 }],
    connections: () => [],
    env: { ...process.env, STRATA_HARNESS_DIR: fakeHarnessDir(root), FAKE_HARNESS_SLOW_MS: '250' },
    startTimeoutMs: 20_000,
  });
  t.after(async () => { await bridge.close(); try { rmSync(root, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  /** A page: its frames, and the text it shows (catch-up, then live pieces). */
  const page = () => { const p = { frames: [], onFrame: f => p.frames.push(f) }; p.text = () => { let s = ''; for (const f of p.frames) { if (f.type === 'resync') s = f.content; if (f.type === 'delta') s += f.text; } return s; }; p.types = () => p.frames.map(f => f.type); p.done = () => p.frames.some(f => f.type === 'done'); return p; };
  const start = (chatId, p, extra = {}) => bridge.runTurn({ chatId, message: 'SCRIPT:slow', workspacePath: ws, mode: 'bypass', onFrame: p.onFrame, ...extra });
  const until = async (fn, ms = 8000) => { const t0 = Date.now(); while (!fn() && Date.now() - t0 < ms) await sleep(20); return fn(); };
  return { bridge, page, start, until };
}

test('two pages on one running turn: the second joins mid-turn, both get the same text to the end, and the catch-up went to the newcomer only', async t => {
  const r = await rig(t), A = r.page(), B = r.page();
  const turn = r.start('chat-1', A);
  assert.equal(await r.until(() => A.frames.filter(f => f.type === 'delta').length >= 2), true, 'the first page is receiving the reply');
  const attached = r.bridge.attach({ chatId: 'chat-1', onFrame: B.onFrame });
  assert.ok(attached, 'the running turn can be taken over');
  assert.deepEqual(B.types().slice(0, 2), ['start', 'resync'], 'the newcomer starts with the catch-up');
  assert.ok(B.frames[1].content.length > 0 && SLOW_TEXT.startsWith(B.frames[1].content), `the catch-up holds the text so far: ${JSON.stringify(B.frames[1].content)}`);
  await turn; await attached.ended;
  assert.equal(A.done(), true); assert.equal(B.done(), true);
  assert.equal(A.text(), SLOW_TEXT, 'the page that started the turn kept receiving after the other joined');
  assert.equal(B.text(), SLOW_TEXT, 'the page that joined has the whole reply');
  assert.equal(A.frames.filter(f => f.type === 'resync').length, 0, 'the catch-up was not sent to the page that already had the turn');
  assert.equal(A.frames.filter(f => f.type === 'start').length, 1, 'no second start for the first page');
  const doneA = A.frames.at(-1), doneB = B.frames.at(-1);
  assert.equal(doneA.type, 'done'); assert.deepEqual(doneB, doneA, 'the closing frame is the same for both');
});

test('a third page joining later gets its own catch-up; the two before keep their streams', async t => {
  const r = await rig(t), A = r.page(), B = r.page(), C = r.page();
  const turn = r.start('chat-1', A);
  await r.until(() => A.frames.filter(f => f.type === 'delta').length >= 1);
  const b = r.bridge.attach({ chatId: 'chat-1', onFrame: B.onFrame });
  await r.until(() => A.frames.filter(f => f.type === 'delta').length >= 4);
  const c = r.bridge.attach({ chatId: 'chat-1', onFrame: C.onFrame });
  assert.equal(C.frames.filter(f => f.type === 'resync').length, 1);
  assert.ok(C.frames.find(f => f.type === 'resync').content.length > B.frames.find(f => f.type === 'resync').content.length, 'it was told more than the page that joined earlier');
  await turn; await b.ended; await c.ended;
  for (const p of [A, B, C]) assert.equal(p.text(), SLOW_TEXT);
  assert.equal(B.frames.filter(f => f.type === 'resync').length, 1, 'the second page was not re-caught-up when the third joined');
  assert.equal(A.frames.filter(f => f.type === 'resync').length, 0);
});

test('a page that leaves does not cut the other: the newcomer leaving, and the starting page leaving', async t => {
  const r = await rig(t);
  {
    const A = r.page(), B = r.page(), leave = new AbortController();
    const turn = r.start('chat-1', A);
    await r.until(() => A.frames.filter(f => f.type === 'delta').length >= 1);
    r.bridge.attach({ chatId: 'chat-1', onFrame: B.onFrame, signal: leave.signal });
    await r.until(() => B.frames.filter(f => f.type === 'delta').length >= 1);
    leave.abort();                                                // the window is closed
    const seen = B.frames.length;
    await turn;
    assert.equal(A.done(), true); assert.equal(A.text(), SLOW_TEXT, 'the page that stayed has the whole reply');
    assert.equal(B.frames.length, seen, 'nothing more is sent to the page that left');
    assert.equal(B.done(), false);
  }
  {
    const A = r.page(), B = r.page(), leave = new AbortController();
    const turn = r.start('chat-2', A, { signal: leave.signal });
    await r.until(() => A.frames.filter(f => f.type === 'delta').length >= 1);
    const b = r.bridge.attach({ chatId: 'chat-2', onFrame: B.onFrame });
    leave.abort();                                                // the page that started the turn goes away (the turn is not stopped by that)
    const seen = A.frames.length;
    await b.ended;
    await turn.catch(() => {});
    assert.equal(B.done(), true); assert.equal(B.text(), SLOW_TEXT, 'the page that took over has the whole reply');
    assert.equal(A.frames.length, seen, 'nothing more to the page that left');
  }
});

test('a page whose connection throws is dropped; the others carry on', async t => {
  const r = await rig(t), A = r.page(), good = r.page();
  const turn = r.start('chat-1', A);
  await r.until(() => A.frames.some(f => f.type === 'delta'));   // the turn is live (attach has something to take over)
  let calls = 0;
  r.bridge.attach({ chatId: 'chat-1', onFrame: () => { calls++; throw new Error('socket closed'); } });
  const b = r.bridge.attach({ chatId: 'chat-1', onFrame: good.onFrame });
  await turn; await b.ended;
  assert.equal(good.text(), SLOW_TEXT); assert.equal(A.text(), SLOW_TEXT);
  assert.equal(calls, 1, 'the broken page was tried once (its first catch-up frame) and never joined');
});

test('a turn that ended while the page was away is still replayed once (the earlier attach behaviour)', async t => {
  const r = await rig(t), A = r.page();
  await r.start('chat-1', A, { signal: undefined, requestId: 'req-late' });
  const B = r.page();
  const attached = r.bridge.attach({ chatId: 'chat-1', requestId: 'req-late', onFrame: B.onFrame });
  assert.ok(attached);
  assert.equal(B.text(), SLOW_TEXT);
  assert.equal(B.frames.at(-1).type, 'done');
});
