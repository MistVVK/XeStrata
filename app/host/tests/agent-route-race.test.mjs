// SOAK-APP 11-17 12:39: after a model reload three chats started a turn at once on a route the running Harness did not know; two of them got "Starting..." and
// "Reading your message..." and then no frame for 20 minutes while the Harness did the work. These tests drive the real bridge with a fake gateway (no Harness) and pin
// the safety net: (a) only items that belong to a turn show that it is alive, (b) a turn that never becomes live ends with a plain message, (c) a restart of the Harness
// waits for a chat that already holds the running one, (d) a stream opened on a closed gateway tells its owner.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { GatewayClient } from '../server/agent/gateway-client.mjs';

const sleep = ms => new Promise(r => setTimeout(r, ms));
const instance = id => ({ id, label: 'S', baseUrl: 'http://127.0.0.1:9', modelId: `m-${id}`, context: 32768, slots: 2 });
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };

/** A bridge over fake runtimes: every start of the "Harness" is recorded, every stop too; a gateway that answers what the start of a turn asks. */
function rig(t, { instances, hold = null, ...options } = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'strata-race-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const current = { list: instances ?? [instance('i1')] };
  const starts = [], stops = [], sessions = new Map(), follows = [];
  let made = 0;
  const startRuntimeImpl = async profile => {
    const number = ++made;
    starts.push({ number, routes: profile.routes.map(r => r.modelId) });
    const permissions = new Map();
    const gw = {
      closed: false,
      close() { gw.closed = true; },
      async rpc(method, args = {}) {
        if (method === 'session/create') { const id = `sess-${number}-${sessions.size + 1}`; sessions.set(id, number); return { sessionId: id }; }
        if (method === 'session/selectModel') { if (hold) await hold(args); return {}; }
        if (method === 'session/projections') return { values: { permissions: { currentValue: permissions.get(args.request?.sessionId) ?? null } } };
        if (method === 'commands/execute') {
          const m = /^\/permission (\S+)/.exec(args.line ?? '');
          if (m) { permissions.set(args.agentId, m[1]); return { result: { kind: 'success' } }; }
          return { result: { kind: 'error', text: 'unknown command' } };
        }
        return {};
      },
      openStream(endpoint, args, handlers) {
        const entry = { endpoint, args, handlers, closed: false, gw: number };
        if (endpoint === 'session/follow') { follows.push(entry); queueMicrotask(() => handlers.onItem?.({ type: 'snapshot', cursor: 0, records: [] })); }
        return { close: () => { entry.closed = true; } };
      },
    };
    return { hp: { running: true, pid: 1000 + number, stop: async () => { stops.push(number); } }, gw, profile, clientId: 'c', events: { close() {} } };
  };
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => current.list, connections: () => [], env: process.env, startRuntimeImpl, ...options });
  t.after(async () => { await bridge.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const turn = (chatId, extra = {}) => {
    const frames = [];
    const done = bridge.runTurn({ chatId, message: 'hello', workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f), requestId: `req-${chatId}`, ...extra }).catch(e => ({ thrown: e }));
    return { frames, done, ended: () => frames.some(f => f.type === 'done' || f.type === 'error'), error: () => frames.find(f => f.type === 'error') };
  };
  /** An event of the session's own stream, as the Harness sends it (follow items go to the turn that follows the session). */
  const push = (sessionIndex, item) => follows[sessionIndex]?.handlers.onItem?.(item);
  return { bridge, current, starts, stops, follows, turn, push, ws };
}
const untilEnded = async (turn, ms) => { const t0 = Date.now(); while (!turn.ended() && Date.now() - t0 < ms) await sleep(50); return turn.ended(); };

test('(b) a turn whose own message never shows up in the session events ends with a plain message, also while the engine works for other chats', async t => {
  // the engine's progress changes all the time (it serves other chats), so neither the stall limit nor the engine grace may be what ends it
  let ticks = 0;
  const r = rig(t, { liveMs: 700, stallMs: 4000, stallToolMs: 4000, engineGraceMs: 600_000, engineProgress: () => ++ticks });
  const x = r.turn('chat-x');
  await sleep(1000);
  assert.equal(r.follows.length, 1, 'the turn follows its session');
  // events of the session that are not this turn's (an older turn, a replay) arrive while it waits
  for (let i = 1; i <= 5; i++) r.push(0, { type: 'event', event: { seq: i, type: 'step/start', data: {} } });
  assert.equal(await untilEnded(x, 4000), true, 'the turn ended instead of waiting for ever');
  const e = x.error();
  assert.ok(e, JSON.stringify(x.frames.map(f => f.type)));
  assert.match(e.message, /The assistant stopped answering for .*so I ended this turn\. Anything it already finished is kept\. Send your message again to go on\./);
});

test('(a) events that are not this turn\'s do not keep it alive: after the engine grace the stall limit ends it', async t => {
  let ticks = 0;
  const r = rig(t, { liveMs: 3_600_000, stallMs: 1200, stallToolMs: 1200, engineGraceMs: 500, engineProgress: () => ++ticks });
  const x = r.turn('chat-x');
  await sleep(300);
  let seq = 0;
  const noise = setInterval(() => r.push(0, { type: 'event', event: { seq: ++seq, type: 'step/start', data: {} } }), 100);
  t.after(() => clearInterval(noise));
  assert.equal(await untilEnded(x, 6000), true, 'dropped events kept the turn alive');
  clearInterval(noise);
  assert.match(x.error()?.message ?? '', /The assistant stopped answering/);
});

test('a turn that does become live (its own user message arrives) is not ended by the fuse', async t => {
  let ticks = 0;
  const r = rig(t, { liveMs: 600, stallMs: 4000, stallToolMs: 4000, engineGraceMs: 600_000, engineProgress: () => ++ticks });
  const x = r.turn('chat-x');
  await sleep(300);
  r.push(0, { type: 'event', event: { seq: 1, type: 'user/message', data: { source: { rpcId: 'req-chat-x' } } } });
  await sleep(1500);
  assert.equal(x.ended(), false, 'a live turn waits for the model');
  await r.bridge.stop('chat-x').catch(() => {});
});

test('(c) a restart of the Harness waits for a chat that already holds the running one, and goes on when that chat\'s turn has ended', async t => {
  const gate = deferred();
  let held = 0;
  const r = rig(t, { liveMs: 600, stallMs: 4000, stallToolMs: 4000, hold: async () => { if (++held === 1) await gate.promise; } });
  const x = r.turn('chat-x');                         // gets the first Harness, then waits in selectModel: it holds it without being a turn yet
  await sleep(300);
  assert.equal(r.starts.length, 1);
  r.current.list = [instance('i2')];                  // the model was reloaded: the route set changed
  const y = r.turn('chat-y');                         // needs a Harness that knows the new route
  await sleep(500);
  assert.deepEqual(r.stops, [], 'the Harness is not stopped under a chat that is starting on it');
  assert.equal(r.starts.length, 1);
  gate.resolve();                                      // x goes on, never becomes live, and ends by the fuse: then the restart may happen
  assert.equal(await untilEnded(x, 4000), true);
  await sleep(500);
  assert.deepEqual(r.stops, [1], 'the first Harness is stopped once nobody works on it');
  assert.equal(r.starts.length, 2, 'the second Harness knows the new route');
  assert.deepEqual(r.starts[1].routes, ['m-i2']);
  assert.equal(await untilEnded(y, 4000), true, 'the other chat\'s turn ran on the new Harness and ended too');
});

test('(d) a stream opened on a closed gateway tells its owner at once, and one closed before it opened does too', async () => {
  const gw = new GatewayClient({ baseUrl: 'http://127.0.0.1:9', token: 'x' });
  const first = [];
  const early = gw.openStream('session/follow', { request: {} }, { onError: e => first.push(e.code), onItem: () => first.push('item') });
  gw.close();                                          // closed while the stream was waiting for its socket
  await sleep(50);
  assert.deepEqual(first, ['gateway/closed'], 'a stream that never opened tells its owner when the client is closed');
  assert.equal(gw.closed, true);
  const second = [];
  gw.openStream('session/follow', { request: {} }, { onError: e => second.push(e.code), onItem: () => second.push('item') });
  await sleep(50);
  assert.deepEqual(second, ['gateway/closed'], 'a stream opened on a closed client tells its owner');
  early.close();
});
