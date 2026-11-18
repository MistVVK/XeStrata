// FIX-APP-0216 A5 (soak 11-18): after a Settings -> Tools change (a blocked folder removed) the next message waits for every other chat to be idle before the Harness restarts,
// and with other chats working it failed after the wait with "Another chat is still working and the list of models changed." (agent-busy). The wait is kept (the restart ends the
// other chats' running turns); the sentence now names the real reason: Tools settings for a Tools change, the models sentence for a changed model list.
// The rig is agent-route-race.test.mjs's: the real bridge over fake runtimes, no Harness.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { createAgentTools } from '../server/agent/tools-settings.mjs';

const sleep = ms => new Promise(r => setTimeout(r, ms));
const instance = id => ({ id, label: 'S', baseUrl: 'http://127.0.0.1:9', modelId: `m-${id}`, context: 32768, slots: 2 });
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
const TOOLS_WORDS = 'Another chat is still working, so your new Tools settings are not in use yet. Send your message again when that chat has finished.';
const MODELS_WORDS = 'Another chat is still working and the list of models changed. Try again when it finishes.';

function rig(t, { hold = null, ...options } = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'strata-restartwait-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  let saved = {};
  const tools = createAgentTools({ read: () => saved, write: v => { saved = v; }, dataDir: dir });
  const current = { list: [instance('i1')] };
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
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => current.list, connections: () => [], tools: () => tools.profileView(), env: process.env, startRuntimeImpl, ...options });
  t.after(async () => { await bridge.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const turn = (chatId, extra = {}) => {
    const frames = [];
    const done = bridge.runTurn({ chatId, message: 'hello', workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f), requestId: `req-${chatId}`, ...extra }).catch(e => ({ thrown: e }));
    return { frames, done, ended: () => frames.some(f => f.type === 'done' || f.type === 'error'), error: () => frames.find(f => f.type === 'error') };
  };
  return { bridge, current, starts, stops, turn, tools, ws, dir };
}
/** The sentence a turn ended with: a thrown AgentUnavailable or an error frame. */
const sentence = async turn => { const r = await turn.done; return r?.thrown?.message ?? turn.error()?.message ?? null; };

test('a Tools change with another chat working: the next message ends after the wait with the Tools sentence (code agent-busy), not the models one', async t => {
  const gate = deferred();
  let held = 0;
  const r = rig(t, { restartWaitMs: 500, liveMs: 600, stallMs: 4000, stallToolMs: 4000, hold: async () => { if (++held === 1) await gate.promise; } });
  const x = r.turn('chat-x');                                   // holds the running Harness (selectModel waits): another chat is working
  await sleep(300);
  assert.equal(r.starts.length, 1);
  r.tools.update({ blockedFolders: { add: join(r.dir, 'private') } });   // a change in Settings -> Tools
  const y = r.turn('chat-y');
  const result = await y.done;
  const text = result?.thrown?.message ?? y.error()?.message;
  assert.equal(text, TOOLS_WORDS);
  assert.doesNotMatch(text, /models/i, 'a Tools change is not "the list of models"');
  assert.equal(result?.thrown?.code ?? y.error()?.code, 'agent-busy');
  assert.deepEqual(r.stops, [], 'the running Harness was not stopped under the other chat');
  gate.resolve(); await x.done;
});

test('a changed model list with another chat working keeps the models sentence', async t => {
  const gate = deferred();
  let held = 0;
  const r = rig(t, { restartWaitMs: 500, liveMs: 600, stallMs: 4000, stallToolMs: 4000, hold: async () => { if (++held === 1) await gate.promise; } });
  const x = r.turn('chat-x');
  await sleep(300);
  r.current.list = [instance('i2')];
  const y = r.turn('chat-y');
  assert.equal(await sentence(y), MODELS_WORDS);
  gate.resolve(); await x.done;
});

test('both changed: the Tools sentence (the restart is needed for the settings first)', async t => {
  const gate = deferred();
  let held = 0;
  const r = rig(t, { restartWaitMs: 500, liveMs: 600, stallMs: 4000, stallToolMs: 4000, hold: async () => { if (++held === 1) await gate.promise; } });
  const x = r.turn('chat-x');
  await sleep(300);
  r.current.list = [instance('i2')];
  r.tools.update({ webWithoutAsking: true });
  assert.equal(await sentence(r.turn('chat-y')), TOOLS_WORDS);
  gate.resolve(); await x.done;
});

test('the wait ends as soon as the other chat is done: the restart happens and the message is not refused', async t => {
  const gate = deferred();
  let held = 0;
  const r = rig(t, { restartWaitMs: 30_000, liveMs: 600, stallMs: 4000, stallToolMs: 4000, hold: async () => { if (++held === 1) await gate.promise; } });
  const x = r.turn('chat-x');
  await sleep(300);
  r.tools.update({ blockedFolders: { add: join(r.dir, 'private') } });
  const y = r.turn('chat-y');
  await sleep(300);
  assert.deepEqual(r.stops, [], 'still waiting');
  gate.resolve();                                                 // x goes on, never becomes live and ends by the fuse
  const t0 = Date.now();
  while (!x.ended() && Date.now() - t0 < 5000) await sleep(50);
  assert.equal(x.ended(), true);
  const t1 = Date.now();
  while (r.starts.length < 2 && Date.now() - t1 < 5000) await sleep(50);
  assert.equal(r.starts.length, 2, 'the Harness restarted for the new Tools settings');
  assert.deepEqual(r.stops, [1]);
  const text = await sentence(y);
  assert.ok(!text || !/Tools settings are not in use yet/.test(text), text);
});
