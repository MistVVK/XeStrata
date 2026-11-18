// FIX-APP-0216 A6, live check with the real Harness (skipped unless STRATA_HARNESS_DIR names a built Harness, like agent-blocked-live.test.mjs): one running turn of a chat, the page
// that started it plus two pages that take it over (one attaches early, one late, one of them leaves early), and every page that stayed has the whole reply, the catch-up having gone
// to the newcomer only. The model is the scripted worker (SCRIPT:slow: 120 text chunks, 40 ms apart); the Harness, the gateway client, the session follows and the bridge are real.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const sleep = ms => new Promise(r => setTimeout(r, ms));

test('real Harness: pages that join a running turn all get the live frames; the page that started it keeps them; a page that leaves drops out alone', { skip, timeout: 120_000 }, async t => {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-attachlive-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 2 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const page = () => { const p = { frames: [], onFrame: f => p.frames.push(f), at: [] }; p.text = () => { let s = ''; for (const f of p.frames) { if (f.type === 'resync') s = f.content; if (f.type === 'delta') s += f.text; } return s; }; return p; };
  const A = page(), B = page(), C = page(), leave = new AbortController();
  const turn = bridge.runTurn({ chatId: 'chat-live', message: 'SCRIPT:slow 120', workspacePath: ws, mode: 'bypass', onFrame: A.onFrame });
  const until = async (fn, ms = 20_000) => { const t0 = Date.now(); while (!fn() && Date.now() - t0 < ms) await sleep(20); return fn(); };
  assert.equal(await until(() => A.frames.filter(f => f.type === 'delta').length >= 5), true, 'the first page is receiving the reply');
  const b = bridge.attach({ chatId: 'chat-live', onFrame: B.onFrame, signal: leave.signal });
  assert.ok(b, 'the running turn can be taken over');
  const deltasAtJoinA = A.frames.filter(f => f.type === 'delta').length;
  await until(() => B.frames.filter(f => f.type === 'delta').length >= 10);
  await until(() => A.frames.filter(f => f.type === 'delta').length >= deltasAtJoinA + 30);
  assert.ok(A.frames.filter(f => f.type === 'delta').length >= deltasAtJoinA + 30, 'the page that started the turn keeps receiving live frames after the second page joined');
  const c = bridge.attach({ chatId: 'chat-live', onFrame: C.onFrame });
  assert.ok(c);
  leave.abort();                                          // the second window is closed mid-turn
  const bSeen = B.frames.length;
  await turn; await c.ended;
  assert.equal(A.frames.at(-1).type, 'done'); assert.equal(C.frames.at(-1).type, 'done');
  const full = A.text();
  assert.ok(full.length > 200, `the reply is long enough to mean something (${full.length} chars)`);
  assert.equal(C.text(), full, 'the late page has the same reply as the page that started the turn');
  assert.equal(B.frames.length, bSeen, 'nothing more reached the page that left');
  assert.equal(A.frames.filter(f => f.type === 'resync').length, 0);
  assert.equal(C.frames.filter(f => f.type === 'resync').length, 1);
});
