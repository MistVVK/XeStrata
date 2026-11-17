// #5996: the helper time limit (Settings > General; Off by default, or 30 minutes, 1 hour, 2 hours) and the note on a helper that has worked long.
// Pure tests always run; the one with the real Harness needs STRATA_HARNESS_DIR and skips loudly without it.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { transform } from 'esbuild';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { ChildProjector, HELPER_LIMITS, helperLimitMin } from '../server/agent/children.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;
const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const load = async name => {
  const { code } = await transform(src(name), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
};
const ev = (type, data, time = 1000) => ({ type: 'event', event: { seq: 1, time, type, data } });
const LANGS = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];

test('the choices are Off, 30, 60 and 120 minutes, on the host and on the page alike; anything else is Off', async () => {
  assert.deepEqual([...HELPER_LIMITS], [30, 60, 120]);
  for (const m of HELPER_LIMITS) { assert.equal(helperLimitMin(m), m); assert.equal(helperLimitMin(String(m)), m); }
  for (const v of [undefined, null, '', 0, 1, 29, 31, 90, '45', 'off', true, false, {}, [], NaN, -30]) assert.equal(helperLimitMin(v), null, String(v));
  const { HELPER_LIMITS: page, validHelperLimit, HELPER_NOTE_MS, helperRanLong } = await load('lib/helper-limit.ts');
  assert.deepEqual([...page], [...HELPER_LIMITS], 'the page offers what the host accepts');
  for (const m of page) assert.equal(validHelperLimit(m), m);
  assert.equal(validHelperLimit(null), null); assert.equal(validHelperLimit(45), null);
  assert.equal(HELPER_NOTE_MS, 30 * 60_000);
  assert.equal(helperRanLong('working', 30 * 60_000), false);
  assert.equal(helperRanLong('working', 30 * 60_000 + 1), true);
  assert.equal(helperRanLong('completed', 5 * 3600_000), false, 'the note is for a helper that is still working');
});

test('a helper counts its time from when it first works, not from its wait; a stop by the limit says so; a restart clears it', () => {
  let t = 1000;
  const cp = new ChildProjector({ id: 'h', name: 'Check', now: () => t });
  assert.equal(cp.workStartedAt, null, 'queued: not counting');
  t = 5000; cp.consume(ev('tool/call', { callId: 'a', name: 'read', arguments: '{"file_path":"x"}' }));
  assert.equal(cp.agent().status, 'working'); assert.equal(cp.workStartedAt, 5000);
  t = 9000; cp.consume(ev('tool/result', { message: { toolCallId: 'a', isError: false, content: [{ type: 'text', text: 'ok' }] } }));
  assert.equal(cp.workStartedAt, 5000, 'it stays the time it first went to working');
  cp.limitReached = true;
  cp.consume(ev('turn/end', { reason: { kind: 'aborted' } }));
  assert.equal(cp.agent().status, 'cancelled');
  assert.equal(cp.agent().description, 'Stopped: it reached the helper time limit.');
  cp.limitReached = false;
  assert.equal(cp.agent().description, 'Stopped.', 'a stop by the person keeps its own words');
  t = 20000; cp.consume(ev('turn/start', {}));                                   // a new message after it finished
  assert.equal(cp.workStartedAt, null); assert.equal(cp.limitReached, false);
});

test('the page: the row notes a helper that worked over 30 minutes, Settings has the choice, the host has the endpoint and reads it live', () => {
  const row = src('components/SubagentRow.tsx');
  assert.ok(row.includes('helperRanLong(a.status, duration) ? <p className="small subagent-row__note" role="status">{tr("This helper has been working for over 30 minutes.")}</p> : null'));
  assert.ok(row.includes('<p title={a.description}>{tr(a.description)}</p>'));
  const settings = src('screens/Settings.tsx');
  assert.ok(settings.includes("{ section: 'General', label: 'Helper time limit', hint: 'Stops a helper that keeps working longer than this. Off lets helpers run as long as they need.'"));
  assert.ok(src('components/HelperLimitSetting.tsx').includes("{ value: 'off', label: 'Off' }, { value: '30', label: '30 minutes' }, { value: '60', label: '1 hour' }, { value: '120', label: '2 hours' }"));
  const index = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.match(index, /helperLimitMs: \(\) => \(helperLimitMin\(store\.get\(\)\.agentChat\?\.helperLimitMin\) \?\? 0\) \* 60_000,/);
  assert.match(index, /path === '\/api\/agent\/helper-limit' && method === 'POST'/);
  assert.match(readFileSync(new URL('../server/store.mjs', import.meta.url), 'utf8'), /agentChat: \{ thresholdRatio: null, headroomFraction: null, helperLimitMin: null \}/);
});

test('the texts are in the catalog in all ten languages ("Off" is the existing string)', () => {
  const cat = Object.fromEntries(LANGS.map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/guic-${l}.json`, import.meta.url), 'utf8'))]));
  for (const text of ['Helper time limit', 'Stops a helper that keeps working longer than this. Off lets helpers run as long as they need.', '30 minutes', '1 hour', '2 hours',
    'This helper has been working for over 30 minutes.', 'Stopped: it reached the helper time limit.']) {
    const key = Object.keys(cat.en).find(k => cat.en[k] === text);
    assert.ok(key, `${text} is in guic-en.json`);
    for (const l of LANGS) assert.ok(cat[l][key] && cat[l][key].trim(), `${l}: ${key}`);
    assert.equal(/[^  ][;:?!]| [;:?!]/.test(cat.fr[key]), false, `French: ${text}`);
  }
});

// ---------------------------------------------------------------------------------- with the real Harness

async function rig(t, helperLimitMs) {
  const worker = await startScriptedWorker({});
  const dir = mkdtempSync(join(tmpdir(), 'strata-hl-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const local = [{ id: 'inst-1', label: 'Scripted', baseUrl: worker.url, modelId: 'scripted-model', context: 32768, slots: 4 }];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => local, connections: () => [], env: process.env, startTimeoutMs: 90_000, helperLimitMs });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  return { bridge, ws };
}
const run = async (r, chatId, message) => {
  const frames = [];
  await r.bridge.runTurn({ chatId, message, workspacePath: r.ws, mode: 'bypass', onFrame: f => frames.push({ ...f, at: Date.now() }) });
  return frames;
};
const lastRows = frames => { const m = new Map(); for (const f of frames.filter(x => x.type === 'agent')) m.set(f.agent.id, f.agent); return [...m.values()]; };

test('a helper that works past the limit is stopped, the chat goes on', { skip, timeout: 180_000 }, async (t) => {
  const r = await rig(t, () => 1500);                                         // 1.5 s here; the choices are minutes, the mechanism is the same
  const started = Date.now();
  const frames = await run(r, 'limited', 'SCRIPT:spawn 1 slow 2000');          // a helper that would work for about 80 s
  const [helper] = lastRows(frames);
  assert.equal(helper.status, 'cancelled');
  assert.equal(helper.description, 'Stopped: it reached the helper time limit.');
  assert.ok(Date.now() - started < 60_000, 'it did not run its 80 s');
  const done = frames.at(-1);
  assert.equal(done.type, 'done'); assert.notEqual(done.cancelled, true, 'the chat itself was not stopped');
});

test('with no limit a helper is left alone', { skip, timeout: 180_000 }, async (t) => {
  const r = await rig(t, () => 0);
  const frames = await run(r, 'unlimited', 'SCRIPT:spawn 1 slow 40');
  const [helper] = lastRows(frames);
  assert.equal(helper.status, 'completed');
  assert.notEqual(helper.description, 'Stopped: it reached the helper time limit.');
});
