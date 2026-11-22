// The turn guard: a search-heavy /plan that keeps failing searches and rewriting its plan is stopped by the app and ends with a plain message;
// a turn with no answer never ends silently. The pure counting always runs; the end-to-end ones need a built Harness (STRATA_HARNESS_DIR) and
// run the real search code on a pretend web that refuses (no network).
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { TurnGuard, GUARD_LIMITS, GUARD_NOTES, guardNote } from '../server/agent/turn-guard.mjs';
import { ProgressGuard } from '../server/agent/progress-guard.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

const search = (id, state, result = '') => ({ id, name: 'web_search', state, result });
const LIMITED = 'Web search is limited right now. Try again in a minute, or add a search key (Exa, Perplexity or DeepSeek) in Settings → Tools → Web search. Do not retry this search';

test('three failed searches in a row stop the turn; a search that worked starts the count again', () => {
  const g = new TurnGuard({ plan: true });
  assert.equal(g.card(search('a', 'running')), null, 'a running search is not counted yet');
  assert.equal(g.card(search('a', 'error', LIMITED)), null);
  assert.equal(g.card(search('a', 'error', LIMITED)), null, 'the same card twice is one search');
  assert.equal(g.card(search('b', 'done', 'three results')), null);
  assert.equal(g.card(search('c', 'error', LIMITED)), null);
  assert.equal(g.card(search('d', 'done', LIMITED)), null, 'a refusal that arrives as a normal result still counts');
  assert.equal(g.card(search('e', 'error', LIMITED)), 'searches-failed');
  assert.equal(g.card(search('f', 'error', LIMITED)), 'searches-failed', 'once stopped it stays stopped');
  assert.equal(g.searches, 5);
});

test('other tools are not searches; too many searches in one turn stop it even when they work', () => {
  const g = new TurnGuard({ plan: true });
  for (let i = 0; i < 50; i++) assert.equal(g.card({ id: `r${i}`, name: 'read', state: 'done' }), null);
  for (let i = 0; i < GUARD_LIMITS.searchesPlan - 1; i++) assert.equal(g.card(search(`s${i}`, 'done', 'ok')), null);
  assert.equal(g.card(search('last', 'done', 'ok')), 'searches');
  const t = new TurnGuard({ plan: false });
  for (let i = 0; i < GUARD_LIMITS.searchesTurn - 1; i++) assert.equal(t.card(search(`s${i}`, 'done', 'ok')), null);
  assert.equal(t.card(search('last', 'done', 'ok')), 'searches');
});

test('a plan turn that writes its numbered plan from 1. again and again is stopped; a single plan is not', () => {
  const g = new TurnGuard({ plan: true });
  const plan = '## Plan\n\n1. First\n2. Second\n3. Third\n\n';
  let reason = null;
  for (let i = 0; i < GUARD_LIMITS.planRewrites - 1; i++) reason ??= g.text(plan);
  assert.equal(reason, null, `${GUARD_LIMITS.planRewrites - 1} versions are fine`);
  assert.equal(g.text(plan), 'rewrites');
  const one = new TurnGuard({ plan: true });
  for (const piece of ['Some looking around.\n\n## Plan\n\n', '1. First\n', '2. Second\n', '   1. a sub point\n', '3. Third\n']) assert.equal(one.text(piece), null, piece);
  const ordinary = new TurnGuard({ plan: false });
  for (let i = 0; i < 10; i++) assert.equal(ordinary.text(plan), null, 'only a plan turn counts rewrites');
});

test('a turn that ends with no words reports whether any tool work was done', () => {
  const g = new TurnGuard();
  const done = { kind: 'completed' };
  assert.equal(g.silent({ ended: done, wrote: false }), 'empty');
  assert.equal(g.silent({ ended: done, wrote: false, acted: true }), 'work-only', 'tool cards alone do not finish the answer');
  assert.equal(g.silent({ ended: done, wrote: true }), null, 'there are words: nothing to add');
  assert.equal(g.silent({ ended: { kind: 'error', error: { message: 'x' } }, wrote: false }), null, 'an error says why by itself');
  assert.equal(g.silent({ ended: done, wrote: false, stoppedByPerson: true }), null, 'Stop is the person\'s own');
});

test('every message is a plain sentence with a next step, carries its interface key, and is in the catalog in all nine languages', () => {
  const langs = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
  const catalogs = Object.fromEntries(langs.map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/${l}.json`, import.meta.url), 'utf8'))]));
  const idOf = text => Object.entries(catalogs.en).find(([, v]) => v === text)?.[0];
  for (const [reason, text] of Object.entries(GUARD_NOTES)) {
    const frame = guardNote(reason);
    assert.equal(frame.type, 'delta'); assert.equal(frame.key, text); assert.equal(frame.own, true);
    assert.ok(text.length < 220, reason);
    assert.match(text, /(Try again|Tell me|Say|Send|approve|resume)/i, `${reason} gives a next step`);
    const id = idOf(text);
    assert.ok(id, `${reason} is in the English catalog`);
    for (const l of langs) assert.ok(catalogs[l][id]?.trim(), `${reason} in ${l}`);
  }
});

// ------------------------------------------------------------------------------------------------ with the real Harness

async function rig(t, options = {}) {
  const worker = await startScriptedWorker({ chunkDelayMs: 5 });
  const dir = mkdtempSync(join(tmpdir(), 'strata-guard-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const logs = [];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: options.context ?? 32768, slots: 2 }], connections: () => [],
    // these chats are in Manual and search without a person at the card: the Settings switch that skips the web question is on
    tools: () => ({ off: new Set(), search: { provider: 'builtin', key: '' }, servers: {}, webWithoutAsking: true }),
    env: process.env, startTimeoutMs: 90_000, searchMock: 'limited', log: l => logs.push(l) });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  return { worker, bridge, ws, logs, base: (chatId, onFrame) => ({ chatId, workspacePath: ws, mode: 'manual', onFrame }) };
}
const words = frames => frames.filter(f => f.type === 'delta').map(f => f.text).join('');

test('progress follows changed results and recognizes reordered arguments and repeated outcomes', () => {
  const guard = new ProgressGuard();
  const step = (id, args, result, isError) => {
    guard.observe({ type: 'tool/call', data: { callId: id, name: 'read', arguments: args } });
    guard.observe({ type: 'tool/result', data: { message: { toolCallId: id, content: [{ type: 'text', text: result }], isError } } });
    return guard.observe({ type: 'step/end' });
  };
  for (let i = 0; i < 30; i++) assert.equal(step(i + 1, { path: 'x' }, String(i)), null);
  assert.equal(step(31, '{"path":"x","offset":0}', 'same'), null);
  for (let i = 32; i < 34; i++) assert.equal(step(i, '{"offset":0,"path":"x"}', 'same'), null);
  assert.equal(step(34, { path: 'x', offset: 0 }, 'same'), 'repeated-work');
});

test('changing errors do not let a run retry indefinitely', () => {
  const guard = new ProgressGuard();
  for (let id = 1; id <= 12; id++) {
    guard.observe({ type: 'tool/call', data: { callId: id, name: 'read', arguments: { path: String(id) } } });
    guard.observe({ type: 'tool/result', data: { message: { toolCallId: id, content: [{ type: 'text', text: 'missing' }], isError: true } } });
    assert.equal(guard.observe({ type: 'step/end' }), id === 12 ? 'no-progress' : null);
  }
});

const resultStep = (guard, name, args, text) => {
  guard.observe({ type: 'tool/call', data: { callId: 'call', name, arguments: args } });
  guard.observe({ type: 'tool/result', data: { message: { toolCallId: 'call', isError: false, content: [{ type: 'text', text }] } } });
  return guard.observe({ type: 'step/end' });
};

test('repeated test reports do not make progress by changing only runner durations', () => {
  const reports = [
    ['node --test fixture.mjs', n => '✔ check settings (' + n + '.123ms)\nℹ pass 1\nℹ duration_ms ' + n + '.4'],
    ['python -m pytest -q', n => '......... [100%]\n9 passed in ' + n + '.4s'],
    ['ctest --test-dir build', n => '1/1 Test #1: Settings .... Passed    ' + n + '.1 sec\nTotal Test time (real) = ' + n + '.1 sec'],
  ];
  for (const [command, report] of reports) {
    const guard = new ProgressGuard();
    for (let run = 1; run <= 4; run++) {
      const result = resultStep(guard, 'pwsh', { command, description: 'Run ' + run }, report(run));
      assert.equal(result, run === 4 ? 'repeated-work' : null, command);
    }
  }
});

test('test outcome changes, successful edits and non-test shell output still count as progress', () => {
  const guard = new ProgressGuard();
  const command = 'node --test fixture.mjs';
  for (let i = 1; i <= 5; i++) {
    if (i > 1) assert.equal(resultStep(guard, 'edit', { file_path: 'fixture.mjs', new_string: String(i) }, 'Changed the file.'), null);
    assert.equal(resultStep(guard, 'pwsh', { command }, '✔ check settings (1.123ms)\nℹ pass 1'), null);
  }
  for (let i = 1; i <= 5; i++) {
    assert.equal(resultStep(guard, 'bash', { command }, '✖ check settings (1.123ms)\nfound ' + i + ' unexpected values'), null);
    assert.equal(resultStep(guard, 'bash', { command: 'node --test fixture.mjs; node monitor.mjs' }, 'ℹ duration_ms ' + i), null);
  }
});

test('the real Harness stops repeated reads, while 14 distinct reads finish and report', { skip }, async t => {
  const r = await rig(t);
  for (let i = 0; i < 14; i++) writeFileSync(join(r.ws, `fixture-${i}.txt`), `fixture ${i}\n`);
  const repeat = [];
  const run = { ...r.base('repeat', f => repeat.push(f)), mode: 'plan', requestId: 'repeat-case', message: 'SCRIPT:repeatread' };
  await r.bridge.runTurn(run);
  assert.equal(repeat.at(-1).noteReason, 'repeated-work');
  const requests = directive => r.worker.requests.filter(q => q.directive === directive && q.toolNames.length);
  assert.ok(requests('repeatread').length >= 4 && requests('repeatread').length <= 5);
  const attached = [];
  const replay = r.bridge.attach({ chatId: 'repeat', requestId: 'repeat-case', onFrame: f => attached.push(f) });
  assert.equal(replay.live, false);
  assert.equal(attached.filter(f => f.type === 'delta' && f.noteReason === 'repeated-work').length, 1);
  assert.equal(attached.at(-1).noteReason, 'repeated-work');

  const work = [];
  await r.bridge.runTurn({ ...r.base('work', f => work.push(f)), mode: 'plan', message: 'SCRIPT:distinctreads' });
  assert.equal(requests('distinctreads').length, 15);
  assert.match(words(work), /Checked all fixtures/);
  assert.equal(work.at(-1).noteReason, undefined);
});

test('an empty completion after a real tool result is reported without repeating the empty request', { skip }, async t => {
  const r = await rig(t);
  writeFileSync(join(r.ws, 'fixture-0.txt'), 'completed step\n');
  const out = [];
  await r.bridge.runTurn({ ...r.base('empty-after-read', f => out.push(f)), mode: 'plan', message: 'SCRIPT:emptyafterread' });
  assert.equal(r.worker.requests.filter(q => q.directive === 'emptyafterread' && q.toolNames.length).length, 2);
  assert.ok(out.some(f => f.type === 'agent-tool' && f.tool.state === 'done'));
  assert.equal(out.at(-1).key, GUARD_NOTES['work-only']);
  assert.equal(out.at(-1).code, 'EMPTY_RESPONSE');
});

test('periodic reasoning stops during the stream; distinct reasoning and new attempts do not trip the guard', () => {
  const guard = new ProgressGuard();
  const frame = (text, attemptId) => ({ type: 'chunk', attemptId, chunk: { type: 'reasoning-delta', text } });
  for (let i = 0; i < 20; i++) assert.equal(guard.observeStream(frame(`${i}.`.repeat(30), i)), null);
  const distinct = Array.from({ length: 800 }, (_, i) => String(i)).join(' ');
  assert.equal(guard.observeStream(frame(distinct, 21)), null);
  const repeated = 'I will look again at the same option and retrace the same checks before deciding what to do. ';
  assert.equal(guard.observeStream(frame(repeated.repeat(24), 22)), 'repeated-thinking');
});

test('the real Harness aborts a repeated reasoning stream and reports why', { skip }, async t => {
  const r = await rig(t);
  const out = [];
  await r.bridge.runTurn({ ...r.base('repeat-thinking', f => out.push(f)), mode: 'plan', message: 'SCRIPT:repeatthinking' });
  assert.equal(out.at(-1).noteReason, 'repeated-thinking');
  assert.ok(r.worker.aborted > 0, 'the model stream was interrupted');
});

test('a looping helper stops while an independent helper finishes its work', { skip }, async t => {
  const r = await rig(t);
  for (let i = 0; i < 14; i++) writeFileSync(join(r.ws, `fixture-${i}.txt`), `fixture ${i}\n`);
  const frames = [];
  await r.bridge.runTurn({ ...r.base('two-helpers', f => frames.push(f)), mode: 'bypass', message: 'SCRIPT:spawn 2 mixedguard' });
  const rows = new Map(frames.filter(f => f.type === 'agent').map(f => [f.agent.name, f.agent]));
  assert.equal(rows.get('Helper 1')?.status, 'cancelled');
  assert.equal(rows.get('Helper 1')?.description, GUARD_NOTES['repeated-work']);
  assert.equal(rows.get('Helper 2')?.status, 'completed');
  assert.match(rows.get('Helper 2')?.description ?? '', /Checked all fixtures/);
  assert.equal(frames.at(-1).type, 'done');
});

test('a search-heavy /plan on a web that keeps refusing stops after 3 failed searches, proposes the plan it has, and says why', { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  await r.bridge.planCommand({ ...r.base('p1', f => out.push(f)), line: 'SCRIPT:searchloop' });
  const cards = out.filter(f => f.type === 'agent-tool' && f.tool.name === 'web_search' && ['done', 'error'].includes(f.tool.state));
  assert.ok(cards.length >= 3 && cards.length <= 4, `searches stopped at the limit (${cards.length})`);
  assert.ok(cards.some(f => /Web search is limited right now/.test(f.tool.result ?? '')), 'the tool said web search is limited');
  assert.equal(out.filter(f => f.type === 'error').length, 0, 'not an error: a normal end');
  const done = out.find(f => f.type === 'done');
  assert.ok(done && !done.cancelled, 'the turn ended normally');
  const plan = out.find(f => f.type === 'plan');
  assert.equal(plan?.event, 'proposed', 'the plan it had is offered');
  assert.equal(plan.plan.steps.length, 3);
  assert.equal(plan.plan.caution, GUARD_NOTES['searches-failed'], 'and the card says why it stopped early');
  const requests = r.worker.requests.filter(q => q.directive === 'searchloop');
  assert.ok(requests.length <= 5, `the model was not asked again and again (${requests.length} requests)`);
  assert.ok(r.logs.some(l => /turn guard: searches-failed/.test(l)));
});

test('an ordinary turn whose searches keep failing ends with the plain sentence as its last words', { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  await r.bridge.runTurn({ ...r.base('c1', f => out.push(f)), message: 'SCRIPT:searchloop' });
  assert.equal(out.filter(f => f.type === 'error').length, 0);
  assert.ok(out.find(f => f.type === 'done'));
  const note = out.filter(f => f.type === 'delta' && f.key).at(-1);
  assert.equal(note?.key, GUARD_NOTES['searches-failed']);
  assert.match(words(out), /Settings → Tools → Web search/);
  assert.ok(out.findIndex(f => f.key) < out.findIndex(f => f.type === 'done'), 'the sentence comes before the end of the turn');
});

test('large reported usage preserves a completed reply and does not start post-turn compaction', { skip }, async (t) => {
  const r = await rig(t, { context: 32768 });
  const out = [];
  await r.bridge.runTurn({ ...r.base('f1', f => out.push(f)), message: 'SCRIPT:bloat 32000 Verified final answer.' });
  assert.equal(words(out), 'Verified final answer.');
  assert.equal(out.find(f => f.type === 'done')?.finishReason, 'stop');
  assert.equal(out.some(f => f.noteReason), false);
  const out2 = [];
  await r.bridge.runTurn({ ...r.base('f2', f => out2.push(f)), message: 'SCRIPT:bloat 32000' });
  assert.equal(out2.filter(f => f.type === 'delta' && f.key).at(-1)?.key, GUARD_NOTES.empty);
  assert.equal(r.worker.requests.length, 2, 'only the two requested replies use the model');
  assert.ok(r.logs.every(line => !line.includes('compact after a stopped turn')));
});
