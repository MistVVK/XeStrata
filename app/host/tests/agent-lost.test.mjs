// AGENT-LOST (11-01): a reply the length limit cut inside a tool call used to vanish: no file, no card, no sentence.
// Pure parts always run; the end-to-end ones need a built Harness (STRATA_HARNESS_DIR) and use the scripted model server.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, existsSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { replyCap, unescapeJsonTail, ToolDraft, cutCalls, cutNotice, cutNudge } from '../server/agent/reply-cut.mjs';
import { TurnProjector } from '../server/agent/events.mjs';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge, MAX_CUT_RETRIES } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { localOutputBudget } from '../server/agent/profile.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

test('the reply ceiling is half the window up to 64K tokens and never below what compaction reserves', () => {
  assert.equal(replyCap(65536, localOutputBudget(65536)), 32768);
  assert.ok(replyCap(65536, localOutputBudget(65536)) > localOutputBudget(65536), '19,660 was the cut that lost the pagoda');
  assert.equal(replyCap(131072, localOutputBudget(131072)), 65536);
  assert.equal(replyCap(262144, localOutputBudget(262144)), 65536, 'at most 64K tokens, so a loop cannot run for an hour');
  assert.equal(replyCap(8192, localOutputBudget(8192)), 4096);
  assert.equal(replyCap(0, 5000), 16384, 'no window known: 32K is assumed');
  assert.equal(replyCap(4096, 4096), 4096, 'never below the reserve');
});

test('JSON text is read back for the preview, and a half-written escape is dropped', () => {
  assert.equal(unescapeJsonTail('a\\nb\\t\\"c\\"\\\\d'), 'a\nb\t"c"\\d');
  assert.equal(unescapeJsonTail('line\\'), 'line');
  assert.equal(unescapeJsonTail('x\\u00e9y'), 'xéy');
  assert.equal(unescapeJsonTail('x\\u00'), 'x');
});

test('a call being written knows its file, its size, its lines and the tail of its text', () => {
  const d = new ToolDraft('c1', 'write');
  for (const piece of ['{"file_pa', 'th":"C:\\\\work\\\\pagoda.html","con', 'tent":"<!doctype html>\\n<html>\\n']) d.add(piece);
  assert.equal(d.target, 'C:\\work\\pagoda.html');
  assert.equal(d.lines, 2);
  assert.equal(d.preview(), '<!doctype html>\n<html>\n');
  for (let i = 0; i < 400; i++) d.add(`<div class=\\"row${i}\\">text ${i}</div>\\n`);
  assert.ok(d.bytes > 10_000);
  assert.equal(d.lines, 402);
  const p = d.preview(1000);
  assert.ok(p.length <= 1000 && p.endsWith('</div>\n') && !p.includes('\\n') && p.includes('class="row'), 'the tail, read back, ends where the model is');
  const f = d.frame({ scope: 'write', verb: 'Writing' });
  assert.deepEqual(Object.keys(f.draft).sort(), ['bytes', 'done', 'id', 'lines', 'name', 'preview', 'scope', 'target', 'verb']);
  assert.equal(d.frame({ done: true }).draft.preview, '', 'a finished draft carries no text');
});

test('the file is found when the model writes its name after the text, or split between two pieces', () => {
  const d = new ToolDraft('c2', 'write');
  d.add('{"content":"line one\nline two\n');
  assert.equal(d.target, '');
  d.add('more","file_pa'); d.add('th":"/w/big.txt"}');
  assert.equal(d.target, '/w/big.txt');
});

test('the sentences name the file and the size, say what to do, and the nudge asks for smaller parts', () => {
  const draft = new ToolDraft('a', 'write'); draft.add('{"file_path":"/w/pagoda.html","content":"'); draft.bytes = 52_300;
  const calls = cutCalls([draft]);
  assert.deepEqual(calls, [{ name: 'write', target: '/w/pagoda.html', bytes: 52_300 }]);
  const again = cutNotice({ calls, tokens: 19660, retrying: true });
  assert.match(again.text, /reached its length limit \(19,660 tokens\) while writing pagoda\.html \(52 KB\), so that write did not happen\. I asked the assistant to look at what is on disk and write the rest in smaller parts\./);
  const last = cutNotice({ calls, tokens: 19660 });
  assert.match(last.text, /Say “continue”/);
  assert.match(cutNotice({ tokens: 500 }).text, /^The reply reached its length limit \(500 tokens\) and was cut off\. Say “continue”/);
  assert.match(cutNudge({ calls, tokens: 19660 }), /writing \/w\/pagoda\.html in one tool call, so that call did not run[\s\S]*Do not delete or start over a file that is already complete[\s\S]*smaller parts/);
  assert.match(cutNudge({}), /cut off by the output limit/);
  for (const n of [again, last]) assert.equal(n.text, n.key.replace(/\{(\d)\}/g, (_, i) => n.values[i]), 'the key and its values make the sentence');
});

// ------------------------------------------------------------------------------------------------ the projector

const chunk = (c, attemptId = 'a1') => ({ type: 'assistant-stream', frame: { type: 'chunk', attemptId, chunk: c } });
const event = (type, data = {}) => ({ type: 'event', event: { type, time: Date.now(), data } });

test('the projector streams a growing draft frame for a call being written, and the whole call ends it', () => {
  const p = new TurnProjector();
  const frames = [];
  const head = '{"file_path":"big.txt","content":"';
  frames.push(...p.consume(chunk({ type: 'tool-call-delta', id: 'c1', name: 'write', argumentsDelta: head })));
  assert.equal(frames.filter(f => f.type === 'agent-tool-draft').length, 1, 'the first piece shows at once');
  const first = frames.find(f => f.type === 'agent-tool-draft').draft;
  assert.equal(first.name, 'write'); assert.equal(first.target, 'big.txt'); assert.equal(first.scope, 'write'); assert.equal(first.verb, 'Writing a file');
  // A burst inside 200 ms sends no more frames, but the next one after the pause carries everything so far.
  for (let i = 0; i < 20; i++) p.consume(chunk({ type: 'tool-call-delta', id: 'c1', argumentsDelta: `entry ${i}\\n` }));
  assert.equal(p.composing.get('c1').lines, 20);
  p.composing.get('c1').at = 0;
  const later = p.consume(chunk({ type: 'tool-call-delta', id: 'c1', argumentsDelta: 'entry 20\\n' })).find(f => f.type === 'agent-tool-draft').draft;
  assert.equal(later.lines, 21); assert.match(later.preview, /entry 20\n$/);
  const out = p.consume(event('tool/call', { callId: 'c1', name: 'write', arguments: '{"file_path":"big.txt","content":"x"}' }));
  assert.ok(out.some(f => f.type === 'agent-tool'), 'the ordinary card takes over');
  assert.equal(out.find(f => f.type === 'agent-tool-draft')?.draft.done, true, 'and the draft is closed');
  assert.equal(p.composing.size, 0);
});

test('a reply cut in a call: the done frame says length and what was lost, and the draft is taken down', () => {
  const p = new TurnProjector();
  p.consume(chunk({ type: 'tool-call-delta', id: 'c9', name: 'write', argumentsDelta: '{"file_path":"/w/pagoda.html","content":"abc' }));
  p.consume(chunk({ type: 'usage', usage: { inputTokens: 6000, outputTokens: 19660 } }));
  p.consume(event('turn/end', { reason: { kind: 'max-tokens' } }));
  assert.deepEqual(p.cut(), { tokens: 19660, calls: [{ name: 'write', target: '/w/pagoda.html', bytes: p.composing.get('c9').bytes }] });
  const frames = p.finish({ model: 'm', contextMax: 65536 });
  const drop = frames.find(f => f.type === 'agent-tool-draft');
  assert.ok(drop && drop.draft.done && drop.draft.lost, 'the writing card goes away');
  const done = frames.at(-1);
  assert.equal(done.type, 'done'); assert.equal(done.finishReason, 'length');
  assert.equal(done.cut.tokens, 19660); assert.equal(done.cut.calls[0].target, '/w/pagoda.html');
});

test('a call nobody writes a card for (a helper agent) makes no draft', () => {
  const p = new TurnProjector();
  const out = p.consume(chunk({ type: 'tool-call-delta', id: 's1', name: 'subagent', argumentsDelta: '{"prompt":"x' }));
  assert.equal(out.filter(f => f.type === 'agent-tool-draft').length, 0);
});

// ------------------------------------------------------------------------------------------------ with the real Harness

async function rig(t, { window = 65536 } = {}) {
  const worker = await startScriptedWorker({ chunkDelayMs: 5 });
  const dir = mkdtempSync(join(tmpdir(), 'strata-lost-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const logs = [];
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: window, slots: 2 }], connections: () => [],
    env: process.env, startTimeoutMs: 90_000, log: l => logs.push(l) });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  return { worker, bridge, ws, logs, base: (chatId, onFrame) => ({ chatId, workspacePath: ws, mode: 'bypass', thinking: 'Off', onFrame }) };
}
const words = frames => frames.filter(f => f.type === 'delta').map(f => f.text).join('');
const capOf = q => q.params.max_tokens ?? q.params.max_completion_tokens;
const writes = r => r.worker.requests.filter(q => q.directive === 'cutwrite' || q.directive === 'cutalways');

test('every model request may produce half the window, not the 30 percent the Harness reserves for compaction', { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  await r.bridge.runTurn({ ...r.base('c0', f => out.push(f)), message: 'SCRIPT:echo hello' });
  const asked = r.worker.requests.find(q => q.directive === 'echo');
  assert.ok(asked, 'the model was asked');
  assert.equal(capOf(asked), 32768, `the request carried ${capOf(asked)}; the pagoda's carried 19660`);
});

test('a small window still gets a request that fits it (the Harness clamps to the room left)', { skip }, async (t) => {
  const r = await rig(t, { window: 8192 });
  await r.bridge.runTurn({ ...r.base('c0s', () => {}), message: 'SCRIPT:echo hello' });
  const asked = r.worker.requests.find(q => q.directive === 'echo');
  assert.ok(capOf(asked) > 0 && capOf(asked) <= 4096, `max_tokens ${capOf(asked)} for an 8K window`);
});

test('a reply cut inside a file write: the chat says so, the agent is asked to go on in smaller parts, and the file gets written', { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  await r.bridge.runTurn({ ...r.base('c1', f => out.push(f)), message: 'SCRIPT:cutwrite big.txt' });
  const said = out.find(f => f.type === 'delta' && f.noteReason === 'cut');
  assert.ok(said, 'a sentence about the cut');
  assert.match(said.text, /while writing big\.txt/); assert.match(said.text, /that write did not happen. I asked the assistant to look at what is on disk and write the rest in smaller parts/);
  assert.equal(out.filter(f => f.type === 'error').length, 0);
  const done = out.filter(f => f.type === 'done').at(-1);
  assert.equal(done.finishReason, 'stop', 'the turn ends on the agent\'s own finish, not on the cut');
  assert.ok(existsSync(join(r.ws, 'big.txt')), 'the file is on disk');
  assert.match(readFileSync(join(r.ws, 'big.txt'), 'utf8'), /entry 0001/);
  assert.match(words(out), /Written in parts/);
  assert.equal(writes(r).length, 3, 'the cut reply, the nudged write, the closing words');
  assert.ok(writes(r)[1].raw.includes('cut off by the output limit'), 'the second request carries the nudge');
  const drafts = out.filter(f => f.type === 'agent-tool-draft');
  assert.ok(drafts.length >= 2, 'the writing was shown live');
  assert.ok(drafts.some(f => !f.draft.done && f.draft.name === 'write'), 'a writing card opened as soon as the model named the call');
  // The Harness hands the streamed pieces over in batches (a few a second at most), so a short script may show only the first and the last.
  assert.ok(drafts.some(f => f.draft.done && f.draft.lost && f.draft.target === 'big.txt' && f.draft.bytes > 1000), 'and taken down, with its file and size, when it was lost');
});

test(`a model that is cut every time is asked ${MAX_CUT_RETRIES} times, then the turn ends with a plain sentence, never silently`, { skip }, async (t) => {
  const r = await rig(t);
  const out = [];
  await r.bridge.runTurn({ ...r.base('c2', f => out.push(f)), message: 'SCRIPT:cutalways big.txt' });
  assert.equal(writes(r).length, MAX_CUT_RETRIES + 1);
  const notes = out.filter(f => f.type === 'delta' && f.noteReason === 'cut');
  assert.equal(notes.length, MAX_CUT_RETRIES + 1, 'one sentence per cut');
  assert.match(notes.at(-1).text, /that write did not happen\. Say “continue”/);
  const done = out.filter(f => f.type === 'done').at(-1);
  assert.ok(done, 'the turn ended');
  assert.equal(done.finishReason, 'length');
  assert.equal(done.cut.calls[0].target, 'big.txt');
  assert.equal(existsSync(join(r.ws, 'big.txt')), false, 'nothing half-written is left behind');
});
