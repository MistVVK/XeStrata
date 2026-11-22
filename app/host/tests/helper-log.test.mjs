// #5254: a helper's row could only show one changing sentence, and "View log" opened only for a failed helper, with nothing but that sentence in it.
// The row now keeps what the helper did (a short, bounded list) and "View log" shows it for every helper that has steps.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { transform } from 'esbuild';
import { ChildProjector, MAX_STEPS } from '../server/agent/children.mjs';
import { createHelperRecords } from '../server/chat-helper-records.mjs';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const load = async name => {
  const { code } = await transform(src(name), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
};
const ev = (type, data, time = 1000) => ({ type: 'event', event: { seq: 1, time, type, data } });
const say = text => ev('assistant/message', { message: { content: [{ type: 'text', text }] } });
const call = (callId, name, args) => ev('tool/call', { callId, name, arguments: JSON.stringify(args) });
const result = callId => ev('tool/result', { message: { toolCallId: callId, isError: false, content: [{ type: 'text', text: 'ok' }] } });

test('the row keeps what the helper did: its tools with their target and its own words, oldest first', () => {
  const cp = new ChildProjector({ id: 'c1', name: 'Check the tests', task: 'Run them' });
  assert.equal(cp.agent().steps, undefined, 'nothing yet: no steps field');
  cp.consume(call('a', 'read', { file_path: 'src/app.js' }));
  cp.consume(result('a'));
  cp.consume(call('b', 'bash', { command: 'npm test' }));
  cp.consume(say('Two tests fail in app.js.'));
  const steps = cp.agent().steps;
  assert.equal(steps.length, 3);
  assert.equal(typeof steps[0].verb, 'string'); assert.ok(steps[0].verb.length > 0);
  assert.match(steps[0].target, /src\/app\.js/);
  assert.match(steps[1].target, /npm test/);
  assert.deepEqual(steps[2], { said: 'Two tests fail in app.js.' });
});

test('the list is bounded and each entry is short', () => {
  const cp = new ChildProjector({ id: 'c2', name: 'Many' });
  for (let i = 0; i < 50; i++) cp.consume(call(`t${i}`, 'read', { file_path: `file${i}-${'x'.repeat(400)}.txt` }));
  cp.consume(say('x'.repeat(5000)));
  const steps = cp.agent().steps;
  assert.equal(MAX_STEPS, 20);
  assert.equal(steps.length, MAX_STEPS);
  assert.ok(steps.every(s => (s.said ?? '').length <= 400 && (s.target ?? '').length <= 160 && (s.verb ?? '').length <= 40));
  assert.equal(steps.at(-1).said.length <= 400, true);
  assert.match(steps.at(-2).target, /file49/ , 'the newest tool step is kept');
});

test('two helpers keep their own steps (nothing of one reaches the other)', () => {
  const a = new ChildProjector({ id: 'a', name: 'A' }), b = new ChildProjector({ id: 'b', name: 'B' });
  a.consume(call('1', 'read', { file_path: 'only-a.txt' }));
  b.consume(call('1', 'read', { file_path: 'only-b.txt' }));
  assert.match(a.agent().steps[0].target, /only-a/); assert.equal(a.agent().steps.length, 1);
  assert.match(b.agent().steps[0].target, /only-b/); assert.equal(b.agent().steps.length, 1);
});

test('the page merges the steps of a frame, keeps them when a later frame has none, and drops malformed ones', async () => {
  const { updateAgent } = await load('lib/agents.ts');
  const row = { id: 'h', name: 'H', status: 'working', description: '', durationMs: 0, tokens: 0 };
  let agents = updateAgent([], { ...row, steps: [{ verb: 'Reading', target: 'a.txt' }, { said: 'done' }, 'junk', { verb: 5 }, null] }, 1);
  assert.deepEqual(agents[0].steps, [{ verb: 'Reading', target: 'a.txt' }, { said: 'done' }]);
  agents = updateAgent(agents, { id: 'h', status: 'completed', tokens: 9, updatedAt: 5 }, 5);
  assert.deepEqual(agents[0].steps, [{ verb: 'Reading', target: 'a.txt' }, { said: 'done' }], 'a frame without steps keeps them');
  const many = Array.from({ length: 60 }, (_, i) => ({ verb: 'Reading', target: `f${i}` }));
  agents = updateAgent(agents, { id: 'h', steps: many, updatedAt: 9 }, 9);
  assert.equal(agents[0].steps.length, 20);
  assert.equal(agents[0].steps.at(-1).target, 'f59');
});

test('View log opens for any helper that has steps, and lists them', () => {
  const row = src('components/SubagentRow.tsx');
  assert.match(row, /a\.status === 'failed' \|\| a\.steps\?\.length \|\| messages\.length\) \? <Btn size="sm" onClick=\{showLog\}>/);
  assert.match(row, /a\.steps\.map\(\(step, i\) => <li key=\{i\}/);
  assert.doesNotMatch(row, /\{tr\(a\.status === 'failed' \? <button className="text-action"/);
});

test('messages and archive choices survive reopening and are scoped to their chat and helper', t => {
  const root = mkdtempSync(join(tmpdir(), 'strata-helper-records-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const writer = createHelperRecords(root);
  const message = writer.message('c1', 'h1', { messageId: 'm1', delivery: 'steer' }, 'Keep the exact text.\nSecond line.');
  writer.message('c1', 'h1', { messageId: 'm1', delivery: 'steer' }, 'Ignore the duplicate receipt.');
  writer.archive('c1', ['h1', 'h2'], true);
  const reader = createHelperRecords(root);
  assert.deepEqual(reader.read('c1'), { h1: { archived: true, messages: [message] }, h2: { archived: true } });
  assert.deepEqual(reader.read('c2'), {});
  reader.archive('c1', ['h1', 'h2'], false);
  assert.deepEqual(reader.read('c1'), { h1: { messages: [message] } });
  reader.forget('c1');
  assert.deepEqual(reader.read('c1'), {});
});
