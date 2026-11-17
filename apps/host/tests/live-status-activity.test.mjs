// 0.2.16 (owner): the status line under the newest message says what the model does now: thinking, waiting for it, writing the answer,
// writing a tool call (its name, or the file it writes) and which tool runs. No playful words rotate any more.
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { withPlainStatus } from '../server/agent/status.mjs';

const built = await build({ stdin: { contents: "export * from './src/lib/live-status'; export { setTranslationLanguage, loadCatalog } from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const L = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
const frame = (phase, code, args, text = '') => ({ seq: 1, phase, code, text, args, elapsedMs: 0, receivedAt: 0 });

test('thinking and waiting say so plainly, and nothing rotates', () => {
  for (const r of [0, 1, 5]) assert.equal(L.statusWords(frame('thinking', 'thinking', undefined, 'Thinking… 120 tokens'), r), 'Thinking…');
  assert.equal(L.statusWords(frame('waiting', 'model-working')), 'Waiting for the model…');
  assert.equal(L.isPlayful(frame('thinking', 'thinking')), false);
  assert.equal(L.statusWords(frame('answering', 'answering')), 'Writing the answer…');
  assert.equal(L.statusWords(frame('waiting', 'reading-prompt')), 'Reading your message…');
});

test('a tool call being written names the tool, or the file it writes', () => {
  assert.equal(L.statusWords(frame('preparing', 'preparing-tool', { tool: 'read', verb: 'Reading a file', target: '' })), 'Calling read…');
  assert.equal(L.statusWords(frame('preparing', 'preparing-tool', { tool: 'write', verb: 'Writing a file', target: 'C:\\work\\src\\app.js' })), 'Writing file app.js…');
  assert.equal(L.statusWords(frame('preparing', 'preparing-tool', { tool: 'edit', verb: 'Changing a file', target: 'src/app.js' })), 'Writing file app.js…');
  assert.equal(L.statusWords(frame('preparing', 'preparing-tool', { tool: 'write', verb: 'Writing a file', target: '' })), 'Calling write…');
  assert.equal(L.statusWords(frame('preparing', 'preparing-tool', { tool: 'mcp__my_server__lookup', verb: 'Using a tool', target: '' })), 'Calling lookup…');
});

test('a running tool: commands and tools without words of their own say the tool\'s name; the others keep their sentence', () => {
  assert.equal(L.statusWords(frame('tool', 'tool', { tool: 'pwsh', verb: 'Running a command', target: 'Get-ChildItem' })), 'Running pwsh…');
  assert.equal(L.statusWords(frame('tool', 'tool', { tool: 'bash', verb: 'Running a command', target: 'ls' })), 'Running bash…');
  assert.equal(L.statusWords(frame('tool', 'tool', { tool: 'mcp__other__thing', verb: 'Using a tool', target: 'mcp__other__thing' })), 'Running thing…');
  assert.equal(L.statusWords(frame('tool', 'tool', { tool: 'write', verb: 'Writing a file', target: 'C:\\work\\plan.txt' })), 'Writing file plan.txt…', 'a file tool that runs names the file, not its path');
  assert.equal(L.statusWords(frame('tool', 'tool', { tool: 'read', verb: 'Reading a file', target: 'notes.md' })), 'Reading a file · notes.md…');
  assert.equal(L.statusWords(frame('tool', 'tool', { tool: 'read' }, 'Running read…')), 'Running read…', 'a tool its server runs has no verb: its name, translated');
});

test('the plain chat path names a tool its server runs, then waits for the model again', () => {
  const out = [];
  let t = 0;
  const f = withPlainStatus(x => out.push(x), { now: () => t, gapMs: 0, beatMs: 0 });
  t = 10; f({ type: 'start' });
  t = 20; f({ type: 'native-tool', event: { event: 'call', id: 'c1', name: 'mcp__files__read' } });
  t = 30; f({ type: 'native-tool', event: { event: 'result', id: 'c1', ok: true } });
  t = 40; f({ type: 'delta', text: 'Hi' });
  t = 50; f({ type: 'done' });
  const status = out.filter(x => x.type === 'status').map(x => [x.phase, x.code, x.text, x.key ?? null]);
  assert.deepEqual(status, [
    ['starting', 'starting', 'Starting…', null], ['waiting', 'reading-prompt', 'Reading your message…', null],
    ['tool', 'tool', 'Running read…', 'Running {0}…'], ['waiting', 'model-working', 'Waiting for the model…', null],
    ['answering', 'answering', 'Writing the answer…', null], ['idle', 'idle', '', null]]);
  assert.equal(out.filter(x => x.type !== 'status').length, 5, 'every frame of the stream still passes');
  f.dispose();
});
