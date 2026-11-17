import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/clipboard.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { copyText } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const toasts = () => { const list = []; return { list, toast: (...a) => list.push(a) }; };

test('Copy says Copied when the clipboard takes the text (#3731)', async () => {
  const t = toasts(); let got = '';
  assert.equal(await copyText('hello', 'Response', { clipboard: { writeText: async v => { got = v; } }, toast: t.toast }), true);
  assert.equal(got, 'hello'); assert.equal(t.list[0][0], 'success');
});
test('a refused clipboard falls back to the older copy command', async () => {
  const t = toasts(); let got = '';
  assert.equal(await copyText('x', 'Response', { clipboard: { writeText: async () => { throw new Error('denied'); } }, legacyCopy: v => { got = v; return true; }, toast: t.toast }), true);
  assert.equal(got, 'x'); assert.equal(t.list[0][0], 'success');
});
test('when nothing works the person is told, never left with a silent click', async () => {
  const t = toasts();
  assert.equal(await copyText('x', 'Response', { clipboard: null, legacyCopy: () => false, toast: t.toast }), false);
  assert.equal(t.list.length, 1); assert.equal(t.list[0][0], 'warn'); assert.match(t.list[0][2], /Ctrl\+C/);
});
test('the typed message keeps its line breaks (#3720) and the chat uses the shared copy', async () => {
  const css = await readFile(new URL('../src/styles/app.css', import.meta.url), 'utf8');
  assert.match(css, /\.st-bubble__text\s*\{[^}]*white-space:\s*pre-wrap/);
  const chat = await readFile(new URL('../src/screens/Chat.tsx', import.meta.url), 'utf8');
  assert.doesNotMatch(chat, /void navigator\.clipboard\.writeText/);
});

test('a copy that worked shows no corner popover: the control shows it; a failed copy still has its toast', async () => {
  const { readFile } = await import('node:fs/promises');
  const hook = await readFile(new URL('../src/lib/use-copy.ts', import.meta.url), 'utf8');
  assert.match(hook, /toast: \(tone, title, body\) => \{ if \(tone !== 'success'\) pushToast\(tone, title, body\); \}/);
  assert.match(hook, /pushToast\('warn', 'Couldn’t copy'\)/);
});
