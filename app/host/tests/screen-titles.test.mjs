import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

const root = resolve(import.meta.dirname, '..');
const read = (p) => readFileSync(resolve(root, p), 'utf8');
const liquid = read('src/styles/liquid.css');

test('every screen header title uses the chat title step, from one rule', () => {
  const rule = liquid.match(/\.chat-toolbar__conversation h1,\s*\.main__head \.main__title \{([^}]*)\}/);
  assert.ok(rule, 'the chat title and the screen titles share one rule');
  assert.match(rule[1], /font-size:\s*var\(--st-fs-md\)/);
  assert.match(liquid, /\.main__head \.main__title \{ font-weight: 600; \}/);
  const imports = [...read('src/main.tsx').matchAll(/import '\.\/styles\/([^']+)'/g)].map(m => m[1]);
  assert.equal(imports.at(-1), 'liquid.css', 'liquid.css loads last, so no sheet after it can resize a title');
});

test('the chat title and the screen titles are the same size', () => {
  const app = read('src/styles/app.css');
  const chat = app.match(/\.chat-toolbar__conversation h1 \{[^}]*font-size:\s*var\(--st-fs-sm\)/);
  assert.ok(chat, 'app.css still sets the chat title; liquid.css raises it to --st-fs-md for chat and screens together');
});

test('the Tasks header has no subtitle under it', () => {
  const tasks = read('src/screens/Tasks.tsx');
  assert.ok(!/Everything that is running, waiting for you/.test(tasks));
  assert.ok(!/<Screen[^\n]*\bsub=/.test(tasks));
});
