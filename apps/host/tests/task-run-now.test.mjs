// #5383: on Tasks -> Schedule every saved task's "Run now" was greyed out for good: it asked whether the older runtime's model library had a
// model loaded (library.loaded), which the native app never sets. It now asks the same as the board and the chat: a ready native model, or
// the older runtime's loaded one.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const src = readFileSync(fileURLToPath(new URL('../src/screens/Tasks.tsx', import.meta.url)), 'utf8');

test('Run now is enabled by a ready native model as well as by the older runtime', () => {
  assert.match(src, /const canRun = !!loaded \|\| !!native\?\.instances\.some\(i => i\.status === 'ready'\);/);
  assert.match(src, /disabled=\{pending \|\| !canRun && currentTask\.status !== 'running'\}/);
  assert.doesNotMatch(src, /disabled=\{pending \|\| !loaded && currentTask\.status/, 'never the older runtime alone');
});

test('the board and the task header ask the same question', () => {
  assert.match(src, /<TaskBoard query=\{query\} canRun=\{canRun\}/);
});
