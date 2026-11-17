// Visual walkthrough (GUI-VIS-1117, finding 7a): "Run now" was disabled on the task list row while a model was ready, and enabled inside the task:
// the row asked only the older runtime (library.loaded), the task page asked the native app's instances too (#5383).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const tasks = readFileSync(new URL('../src/screens/Tasks.tsx', import.meta.url), 'utf8').replace(/\r\n/g, '\n');

test('the list row and the task page decide "Run now" by the same canRun, and a running task can always be stopped', () => {
  assert.ok(/const canRun = !!loaded \|\| !!native\?\.instances\.some\(i => i\.status === 'ready'\);/.test(tasks));
  const row = tasks.match(/<article key=\{t\.id\} className="task-row">[\s\S]*?<\/article>/)?.[0] ?? '';
  assert.ok(row.includes("disabled={pending || !canRun && t.status !== 'running'}"), 'the row button');
  assert.ok(!/disabled=\{pending \|\| !loaded\}/.test(row), 'the row no longer asks only the older runtime');
  assert.ok(tasks.includes("disabled={pending || !canRun && currentTask.status !== 'running'}"), 'the task page button');
});
