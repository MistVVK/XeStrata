// Packaged-app walkthrough (GUI-VIS-1117 A70): "Run now" stays disabled on the Tasks list with an API model picked and no local model. That is right: a task run on the host uses
// the model loaded on this computer (server/index.mjs runTurn / createAgentTaskRunner take an API model only from the developer variable STRATA_TASK_REMOTE_MODEL), but the
// button said nothing. A plain line now says why and what to do.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const tasks = readFileSync(new URL('../src/screens/Tasks.tsx', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const LINE = 'Tasks run on a model loaded on this computer. Load one in Models.';

test('the line shows when there are tasks, no model runs on this computer, and no editor is open', () => {
  const at = tasks.split('\n').find(l => l.includes(LINE)) ?? '';
  assert.ok(at.includes('!canRun && tasks.length && !editor'), at);
  assert.ok(at.includes('role="status"'));
  assert.ok(tasks.includes("const canRun = !!loaded || !!native?.instances.some(i => i.status === 'ready');"), 'canRun is still the local-model question (a task cannot run on an API model)');
});

test('the line exists in all ten languages', () => {
  const en = JSON.parse(readFileSync(new URL('../src/locales/guic-en.json', import.meta.url), 'utf8'));
  const key = Object.keys(en).find(k => en[k] === LINE);
  assert.ok(key, 'the English key');
  for (const l of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he']) {
    const text = JSON.parse(readFileSync(new URL(`../src/locales/guic-${l}.json`, import.meta.url), 'utf8'))[key];
    assert.ok(text && text !== LINE, l);
  }
});
