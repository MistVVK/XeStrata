// Visual walkthrough (GUI-VIS-1117, finding 9b): the "Tools for apps on this computer" switch was drawn only in the "Available models" list, which exists with
// two or more models loaded, so with one model loaded (the usual case) the owner's switch (11-16: keep it) was nowhere on the Serving page.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const serving = readFileSync(new URL('../src/screens/Serving.tsx', import.meta.url), 'utf8').replace(/\r\n/g, '\n');

test('one switch builder serves the model list and the overview', () => {
  assert.equal([...serving.matchAll(/mcpToolsSwitch\(serving,/g)].length, 1, 'the switch is built in one place');
  assert.ok(/toolsSwitch\(i\.apiModelId\)/.test(serving), 'the list of several models draws it per model');
});

test('with exactly one model loaded the overview has a row with the switch (and none without a model)', () => {
  const row = serving.match(/\{tr\(!multipleModels && loaded \? <div><dt>\{tr\("Tools for apps on this computer"\)\}<\/dt><dd>\{toolsSwitch\(loaded\.apiModelId, true\)\}<\/dd><\/div> : null\)\}/);
  assert.ok(row, 'the overview row for a single loaded model');
  assert.ok(/aria-label=\{tr\("Tools for apps on this computer"\)\}/.test(serving), 'the bare switch keeps its accessible name');
});
