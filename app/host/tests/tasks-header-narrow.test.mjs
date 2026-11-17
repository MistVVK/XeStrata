// Visual walkthrough (GUI-VIS-1117, finding 7b): at 420 px the Tasks header controls (Schedule | Board, search, Next run, +) ran off the right edge ("Sche...", "Ne...").
// Measured live in headless Chrome: on the old CSS the controls end at x = 551 of 420 (4 cut); with the rule below they end at 406.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../src/styles/annotation-fixes.css', import.meta.url), 'utf8').replace(/\r\n/g, '\n');

test('on a narrow screen the Tasks header controls take their own row under the title and cannot be wider than the screen', () => {
  const block = css.slice(css.indexOf('@media (max-width: 767px) {\n  .main__head, .main__head.main__head--plain { height: auto; }'));
  assert.ok(block.includes('.main__head:has(.tasks-actions) { flex-wrap: wrap;'), 'the header wraps');
  assert.ok(block.includes('.main__head:has(.tasks-actions) .main__actions { flex: 1 1 100%; min-width: 0; }'), 'the controls take a row and may shrink');
  assert.ok(block.includes('.tasks-actions { min-width: 0; max-width: 100%; justify-content: flex-start; }'), 'the group stays inside the row');
});
