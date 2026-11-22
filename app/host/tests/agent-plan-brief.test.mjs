import test from 'node:test';
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { readRequirement, requirementOf, StepEvidence } from '../server/agent/plan-brief.mjs';
import { editedSteps, parsePlan, parsePoints, stepObjective } from '../server/agent/plan.mjs';

test('the compact brief retains the current instruction, latest result and exact check', () => {
  const old = Array.from({ length: 8 }, (_, i) => ({ text: 'Earlier work item ' + i, summary: 'Recorded result ' + i, state: 'done' }));
  const current = { text: 'Inspect the current decision [verify-read old.txt | OLD]', correction: 'Use the blue decision.', verification: { path: 'blue.txt', line: 'BLUE' } };
  const plan = { title: 'Find the current decision', steps: [...old, current] };
  const brief = stepObjective(plan, 8, true);
  assert.match(brief, /8 earlier steps completed/);
  assert.match(brief, /Recorded result 7/);
  assert.match(brief, /Use the blue decision/);
  assert.match(brief, /blue\.txt.*"BLUE"/);
  assert.doesNotMatch(brief, /Earlier work item 0|old\.txt/);
  assert.match(stepObjective(plan, 8), /Earlier work item 0/);
  assert.match(stepObjective({ title: 'Inspect', steps: [{ text: 'Open the page in the browser' }] }, 0, true), /headless browser/);
});

test('read verification survives editing and numbering without weakening invalid rules', () => {
  const input = 'Inspect ' + 'long introduction '.repeat(12) + '[verify-read notes/current.txt | 2. exact  text]';
  const expected = { path: 'notes/current.txt', line: '2. exact  text' };
  assert.deepEqual(requirementOf(editedSteps([input], [], true)[0]), expected);
  assert.deepEqual(requirementOf(parsePlan('## Read\n1. ' + input + '\n2. Report result', '', true).steps[0]), expected);
  assert.deepEqual(requirementOf(parsePoints('Inspection: 1. ' + input + ' 2. Report result', true).steps[0]), expected);
  assert.deepEqual(readRequirement('[verify-read ../other.txt | line]'), { invalid: true });
  assert.deepEqual(readRequirement('[verify-read C:/temp/a | line]'), { invalid: true });
  assert.deepEqual(readRequirement('[verify-read target.txt'), { invalid: true });
  assert.equal(readRequirement('Ordinary step'), null);
});

test('only a matching read for the active verification can prove completion', () => {
  const cwd = resolve('fixture');
  const proof = new StepEvidence(cwd, { path: 'red.txt', line: 'red' });
  const call = (seq, name, id, file) => proof.observe({ type: 'tool/call', seq, data: { name, callId: id, arguments: { file_path: file } } });
  const result = (seq, callId, file, line, isError = false) => proof.observe({ type: 'tool/result', seq, time: 9001, data: { message: { toolCallId: callId, isError }, meta: { path: resolve(cwd, file), lines: [{ number: 2, text: line }] } } });
  call(1, 'read', 'old', 'red.txt');
  proof.correct('Switch [verify-read blue.txt | blue]');
  result(2, 'old', 'red.txt', 'red');
  assert.equal(proof.receipt, null);
  call(3, 'read', 'mismatch', 'blue.txt');
  result(4, 'mismatch', 'blue.txt', 'blueish');
  assert.equal(proof.receipt, null);
  call(5, 'read', 'failed', 'blue.txt');
  result(6, 'failed', 'blue.txt', 'blue', true);
  assert.equal(proof.receipt, null);
  call(7, 'read', 'first', 'blue.txt');
  call(8, 'read', 'second', 'blue.txt');
  result(9, 'first', 'blue.txt', 'blue');
  assert.equal(proof.receipt, null);
  result(10, 'second', 'blue.txt', 'blue');
  assert.deepEqual(proof.receipt, { path: 'blue.txt', line: 2, callId: 'second', seq: 10, at: 9001 });
  proof.correct('How far along are you?');
  assert.equal(proof.receipt.seq, 10);
  call(11, 'edit', 'change', 'blue.txt');
  assert.equal(proof.receipt, null);
});

test('a plain file switch clears the previous receipt and needs a fresh Read of the new file', () => {
  const cwd = resolve('fixture');
  const proof = new StepEvidence(cwd, { path: 'original.txt', line: 'original' });
  const call = (seq, id, path) => proof.observe({ type: 'tool/call', seq, data: { name: 'read', callId: id, arguments: { file_path: path } } });
  const result = (seq, toolCallId, path, text) => proof.observe({ type: 'tool/result', seq, data: { message: { toolCallId, isError: false }, meta: { path: resolve(cwd, path), lines: [{ number: 1, text }] } } });
  call(1, 'old', 'original.txt');
  result(2, 'old', 'original.txt', 'original');
  assert.equal(proof.receipt.path, 'original.txt');
  proof.correct('Do not switch to unrelated.txt.');
  assert.equal(proof.receipt.path, 'original.txt');

  proof.correct('Nimm `notes/new decision.txt` statt `original.txt`.');
  assert.deepEqual(proof.requirement, { path: 'notes/new decision.txt' });
  assert.equal(proof.receipt, null);
  call(3, 'stale', 'original.txt');
  result(4, 'stale', 'original.txt', 'original');
  assert.equal(proof.receipt, null);
  call(5, 'fresh', 'notes/new decision.txt');
  result(6, 'fresh', 'notes/new decision.txt', 'revised');
  assert.equal(proof.receipt.seq, 6);
  assert.equal(proof.receipt.line, undefined);

  proof.correct('Switch to final.txt.');
  assert.deepEqual(proof.requirement, { path: 'final.txt' });
  assert.equal(proof.receipt, null);
  const brief = stepObjective({ title: 'Decision', steps: [{
    text: 'Verify', verification: proof.requirement, verificationDirection: 'Switch to final.txt.', correction: 'Carry on.',
  }] }, 0, true);
  assert.match(brief, /Switch to final\.txt\./);
  assert.match(brief, /Carry on\./);
  assert.match(brief, /read on "final\.txt" to verify its current contents/);
});
