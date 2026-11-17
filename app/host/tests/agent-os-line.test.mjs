/**
 * 0.2.10: the agent's system prompt says which computer it is on, so a model on Windows does not spend its first steps on
 * `ls`, `uname` and `wc`. Pure test of the preset; the check with a real model on Windows needs the owner's OK.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { AGENT_PROMPT, agentPromptFor, standardPreset } from '../server/agent/preset.mjs';

const promptOf = platform => standardPreset({ platform }).plugins.find(p => p.id === 'persona').config.prefix;

test('the Windows prompt names PowerShell and the commands that do not exist', () => {
  const text = promptOf('win32');
  assert.match(text, /runs Windows and the shell tool is PowerShell/);
  assert.match(text, /uname, wc and grep do not exist/);
  assert.ok(text.length < 1900, `${text.length} chars`);
  assert.ok(text.startsWith(AGENT_PROMPT.split('\n\n')[0]), 'the first line stays first');
});

test('macOS and Linux say so and do not mention PowerShell as the shell', () => {
  assert.match(promptOf('darwin'), /runs macOS/);
  assert.match(promptOf('linux'), /runs Linux/);
  assert.doesNotMatch(promptOf('linux').split('\n\n')[1], /PowerShell/);
});

test('an unknown system keeps the plain prompt, and every other rule of it is still there', () => {
  assert.equal(agentPromptFor('freebsd'), AGENT_PROMPT);
  for (const p of ['win32', 'darwin', 'linux']) {
    for (const rule of AGENT_PROMPT.split('\n\n').slice(1)) assert.ok(promptOf(p).includes(rule), `${p}: ${rule.slice(0, 40)}`);
  }
});
