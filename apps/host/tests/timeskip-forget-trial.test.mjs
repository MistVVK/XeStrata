// #4611: Time Skip trials were never forgotten, and a chat's Stop stopped working. A trial that ends now leaves the host's maps (PR #4614),
// and the chat's pointer goes only while it still points at that trial: a chat that started a second trial keeps Stop on the running one.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { forgetTrial } from '../server/timeskip.mjs';

const maps = () => ({ trials: new Map(), trialByChat: new Map(), choices: new Map() });
const start = (m, id, chatId) => { m.trials.set(id, { chatId }); if (chatId) m.trialByChat.set(chatId, id); m.choices.set(id, { modelId: 'x' }); };

test('a trial that ended leaves every map', () => {
  const m = maps();
  for (let i = 0; i < 50; i++) { start(m, `r${i}`, `c${i}`); forgetTrial(m, `r${i}`); }
  assert.equal(m.trials.size + m.trialByChat.size + m.choices.size, 0);
});

test('the first trial ending keeps the chat pointing at its second, running trial', () => {
  const m = maps();
  start(m, 'r1', 'chat'); start(m, 'r2', 'chat');
  forgetTrial(m, 'r1');
  assert.equal(m.trialByChat.get('chat'), 'r2', 'Stop in this chat still reaches the running trial');
  assert.ok(m.trials.has('r2'));
  forgetTrial(m, 'r2');
  assert.equal(m.trialByChat.has('chat'), false);
});

test('an unknown id, or a trial with no chat, is harmless', () => {
  const m = maps();
  start(m, 'r1', 'chat'); start(m, 'r2', null);
  forgetTrial(m, 'nope'); forgetTrial(m, 'r2');
  assert.equal(m.trialByChat.get('chat'), 'r1');
  assert.equal(m.trials.size, 1);
});

test('the host drops trials through forgetTrial', () => {
  const src = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.match(src, /function dropTrial\(id, timer\) \{\s*if \(timer\) clearInterval\(timer\);\s*forgetTrial\(\{ trials, trialByChat, choices: timeskipChoices \}, id\);\s*\}/);
  assert.doesNotMatch(src, /trialByChat\.delete\(/, 'no other path removes a chat pointer unconditionally');
});
