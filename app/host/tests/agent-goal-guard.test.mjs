// #4981: a goal ran 5 h 23 min with 26 failed tool calls and nothing stopped it: goal turns had no guard at all. A goal now has its own guard:
// twelve failed calls in a row, or the same call (tool and arguments) failing four times, pauses it with one sentence. The per-turn limits
// (25 searches, plan rewrites) never apply to a goal, and a call the person denied is not a failure.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { GoalGuard, GUARD_LIMITS, GUARD_NOTES, guardNote } from '../server/agent/turn-guard.mjs';

let n = 0;
const call = (name, args, state) => ({ id: `c${++n}`, name, args, state });

test('twelve failed calls in a row pause the goal; a success in between starts the count again', () => {
  const g = new GoalGuard();
  for (let i = 0; i < 11; i++) assert.equal(g.card(call('read', { path: `f${i}.txt` }, 'error')), null, `failure ${i + 1}`);
  assert.equal(g.card(call('read', { path: 'ok.txt' }, 'done')), null);
  for (let i = 0; i < 11; i++) assert.equal(g.card(call('read', { path: `g${i}.txt` }, 'error')), null);
  assert.equal(g.card(call('read', { path: 'g11.txt' }, 'error')), 'goal-failures');
  assert.equal(GUARD_LIMITS.goalFailedInARow, 12);
});

test('the same call failing four times pauses it, even with other work in between', () => {
  const g = new GoalGuard();
  const same = { command: 'npm test -- --grep slow' };
  for (let i = 0; i < 3; i++) {
    assert.equal(g.card(call('bash', same, 'error')), null);
    assert.equal(g.card(call('read', { path: 'a.txt' }, 'done')), null);
  }
  assert.equal(g.card(call('bash', { command: 'npm test -- --grep fast' }, 'error')), null, 'other arguments are another call');
  assert.equal(g.card(call('bash', same, 'error')), 'goal-failures');
});

test('a denied call, a running card and the same card seen twice are not counted', () => {
  const g = new GoalGuard();
  for (let i = 0; i < 20; i++) assert.equal(g.card(call('write', { path: 'x' }, 'denied')), null);
  const running = call('bash', { command: 'make' }, 'running');
  assert.equal(g.card(running), null);
  const failed = call('bash', { command: 'make' }, 'error');
  for (let i = 0; i < 10; i++) assert.equal(g.card(failed), null, 'one card is one failure');
  assert.equal(g.failedTotal, 1);
});

test('a goal is never stopped by per-turn search and rewrite limits', () => {
  const g = new GoalGuard();
  for (let i = 0; i < 100; i++) assert.equal(g.card(call('web_search', { query: `q${i}` }, 'done')), null);
  assert.equal(g.text('1. step\n'.repeat(50)), null);
  assert.equal(g.silent({ ended: { kind: 'completed' }, wrote: false }), null);
});

test('the sentence is the approved one, with its interface key', () => {
  assert.equal(GUARD_NOTES['goal-failures'], 'This goal kept running into the same errors, so I paused it. Look at the last steps, fix what is in the way, then resume the goal.');
  assert.equal(guardNote('goal-failures').key, GUARD_NOTES['goal-failures']);
});

test('the bridge gives a goal turn the goal guard; a trip stops the turn, and stop() pauses the goal', () => {
  const src = readFileSync(new URL('../server/agent/agent-bridge.mjs', import.meta.url), 'utf8');
  assert.match(src, /guard = goal \? new GoalGuard\(\) : new TurnGuard\(\{ plan: !!planTask \}\)/);
  assert.match(src, /function trip\(reason\) \{[\s\S]{0,400}void stop\(chatId\)/);
  const stop = src.slice(src.indexOf('async function stop(chatId'), src.indexOf('function turnOf(chatId'));
  assert.match(stop, /await beforeAbort\(\(\) => pauseGoal\(runtime, turn\.sessionId, chatId, pauseSignal\), pauseSignal\)/);
});
