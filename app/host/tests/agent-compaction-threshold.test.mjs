// The compaction threshold knob: the default is the Harness's own 0.8, and STRATA_COMPACTION_THRESHOLD_RATIO moves it
// (on a large window the fraction is what decides, so 0.8 compacts a 256K chat far earlier than the window needs).
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildProfile, compactionHeadroom, compactionThreshold, headroomFraction, localOutputBudget, thresholdRatio } from '../server/agent/profile.mjs';

test('the default threshold is the Harness fraction 0.8 and the headroom fraction 0.125', () => {
  assert.equal(thresholdRatio({}), 0.8);
  assert.equal(headroomFraction({}), 0.125);
  // The numbers the 2026-10-22 test recorded: Gyro-S at 64K starts at 40,142 real tokens.
  const out = localOutputBudget(65536);
  assert.equal(out, 19660);
  assert.equal(compactionHeadroom(65536, out), 5734);
  assert.equal(compactionThreshold(65536, out, {}), 40142);
});

test('on a large window the headroom caps the start point, and the two ratios move it up', () => {
  const out = localOutputBudget(262144);                     // 32,768: the reply's room
  const cap = 262144 - out - compactionHeadroom(262144, out); // 200,704: below 0.8 of the window, so it decides
  assert.ok(cap < 262144 * 0.8, 'the cap is what binds at 0.8 on this window');
  assert.equal(compactionThreshold(262144, out, {}), cap);
  // Raising the fraction lifts the threshold to the cap of the new headroom; shrinking the headroom lifts the cap itself.
  assert.equal(compactionThreshold(262144, out, { STRATA_COMPACTION_THRESHOLD_RATIO: '0.95' }), cap);
  const lifted = compactionThreshold(262144, out, { STRATA_COMPACTION_THRESHOLD_RATIO: '0.95', STRATA_COMPACTION_HEADROOM_FRACTION: '0.0625' });
  assert.equal(lifted, 262144 - out - compactionHeadroom(262144, out, { STRATA_COMPACTION_HEADROOM_FRACTION: '0.0625' }));
  assert.ok(lifted > cap, `a smaller headroom starts compaction later (${lifted} > ${cap})`);
  assert.ok(lifted + out <= 262144, 'a prompt at the threshold plus a full reply still fits');
  // On a window where the fraction does bind, the ratio alone moves the start point.
  const w = 1_000_000, out1m = localOutputBudget(w);
  assert.equal(compactionThreshold(w, out1m, {}), Math.floor(w * 0.8));
  assert.equal(compactionThreshold(w, out1m, { STRATA_COMPACTION_THRESHOLD_RATIO: '0.9' }), Math.floor(w * 0.9));
});

test('a ratio outside 0.1..0.99 or not a number keeps the default', () => {
  for (const bad of ['0', '0.05', '1', '1.5', '-0.8', 'abc', '']) assert.equal(thresholdRatio({ STRATA_COMPACTION_THRESHOLD_RATIO: bad }), 0.8, bad);
  for (const bad of ['0.6', '-1', 'x']) assert.equal(headroomFraction({ STRATA_COMPACTION_HEADROOM_FRACTION: bad }), 0.125, bad);
});

test('the generated profile carries the ratio to the Harness per model', () => {
  const before = process.env.STRATA_COMPACTION_THRESHOLD_RATIO;
  process.env.STRATA_COMPACTION_THRESHOLD_RATIO = '0.95';
  try {
    const profile = buildProfile({ local: [{ id: 'a', modelId: 'm', baseUrl: 'http://x', context: 262144 }], connections: [] });
    assert.match(profile.patch, /"provider": "strata-local",\s*"model": "m",\s*"headroomTokens": \d+,\s*"thresholdRatio": 0\.95/);
  } finally {
    if (before === undefined) delete process.env.STRATA_COMPACTION_THRESHOLD_RATIO; else process.env.STRATA_COMPACTION_THRESHOLD_RATIO = before;
  }
});
