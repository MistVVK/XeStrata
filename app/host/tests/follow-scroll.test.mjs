import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/follow-scroll.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const f = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('a wheel notch up while an answer streams lets go at once (#4014)', () => {
  assert.equal(f.wheelLeavesBottom(-100), true);
  assert.equal(f.wheelLeavesBottom(100), false);
  // the notch lands about 100 px from the bottom, inside the old 120 px band: it must not be pulled back
  assert.equal(f.followAfterScroll(100, true, 20), false);
  assert.equal(f.followAfterScroll(40, true, 20), false);
});

test('back at the bottom the chat follows again, by wheel, key, drag or the jump button', () => {
  assert.equal(f.followAfterScroll(0, false, 10_000), true);
  assert.equal(f.followAfterScroll(f.FOLLOW_SLACK, false, 10_000), true);
  assert.equal(f.followAfterScroll(f.FOLLOW_SLACK + 1, false, 10_000), false);
});

test('the page growing under a following reader does not make the chat let go', () => {
  // no input from the reader for a while, a few lines of the answer arrived: keep following
  assert.equal(f.followAfterScroll(80, true, 5_000), true);
  // a reader who had let go stays away while the answer grows
  assert.equal(f.followAfterScroll(80, false, 5_000), false);
  // far away whatever moved it (a scrollbar drag or a jump)
  assert.equal(f.followAfterScroll(f.PILL_DISTANCE + 1, true, 5_000), false);
});

test('the chat screen uses the reader-driven rule, not the 120 px band', async () => {
  const chat = await readFile(new URL('../src/screens/Chat.tsx', import.meta.url), 'utf8');
  assert.doesNotMatch(chat, /clientHeight < 120/);
  assert.match(chat, /followAfterScroll\(/);
  assert.match(chat, /onWheel=/);
  // sending a message still brings the reader to it, even from far up
  assert.match(chat, /pinned\.current \|\| sent/);
  // an upward wheel on a transcript that cannot scroll must not switch following off
  assert.match(chat, /wheelLeavesBottom\(e\.deltaY\) && e\.currentTarget\.scrollTop > 0/);
});
