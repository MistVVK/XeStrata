/** #6068: the app font's files class the spacing accents (` ¨ ¯ ´ ¸ ˆ ˇ ˘ ˙ ˚ ˜ ˝) as combining marks, so a browser draws them
 *  with no width and "```" typed in the message box collapses into one mark. Every face of that font leaves those characters
 *  to the next font in the stack, and still covers everything else it covered. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const css = await readFile(new URL('../src/styles/kit/tokens.css', import.meta.url), 'utf8');
const faces = [...css.matchAll(/@font-face\{([^}]*)\}/g)].map(m => m[1]).filter(b => /font-family:"GT Walsheim Pro"/.test(b));
const ZERO_WIDTH_IN_FONT = [0x60, 0xa8, 0xaf, 0xb4, 0xb8, 0x2c6, 0x2c7, 0x2d8, 0x2d9, 0x2da, 0x2dc, 0x2dd];

function ranges(block) {
  const m = /unicode-range:([^;]+)/.exec(block);
  if (!m) return [[0, 0x10ffff]];
  return m[1].split(',').map(r => r.trim().replace(/^U\+/i, '').split('-').map(h => parseInt(h, 16))).map(([a, b]) => [a, b ?? a]);
}
const covers = (rs, cp) => rs.some(([a, b]) => cp >= a && cp <= b);

test('the app font has its four faces', () => assert.equal(faces.length, 4));

test('no face of the app font draws a spacing accent it would draw with no width', () => {
  for (const f of faces) for (const cp of ZERO_WIDTH_IN_FONT) assert.ok(!covers(ranges(f), cp), `U+${cp.toString(16)} in ${f.slice(0, 80)}`);
});

test('every other character the font covered is still its own', () => {
  for (const f of faces) {
    const rs = ranges(f);
    for (let cp = 0; cp < 0x2ff; cp++) if (!ZERO_WIDTH_IN_FONT.includes(cp)) assert.ok(covers(rs, cp), `U+${cp.toString(16)} lost`);
    for (const cp of [0x300, 0x301, 0x2014, 0x20ac, 0x4e00, 0x10ffff]) assert.ok(covers(rs, cp), `U+${cp.toString(16)} lost`);
  }
});
