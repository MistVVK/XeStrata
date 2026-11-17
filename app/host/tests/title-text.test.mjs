// #4533: titles with emoji and other scripts are cut without cutting a character in half, and survive a save and restore unchanged.
// (The crash itself is WebKitGTK drawing a COLRv1 colour font: apps/desktop/src-tauri/src/fontguard.rs. Needs a real WebKitGTK to show.)
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { transform } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const { code } = await transform(readFileSync(resolve(root, 'src/lib/title-text.ts'), 'utf8'), { loader: 'ts', format: 'esm' });
const { shortTitle } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

const SAMPLES = {
  ordinary: '👋 hello there, can you summarise this long report for me please thanks',
  zwj: 'x'.repeat(40) + '👨‍👩‍👧‍👦 family and more text after it',
  flag: 'y'.repeat(43) + '🇯🇵🇩🇪 flags at the cut',
  skin: 'z'.repeat(44) + '👍🏽 thumbs and more',
  keycap: 'k'.repeat(45) + '1️⃣2️⃣3️⃣ keycaps',
  arabic: 'مرحباً 👋 كيف حالك اليوم؟ أريد أن أسأل عن شيء مهم جداً في هذا المشروع الكبير',
  cjk: '日本語のタイトルです。これはとても長い質問で、絵文字 🎌 も入っています。続きがあります。',
  combining: 'é'.repeat(30) + ' accents',
  hebrew: 'שלום עולם, מה שלומך היום? אני רוצה לשאול משהו חשוב מאוד על הפרויקט',
  mixed: 'Hello 世界 🌍 мир שלום مرحبا — a mixed title that is long enough to be cut somewhere',
};

const loneSurrogate = s => /[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/.test(s);

test('a short title is returned as it is', () => {
  assert.equal(shortTitle('hello 👋'), 'hello 👋');
  assert.equal(shortTitle('x'.repeat(48)), 'x'.repeat(48));
});

test('a long title is cut to the width with an ellipsis, and never inside a character', () => {
  for (const [name, text] of Object.entries(SAMPLES)) {
    for (let cut = 1; cut <= 4; cut++) {          // move the sample so the cut falls on each unit of the emoji
      const t = shortTitle('a'.repeat(cut - 1) + text);
      assert.ok(t.length <= 48, `${name}/${cut}: ${t.length}`);
      assert.ok(!loneSurrogate(t), `${name}/${cut}: a lone surrogate in ${JSON.stringify(t)}`);
      assert.ok(t.endsWith('…') || t.length === (('a'.repeat(cut - 1) + text).length));
    }
  }
});

test('a ZWJ sequence, a flag pair, a skin tone and a keycap are kept whole or dropped whole', () => {
  const seg = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
  for (const name of ['zwj', 'flag', 'skin', 'keycap', 'combining']) {
    const t = shortTitle(SAMPLES[name]);
    const body = t.replace(/…$/, '').trimEnd();
    // every grapheme of the result is a grapheme of the original at the same place
    const original = [...seg.segment(SAMPLES[name])].map(p => p.segment);
    const result = [...seg.segment(body)].map(p => p.segment);
    assert.deepEqual(result, original.slice(0, result.length), name);
  }
});

test('titles survive being saved and restored as JSON, byte for byte', () => {
  for (const text of Object.values(SAMPLES)) {
    const title = shortTitle(text);
    const restored = JSON.parse(JSON.stringify({ title })).title;
    assert.equal(restored, title);
    assert.equal(Buffer.from(restored, 'utf8').toString('utf8'), restored, 'valid UTF-8 both ways');
  }
});

test('without Intl.Segmenter it still cuts at a code point', () => {
  const real = Intl.Segmenter;
  Intl.Segmenter = undefined;
  try {
    for (const text of Object.values(SAMPLES)) {
      const t = shortTitle(text);
      assert.ok(t.length <= 48 && !loneSurrogate(t), JSON.stringify(t));
    }
  }
  finally { Intl.Segmenter = real; }
});

test('the old cut is the bug: slice(0, 47) leaves half an emoji', () => {
  const prompt = 'a'.repeat(46) + '👋 and more';
  const old = prompt.slice(0, 47).trimEnd() + '…';
  assert.ok(loneSurrogate(old));
  assert.ok(!loneSurrogate(shortTitle(prompt)));
});
