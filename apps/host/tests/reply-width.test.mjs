// #5066: a Reply width setting (Comfortable, Wide, Full width) in Settings > Reading & focus, default unchanged, through the one --st-measure token.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const load = async name => {
  const { code } = await transform(src(name), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
};
const LANGS = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
const STRINGS = ['Reply width', 'How wide the text of an answer can be. Tables and code always use the whole width.', 'Wide', 'Full width'];

test('the choice is one of three words; anything else is the default (today\'s width)', async () => {
  const { validReplyWidth, REPLY_WIDTHS, DEFAULT_REPLY_WIDTH } = await load('lib/reply-width.ts');
  assert.deepEqual([...REPLY_WIDTHS], ['comfortable', 'wide', 'full']);
  assert.equal(DEFAULT_REPLY_WIDTH, 'comfortable');
  for (const v of REPLY_WIDTHS) assert.equal(validReplyWidth(v), v);
  for (const v of [undefined, null, '', 'huge', 3, {}, 'WIDE']) assert.equal(validReplyWidth(v), 'comfortable');
});

test('the store keeps it with the other reading prefs, applies it to the page root, and resets it', () => {
  const s = src('lib/store.ts');
  assert.match(s, /replyWidth: 'comfortable',\n/);
  assert.match(s, /replyWidth: validReplyWidth\(parsed\.replyWidth\),/, 'an old or damaged saved value falls back');
  assert.match(s, /root\.dataset\.measure = state\.replyWidth;/);
  assert.match(s, /replyWidth: state\.replyWidth,\n/, 'saved with the prefs');
  assert.match(s, /replyWidth: PREFS_DEFAULTS\.replyWidth,/, 'Reset appearance puts it back');
  assert.match(s, /export function setReplyWidth\(replyWidth: ReplyWidth\) \{\n  set\(\{ replyWidth: validReplyWidth\(replyWidth\) \}\);\n  savePrefs\(\);/);
});

test('the stylesheet moves only --st-measure (so Focus reading follows); Comfortable keeps 56ch', () => {
  const css = src('styles/app.css');
  assert.match(css, /--st-measure: 56ch;/);
  assert.match(css, /:root\[data-measure='wide'\] \{ --st-measure: 72ch; \}/);
  assert.match(css, /:root\[data-measure='full'\] \{ --st-measure: 400ch; \}/);
  assert.doesNotMatch(css, /data-measure='comfortable'/);
  assert.match(css, /max-width: calc\(var\(--st-measure\) \+ 72px\);/, 'the Focus reading column is built from the same token');
});

test('Settings > Reading & focus has the row with the three options', () => {
  const settings = src('screens/Settings.tsx');
  assert.ok(settings.includes("{ section: 'Reading & focus', label: 'Reply width', hint: 'How wide the text of an answer can be. Tables and code always use the whole width.'"));
  assert.ok(settings.includes("options={[{ value: 'comfortable', label: 'Comfortable' }, { value: 'wide', label: 'Wide' }, { value: 'full', label: 'Full width' }]} onChange={setReplyWidth}"));
});

test('the four strings are in the catalog in all ten languages', () => {
  const cat = Object.fromEntries(LANGS.map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/guic-${l}.json`, import.meta.url), 'utf8'))]));
  for (const text of STRINGS) {
    const key = Object.keys(cat.en).find(k => cat.en[k] === text);
    assert.ok(key, `${text} is in guic-en.json`);
    for (const l of LANGS) assert.ok(cat[l][key] && cat[l][key].trim(), `${l}: ${key}`);
    assert.equal(/[^  ][;:?!]| [;:?!]/.test(cat.fr[key]), false, 'French: ; : ? ! only after a no-break space');
  }
  assert.match(cat.de[Object.keys(cat.en).find(k => cat.en[k] === 'Full width')], /^Volle Breite$/);
});
