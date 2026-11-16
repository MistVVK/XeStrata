/** The speed pill's live templates need catalog keys: without a key, the translation lookup
 * re-extracts the slots from the template itself, and a non-English chat footer shows raw
 * "{0}{1}" and English "last reply" instead of the number in the language's own order. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';

const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr'];
const read = async name => JSON.parse(await readFile(`src/locales/${name}.json`, 'utf8'));
const en = await read('en');
const built = await build({ stdin: { contents: "export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { tr, setTranslationLanguage, loadCatalog } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
for (const l of languages.filter(l => l !== 'en')) await loadCatalog(l);   // the other catalogs load on first use (src/lib/i18n.ts)

test('both live templates of the chat speed pill have English catalog keys', () => {
  assert.ok(Object.values(en).includes('{0}{1} tok/s'), 'the English catalog has no key for "{0}{1} tok/s"');
  assert.ok(Object.values(en).includes('{0}{1} tok/s · last reply'), 'the English catalog has no key for "{0}{1} tok/s · last reply"');
});

test('the speed pill places the number in a translated sentence in every language, never a raw placeholder', () => {
  const PLACE = '\u0001';
  const around = (template, before, number) => { const [head, tail = ''] = tr(template, [before, PLACE]).split(PLACE); return head + number + tail; };
  for (const language of languages) {
    setTranslationLanguage(language);
    for (const template of ['{0}{1} tok/s', '{0}{1} tok/s · last reply']) {
      const out = around(template, '≈ ', '18');
      assert.ok(!out.includes('{'), `${language}: ${out} still holds a raw placeholder`);
      assert.ok(out.includes('18'), `${language}: ${out} does not carry the number`);
    }
  }
  setTranslationLanguage('zh');
  assert.equal(around('{0}{1} tok/s · last reply', '≈ ', '18'), '≈ 18 词元/秒 · 上次回复');
  assert.equal(around('{0}{1} tok/s', '≈ ', '18'), '≈ 18 词元/秒');
  setTranslationLanguage('ja');
  assert.ok(around('{0}{1} tok/s · last reply', '', '71').startsWith('71'), 'Japanese puts the number where its words expect it');
  setTranslationLanguage('ko');
  assert.ok(around('{0}{1} tok/s · last reply', '', '71').includes('토큰/초'), 'Ko translates tok/s');
  setTranslationLanguage('en');
});
