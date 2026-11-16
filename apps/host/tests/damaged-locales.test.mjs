// The strings of the damaged-chat surface (#5421, src/locales/dmg-<lang>.json) exist in every language with the same slots and no empty
// text, do not collide with the catalogs they are merged with, and translate through tr(). Same shape as tools-locales.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';

const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr'];
const catalogs = Object.fromEntries(await Promise.all(languages.map(async l => [l, JSON.parse(await readFile(`src/locales/dmg-${l}.json`, 'utf8'))])));
const neighbours = ['en', 'tune-en', 'fix-en', 'gui-en', 'sav-en', 'ui-en'].map(async name => JSON.parse(await readFile(`src/locales/${name}.json`, 'utf8')));
const merged = Object.assign({}, ...(await Promise.all(neighbours)));
const slots = text => [...text.matchAll(/\{\d+\}/g)].map(m => m[0]).sort();
const compiled = await build({ stdin: { contents: "export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { tr, setTranslationLanguage, loadCatalog } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
for (const l of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr']) await loadCatalog(l);  // the non-English catalogs load on first use (src/lib/i18n.ts loadCatalog); load them before the first tr()

test('every language has every damaged-chat string with the same slots and no empty text', () => {
  for (const l of languages) {
    assert.deepEqual(Object.keys(catalogs[l]), Object.keys(catalogs.en), l);
    for (const [id, source] of Object.entries(catalogs.en)) {
      assert.ok(catalogs[l][id].trim(), `${l}:${id} is empty`);
      assert.deepEqual(slots(catalogs[l][id]), slots(source), `${l}:${id}`);
    }
  }
});

test('the damaged-chat ids and texts do not collide with the catalogs they are merged with, and translate through tr()', () => {
  for (const [id, text] of Object.entries(catalogs.en)) {
    assert.ok(!(id in merged), id);
    assert.ok(!(text in merged), `already in another catalog: ${text}`);
  }
  for (const l of languages) {
    setTranslationLanguage(l);
    for (const [id, text] of Object.entries(catalogs.en)) assert.equal(tr(text), catalogs[l][id], `${l}:${id}`);
  }
  setTranslationLanguage('ru');
  assert.equal(tr('Delete {0}.json'.replace('{0}', 's_9ab1cd')), catalogs.ru.dmg05.replace('{0}', 's_9ab1cd'));
  setTranslationLanguage('en');
});
