// Visual walkthrough (GUI-VIS-1117, finding 10): the engine-setting texts the host writes in English (Draft layer (MTP), MTP per request, the groups, the drafting hint)
// and the Monitor's "Last: ..." line stayed English in several languages.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { build } from 'esbuild';

const LANGS = ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
const built = await build({ stdin: { contents: "export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { tr, setTranslationLanguage, loadCatalog } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
for (const l of LANGS) await loadCatalog(l);

// the texts exactly as the host writes them (server/native-settings.mjs); the drafting hint carries the drafter's size
const settings = readFileSync(new URL('../server/native-settings.mjs', import.meta.url), 'utf8');
const fieldTexts = [...settings.matchAll(/(?:label|group|hint):'([^']{4,})'/g)].map(m => m[1]);
const { draftingHint } = await import(new URL('../server/native-settings.mjs', import.meta.url));
const mention = t => /Draft layer|MTP per request|Recovery|GPU & memory|draft layer/.test(t);

test('every text of the engine settings fields that names the draft layer, MTP, or the two groups is translated in all nine languages', async () => {
  const texts = [...new Set(fieldTexts.filter(mention)), draftingHint(1.2), draftingHint(0.8)];
  assert.ok(texts.length >= 6, 'the texts were found');
  for (const l of LANGS) {
    await setTranslationLanguage(l);
    for (const t of texts) {
      const said = tr(t);
      assert.notEqual(said, t, `${l}: still English: ${t.slice(0, 50)}`);
      assert.ok(!/\{\d+\}/.test(said), `${l}: raw slot in ${said.slice(0, 50)}`);
    }
  }
});

test('the drafter size is carried into the sentence, whatever the number', async () => {
  for (const l of LANGS) {
    await setTranslationLanguage(l);
    assert.ok(tr(draftingHint(1.2)).includes('1.2'), `${l} 1.2`);
    assert.ok(tr(draftingHint(0.75)).includes('0.75'), `${l} 0.75`);
  }
});

test('the Monitor "Last:" line is translated with its numbers in place', async () => {
  for (const l of LANGS) {
    await setTranslationLanguage(l);
    const a = tr('Last: 49 tokens at 43.6 tok/s'), b = tr('Last: 49 tokens');
    assert.ok(a.includes('49') && a.includes('43.6') && !a.startsWith('Last'), `${l}: ${a}`);
    assert.ok(b.includes('49') && !b.startsWith('Last'), `${l}: ${b}`);
  }
});
