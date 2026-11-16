import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';

// French (#3970): the catalogs are complete (the generic language tests cover keys and placeholders); these tests cover
// what is specific to French: spacing before ? ! : ;, apostrophes, short labels staying short, and the picker entry.
const names = [['en', 'fr'], ['fix-en', 'fix-fr'], ['tune-en', 'tune-fr'], ['tools-en', 'tools-fr']];
const read = async name => JSON.parse(await readFile(`src/locales/${name}.json`, 'utf8'));
const compiled = await build({ stdin: { contents: "export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { tr, setTranslationLanguage, localeTag, currentLanguage, LANGUAGES, validLanguage, loadCatalog } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
for (const l of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he']) await loadCatalog(l);  // the non-English catalogs load on first use (src/lib/i18n.ts loadCatalog); load them before the first tr()

test('French is in the language picker under its own name', () => {
  const entry = LANGUAGES.find(item => item.value === 'fr');
  assert.equal(entry?.label, 'Français');
  assert.ok(validLanguage('fr'));
  assert.equal(LANGUAGES.length, 9);
  setTranslationLanguage('fr');
  assert.equal(currentLanguage(), 'fr');
  assert.equal(localeTag(), 'fr');
  assert.equal(tr('Settings'), 'Paramètres');
  assert.equal(tr('Cancel'), 'Annuler');
  assert.equal(tr('Open {0}', ['Modèles']).includes('{'), false);
  assert.equal(tr('unrecognized user content Ω'), 'unrecognized user content Ω');
  setTranslationLanguage('en');
});

test('French text keeps the French typography: a no-break space before ? ! : ; and a typographic apostrophe', async () => {
  const bad = [];
  for (const [englishName, frenchName] of names) {
    const english = await read(englishName), french = await read(frenchName);
    for (const [id, text] of Object.entries(french)) {
      if (text === english[id]) continue; // kept as is: names, file names, class names
      if (/[A-Za-zÀ-ÿ\)\]\d%] [?!;:]/.test(text) && !/https?:|\d:\d/.test(text)) bad.push(`${id}: ordinary space before punctuation: ${text}`);
      if (/[A-Za-zÀ-ÿ\)\]\d%][?!;]/.test(text)) bad.push(`${id}: no space before ? ! or ;: ${text}`);
      if (/[A-Za-zÀ-ÿ]'[A-Za-zÀ-ÿ{]/.test(text)) bad.push(`${id}: straight apostrophe: ${text}`);
    }
  }
  assert.deepEqual(bad, []);
});

test('short labels stay short in French (buttons, tabs, chips)', async () => {
  const english = await read('en'), french = await read('fr');
  const long = [];
  for (const [id, text] of Object.entries(english)) {
    if (text.length > 10 || french[id] === text || /\{\d+\}/.test(text)) continue;
    // a label of up to 10 letters may take up to 3 times the room, and never more than 36 characters
    if (french[id].length > Math.max(18, text.length * 3.2) || french[id].length > 36) long.push(`${id}: ${text} -> ${french[id]} (${french[id].length})`);
  }
  assert.deepEqual(long, []);
});

test('no French string is still English by accident: at most 5% of the main catalog is left unchanged', async () => {
  const english = await read('en'), french = await read('fr');
  const same = Object.keys(english).filter(id => english[id] === french[id]);
  assert.ok(same.length < Object.keys(english).length * 0.05, `${same.length} unchanged`);
  // every unchanged string has no real words in it, or is a name that is spelled the same in French
  const words = same.filter(id => english[id].split(/\s+/).length >= 4 && /\s(the|your|you|is|are|this|that|with|from|and)\s/i.test(english[id]) && !/^[-/\\]/.test(english[id]));
  assert.deepEqual(words.map(id => `${id}: ${english[id]}`), []);
});
