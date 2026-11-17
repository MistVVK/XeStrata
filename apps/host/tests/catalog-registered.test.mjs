// Every catalog family file (src/locales/<family>-<lang>.json) must be loaded by that language's catalog (src/lib/catalog-<lang>.ts) and its English file
// by i18n.ts. The Hebrew catalog did not load guic-he.json: every string of the 0.2.16 new-UI features (move a chat, Your rules, ...) showed in English.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';

const locales = new URL('../src/locales/', import.meta.url);
const lib = new URL('../src/lib/', import.meta.url);
const files = readdirSync(locales);
const families = [...new Set(files.map(f => /^(.+)-en\.json$/.exec(f)?.[1]).filter(Boolean))];
const langs = ['zh', 'ja', 'de', 'fr', 'es', 'ru', 'pt', 'ko', 'he'];

test('every catalog family is loaded by every language catalog, and its English file by i18n.ts', () => {
  assert.ok(families.length >= 15, `families found: ${families.join(', ')}`);
  const i18n = readFileSync(new URL('i18n.ts', lib), 'utf8');
  for (const fam of families) assert.ok(i18n.includes(`'../locales/${fam}-en.json'`), `i18n.ts loads ${fam}-en.json`);
  for (const lang of langs) {
    const catalog = readFileSync(new URL(`catalog-${lang}.ts`, lib), 'utf8');
    for (const fam of families) {
      assert.ok(files.includes(`${fam}-${lang}.json`), `${fam}-${lang}.json exists`);
      assert.ok(catalog.includes(`'../locales/${fam}-${lang}.json'`), `catalog-${lang}.ts loads ${fam}-${lang}.json`);
    }
  }
});

test('the merged order in every catalog is the same list of families', () => {
  const spread = text => (/export default \{([^}]*)\}/.exec(text)?.[1] ?? '').split(',').map(s => s.trim().replace(/^\.\.\./, '')).filter(Boolean);
  const reference = spread(readFileSync(new URL('catalog-de.ts', lib), 'utf8'));
  assert.ok(reference.length >= 15 && reference.includes('guic'));
  for (const lang of langs) assert.deepEqual(spread(readFileSync(new URL(`catalog-${lang}.ts`, lib), 'utf8')), reference, `catalog-${lang}.ts merges the same families in the same order`);
});
