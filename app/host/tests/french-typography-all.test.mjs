// French typography in EVERY French catalog (the main one, each add-on family and the web page): a no-break space before ; : ? ! and inside « ».
// The locale generators had NBSP = " " (a plain space, so their French rule changed nothing): 49 strings in df2, dmg, engine, fix, gui, guic, pic,
// pick, r3, sav, ui and upd kept a plain space. french-catalog.test.mjs only looks at the main catalog and the first add-on families.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const APP = join(here, '..', 'src', 'locales');
const WEB = join(here, '..', '..', '..', 'terminal', 'serve', 'web', 'locales');
const files = [...readdirSync(APP).filter(f => /(^|-)fr\.json$/.test(f)).map(f => join(APP, f)), join(WEB, 'fr.json')];

const PLAIN = /(?<=\S) [;:?!]|« (?=\S)|(?<=\S) »/;

test('there are French catalogs to check', () => {
  assert.ok(files.length >= 15, `French catalogs found (${files.length})`);
});

test('no French string keeps a plain space before ; : ? ! or inside « »', () => {
  const wrong = [];
  for (const file of files) {
    const catalog = JSON.parse(readFileSync(file, 'utf8'));
    for (const [key, value] of Object.entries(catalog)) {
      for (const text of typeof value === 'string' ? [value] : Object.values(value)) {
        if (PLAIN.test(text)) wrong.push(`${file.slice(file.lastIndexOf('locales'))}:${key}`);
      }
    }
  }
  assert.deepEqual(wrong, []);
});
