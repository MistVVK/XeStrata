// 0.2.16 translation pass (owner: "all languages checked"): every catalog the app and the engine's web page ship is checked as a whole.
// No value may still carry a [DRAFT] marker, and in every language each value keeps exactly the placeholders and tags of its English
// text ({0}, {name}, <b>...), so a translated string never drops or invents a value the code fills in.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const APP = join(here, '..', 'src', 'locales');
const WEB = join(here, '..', '..', '..', 'terminal', 'serve', 'web', 'locales');
const LANGS = ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];

// family name -> { lang -> file } for the app catalogs (main = en.json, family = <fam>-en.json), plus the web page's catalogs
function families() {
  const out = new Map();
  for (const f of readdirSync(APP)) {
    const m = /^(?:(.+)-)?([a-z]{2})\.json$/.exec(f);
    if (!m) continue;
    const fam = `app:${m[1] ?? 'main'}`;
    if (!out.has(fam)) out.set(fam, {});
    out.get(fam)[m[2]] = join(APP, f);
  }
  if (existsSync(WEB)) {
    const web = {};
    for (const f of readdirSync(WEB)) { const m = /^([a-z]{2})\.json$/.exec(f); if (m) web[m[1]] = join(WEB, f); }
    out.set('web', web);
  }
  return out;
}
const load = file => JSON.parse(readFileSync(file, 'utf8'));
const strings = obj => Object.entries(obj).filter(([, v]) => typeof v === 'string');

/** The placeholders and markup tags in a value, sorted, as one comparable string. */
export function slots(text) {
  return [...String(text).matchAll(/\{[A-Za-z0-9_.]+\}|<\/?[A-Za-z][A-Za-z0-9-]*[^>]*>/g)].map(m => m[0].replace(/\s+/g, ' ')).sort().join(' ');
}

/** English plural endings are filled in as a placeholder glued to a word ("{0} file{1} skipped", {1} = "s" or ""); a language that
 * says the plural its own way leaves that one out. Only such a glued placeholder may be missing; nothing may be added. */
export function withoutPluralSuffix(text) {
  return String(text).replace(/(?<=[A-Za-z])\{\d+\}(?![A-Za-z0-9])/g, '');
}

test('the slot reader finds placeholders and tags, in any order', () => {
  assert.equal(slots('Delete {0}.json'), '{0}');
  assert.equal(slots('{1} von {0}'), slots('{0} of {1}'));
  assert.equal(slots('Branch of {title}'), '{title}');
  assert.equal(slots('Press <b>Run</b>'), '</b> <b>');
  assert.notEqual(slots('{0}{1} tok/s'), slots('{0} tok/s'));
  assert.equal(slots(withoutPluralSuffix('{0} file{1} skipped.')), '{0}', 'an English plural ending may be left out');
  assert.equal(slots(withoutPluralSuffix('{0}{1} tok/s')), '{0} {1}', 'a value placeholder is never a plural ending');
});

test('no catalog value still carries a [DRAFT] marker', () => {
  const left = [];
  for (const [fam, files] of families()) {
    for (const [lang, file] of Object.entries(files)) {
      for (const [k, v] of strings(load(file))) if (/\[DRAFT\]/i.test(v)) left.push(`${fam} ${lang} ${k}`);
    }
  }
  assert.deepEqual(left, []);
});

// The keys of the 0.2.16 pass (GUI-1116 and GUI-B-1116): each language has a real translation, not the English text, unless the English
// is only placeholders and a unit the catalog keeps as is ("tok/s", as at "at {0} tok/s").
const PASS = {
  'app:main': ['s9039', 's9040', 's9041', 's9042', 's9043', 's9044', 's9045', 's9046', 's9047', 's9048', 's9049', 's9050', 's9051', 's9101'],
  'app:dmg': ['dmg01', 'dmg02', 'dmg03', 'dmg04', 'dmg05', 'dmg06', 'dmg07', 'dmg08', 'dmg09', 'dmg10', 'dmg11'],
  'app:fix': ['fx86'],
  web: ['header.github'],
};
const wordsOf = text => String(text).replace(/\{[A-Za-z0-9_.]+\}/g, '').replace(/\btok\/s\b/g, '').replace(/[^\p{L}]+/gu, '');

test('the keys of the 0.2.16 translation pass are translated in every language', () => {
  const fams = families();
  const english = [];
  let seen = 0;
  for (const [fam, keys] of Object.entries(PASS)) {
    const files = fams.get(fam);
    if (!files) continue;
    const en = load(files.en);
    for (const k of keys) {
      if (typeof en[k] !== 'string') continue; // a key of the other lane's branch, until the branches meet
      for (const lang of LANGS) {
        const v = load(files[lang])[k];
        assert.equal(typeof v, 'string', `${fam} ${lang} ${k} exists`);
        seen++;
        if (v === en[k] && wordsOf(en[k]) !== '') english.push(`${fam} ${lang} ${k}`);
      }
    }
  }
  assert.ok(seen >= 25 * LANGS.length, `the pass keys were found (${seen} values)`);
  assert.deepEqual(english, []);
});

test('every translated value keeps the placeholders and tags of its English text', () => {
  const fams = families();
  assert.ok(fams.size >= 10, `catalog families found (${fams.size})`);
  const wrong = [];
  for (const [fam, files] of fams) {
    assert.ok(files.en, `${fam} has an English catalog`);
    const en = load(files.en);
    for (const lang of LANGS) {
      if (!files[lang]) continue;
      const cat = load(files[lang]);
      for (const [k, v] of strings(cat)) {
        if (typeof en[k] !== 'string') continue;
        if (slots(v) !== slots(en[k]) && slots(v) !== slots(withoutPluralSuffix(en[k]))) wrong.push(`${fam} ${lang} ${k}: en [${slots(en[k])}] vs [${slots(v)}]`);
      }
    }
  }
  assert.deepEqual(wrong, []);
});
