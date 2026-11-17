// 0.2.16 languages check (LANG-0216): every catalog of the app and of the engine's web page is complete in all nine languages.
// locale-reviewed.test.mjs checks [DRAFT] markers and the {0} / <b> slots of plain values; this file adds what it leaves out:
// the key set of every catalog family, English sentences left in a translation, printf markers (%s, %d), and the plural forms of the
// web page (each form the language uses for whole numbers is there and keeps the {n} of its English text).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const APP = join(here, '..', 'src', 'locales');
const WEB = join(here, '..', '..', '..', 'serve', 'web', 'locales');
const LANGS = ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
const load = file => JSON.parse(readFileSync(file, 'utf8'));

/** family -> { lang -> parsed catalog }: app:main (en.json ...), app:<fam> (<fam>-en.json ...), web (serve/web/locales). */
export function catalogs(appDir = APP, webDir = WEB) {
  const out = new Map();
  for (const f of readdirSync(appDir)) {
    const m = /^(?:(.+)-)?([a-z]{2})\.json$/.exec(f);
    if (!m) continue;
    const fam = `app:${m[1] ?? 'main'}`;
    if (!out.has(fam)) out.set(fam, {});
    out.get(fam)[m[2]] = load(join(appDir, f));
  }
  const web = {};
  for (const f of readdirSync(webDir)) { const m = /^([a-z]{2})\.json$/.exec(f); if (m) web[m[1]] = load(join(webDir, f)); }
  out.set('web', web);
  return out;
}

/** English running text: three words in a row, written as words of a sentence ("Delete chat file"). Product names ("OpenAI Chat
 * Completions"), paths, commands, units and single words may stay as they are. */
export function isSentence(text) {
  return /(?:^|[^A-Za-z])[A-Za-z][a-z]+(?:['’][a-z]+)? [a-z]{2,} [a-z]{2,}/.test(String(text).replace(/\{[A-Za-z0-9_.]+\}/g, ' '));
}

/** The printf markers of a value (%s, %d, %1$s, %%), sorted. */
export function printfMarkers(text) {
  return [...String(text).matchAll(/%(?:\d+\$)?[sdif%]/g)].map(m => m[0]).sort().join(' ');
}

/** The plural categories a language uses for the whole numbers 0..10000 (the web page counts chats, messages, tool calls). */
export function integerCategories(lang) {
  const rules = new Intl.PluralRules(lang), seen = new Map();
  for (let n = 0; n <= 10000; n++) { const c = rules.select(n); if (!seen.has(c)) seen.set(c, new Set()); if (seen.get(c).size < 3) seen.get(c).add(n); }
  return seen; // category -> a few numbers it covers
}

const values = entry => typeof entry === 'string' ? [entry] : Object.values(entry ?? {}).filter(v => typeof v === 'string');

test('the helpers', () => {
  assert.equal(isSentence('Delete chat file'), true);
  assert.equal(isSentence('{0}{1} tok/s'), false);
  assert.equal(isSentence('Qwen · IQ2_M'), false);
  assert.equal(isSentence('OpenAI Chat Completions'), false);
  assert.equal(isSentence('Documents / Strata / Models'), false);
  assert.equal(isSentence('-y @modelcontextprotocol/server-filesystem /your/folder'), false);
  assert.equal(isSentence("Don't stop now"), true);
  assert.equal(printfMarkers('%s of %d'), '%d %s');
  assert.deepEqual([...integerCategories('ru').keys()].sort(), ['few', 'many', 'one']);
  assert.deepEqual([...integerCategories('ja').keys()], ['other']);
  assert.deepEqual([...integerCategories('fr').get('one')], [0, 1]);
});

test('every catalog family has every language with exactly the keys of its English catalog', () => {
  const fams = catalogs();
  assert.ok(fams.size >= 10, `catalog families found (${fams.size})`);
  const wrong = [];
  for (const [fam, cats] of fams) {
    assert.ok(cats.en, `${fam} has an English catalog`);
    const en = Object.keys(cats.en);
    for (const lang of LANGS) {
      if (!cats[lang]) { wrong.push(`${fam} ${lang}: no catalog`); continue; }
      const have = new Set(Object.keys(cats[lang]));
      for (const k of en) if (!have.has(k)) wrong.push(`${fam} ${lang}: missing ${k}`);
      for (const k of have) if (!(k in cats.en)) wrong.push(`${fam} ${lang}: ${k} is not in English`);
    }
  }
  assert.deepEqual(wrong, []);
});

test('no translation is still the English sentence', () => {
  const left = [];
  for (const [fam, cats] of catalogs()) {
    for (const [k, enEntry] of Object.entries(cats.en)) {
      const enText = typeof enEntry === 'string' ? enEntry : enEntry?.other;
      if (!isSentence(enText)) continue;
      for (const lang of LANGS) for (const v of values(cats[lang]?.[k])) if (values(enEntry).includes(v)) left.push(`${fam} ${lang} ${k}: ${v}`);
    }
  }
  assert.deepEqual(left, []);
});

test('every translation keeps the printf markers of its English text', () => {
  const wrong = [];
  for (const [fam, cats] of catalogs()) {
    for (const [k, enEntry] of Object.entries(cats.en)) {
      const want = printfMarkers(typeof enEntry === 'string' ? enEntry : enEntry?.other);
      for (const lang of LANGS) for (const v of values(cats[lang]?.[k])) if (printfMarkers(v) !== want) wrong.push(`${fam} ${lang} ${k}: [${want}] vs [${printfMarkers(v)}]`);
    }
  }
  assert.deepEqual(wrong, []);
});

test('the web page has every plural form a language uses for whole numbers, each keeping {n}', () => {
  const web = catalogs().get('web'), wrong = [];
  for (const [k, enEntry] of Object.entries(web.en)) {
    if (typeof enEntry === 'string') continue;
    const slotsOf = t => [...String(t).matchAll(/\{[A-Za-z0-9_]+\}/g)].map(m => m[0]).sort().join(' ');
    const want = slotsOf(enEntry.other);
    for (const lang of LANGS) {
      const entry = web[lang]?.[k];
      if (!entry || typeof entry !== 'object') { wrong.push(`${lang} ${k}: not a plural`); continue; }
      for (const [cat, numbers] of integerCategories(lang)) {
        const text = entry[cat];
        if (typeof text !== 'string') { wrong.push(`${lang} ${k}: no '${cat}' form (used for ${[...numbers].join(', ')})`); continue; }
        // a form for 1 alone ("one" in English, French, ...) may say the number in words ("le message"); any other form shows the count
        const onlyOne = [...numbers].every(n => n <= 1);
        if (slotsOf(text) !== want && !(onlyOne && want.split(' ').filter(s => s !== '{n}').join(' ') === slotsOf(text))) wrong.push(`${lang} ${k} ${cat}: [${want}] vs [${slotsOf(text)}]`);
      }
    }
  }
  assert.deepEqual(wrong, []);
});
