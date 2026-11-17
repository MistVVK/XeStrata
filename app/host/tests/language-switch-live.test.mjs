// Visual walkthrough (GUI-VIS-1117, finding 2): after a live language switch, parts of the chat kept the words of the language before (German with Japanese,
// Hebrew with Korean) until a reload, and the tool-summary line of a turn ("Ran a command, edited a file") and the code label were English in every language.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { build } from 'esbuild';

const LANGS = ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
const built = await build({ stdin: { contents: "export * from './src/lib/i18n'; export * from './src/lib/group-line';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { tr, setTranslationLanguage, loadCatalog, groupSentence } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
for (const l of LANGS) await loadCatalog(l);

const walk = dir => readdirSync(dir).flatMap(name => { const p = join(dir, name); return statSync(p).isDirectory() ? walk(p) : [p]; });

test('every memoised component that draws translated words subscribes to the language', () => {
  const missing = [];
  for (const file of walk('src').filter(f => /\.tsx$/.test(f))) {
    const text = readFileSync(file, 'utf8').replace(/\r\n/g, '\n');
    if (!/\bmemo\(/.test(text) || !/\btr\(/.test(text)) continue;
    // each memo() component body (or the one the memo wraps) must read the language from the store, else a live switch changes no prop and nothing redraws
    const bodies = [...text.matchAll(/memo\((?:function\s+\w+|\w+)/g)].length;
    const reads = [...text.matchAll(/useStore\(\s*\(?s\)?\s*=>\s*s\.language\s*\)/g)].length;
    if (reads < bodies) missing.push(`${file}: ${bodies} memo(), ${reads} language reads`);
  }
  assert.deepEqual(missing, []);
});

test('the English line of a turn is the sentence the host wrote', async () => {
  await setTranslationLanguage('en');
  for (const line of ['Ran a command', 'Ran a command, edited a file', 'Ran 2 commands, read a file, edited 3 files', 'Searched 4 searches, looked up a web page', 'Used the browser for 2 browser steps', 'Started a helper'])
    assert.equal(groupSentence(line), line);
  assert.equal(groupSentence('A sentence the page does not know'), 'A sentence the page does not know');
  assert.equal(groupSentence(''), '');
});

test('the line is said in the language of the moment and holds no word of the host\'s English in any language', async () => {
  for (const l of LANGS) {
    await setTranslationLanguage(l);
    for (const line of ['Read a file', 'Ran a command, edited a file', 'Ran 2 commands, read 3 files, edited a file', 'Searched a search, looked up 2 web pages', 'Used 2 tools']) {
      const said = groupSentence(line);
      assert.notEqual(said, line, `${l}: "${line}" is still English`);
      assert.ok(!/\{\d+\}/.test(said), `${l}: raw slot in ${said}`);
      assert.ok(!/\b(Ran|Read|Edited|Searched|Looked up|Started|Used)\b/.test(said), `${l}: English verb left in ${said}`);
    }
  }
});

test('a switch from one language to another changes the sentence at once (no cache of the old one)', async () => {
  await setTranslationLanguage('ja');
  const ja = groupSentence('Ran a command, edited a file');
  await setTranslationLanguage('de');
  const de = groupSentence('Ran a command, edited a file');
  await setTranslationLanguage('he');
  const he = groupSentence('Ran a command, edited a file');
  assert.equal(new Set([ja, de, he]).size, 3);
  assert.equal(de, 'Befehl ausgeführt, Datei bearbeitet');
  assert.match(ja, /コマンド/);
  assert.match(he, /[֐-׿]/);
});

test('the Spanish, French, Portuguese and Russian lines join with a small letter after the first part, Japanese and Chinese with their own comma', async () => {
  await setTranslationLanguage('es');
  assert.equal(groupSentence('Ran a command, edited a file'), 'Comando ejecutado, archivo editado');
  await setTranslationLanguage('fr');
  assert.equal(groupSentence('Ran a command, edited a file'), 'Commande exécutée, fichier modifié');
  await setTranslationLanguage('ja');
  assert.ok(groupSentence('Ran a command, edited a file').includes('、'));
  await setTranslationLanguage('zh');
  assert.ok(groupSentence('Ran a command, edited a file').includes('，'));
});

test('the code label has a word of its own in every language', async () => {
  for (const l of LANGS) {
    await setTranslationLanguage(l);
    assert.ok(tr('code').length > 0);
    if (l === 'he' || l === 'ja' || l === 'ko' || l === 'zh' || l === 'ru') assert.notEqual(tr('code'), 'code', l);
  }
});
