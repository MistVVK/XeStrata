import test from 'node:test';
import assert from 'node:assert/strict';
import ts from 'typescript';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { build } from 'esbuild';

const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr'];
const read = async name => JSON.parse(await readFile(`src/locales/${name}.json`, 'utf8'));
const built = await build({ stdin: { contents: "export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { tr, setTranslationLanguage, loadCatalog } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
for (const l of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr']) await loadCatalog(l);  // the non-English catalogs load on first use (src/lib/i18n.ts loadCatalog); load them before the first tr()

test('every tr("...") literal and every command hint has a key in the English catalog, so no language is stuck in English (#3479)', async () => {
  const known = new Set();
  for (const name of ['en', 'tune-en', 'tools-en', 'fix-en', 'gh-en', 'ui-en', 'engine-en', 'pick-en', 'df2-en', 'upd-en', 'r3-en', 'pic-en', 'gui-en', 'sav-en', 'dmg-en']) for (const v of Object.values(await read(name))) known.add(String(v));
  const missing = [];
  const scan = async dir => {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) { if (entry.name !== 'locales') await scan(path); continue; }
      if (!/\.tsx?$/.test(path)) continue;
      const source = await readFile(path, 'utf8'), sf = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true, path.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
      const visit = node => {
        if (ts.isCallExpression(node) && node.expression.getText(sf) === 'tr' && node.arguments[0] && (ts.isStringLiteral(node.arguments[0]) || ts.isNoSubstitutionTemplateLiteral(node.arguments[0]))) {
          const text = node.arguments[0].text.trim();
          if (text && /[a-zA-Z]{2}/.test(text) && !known.has(text)) missing.push(`${path}: ${text}`);
        }
        if (ts.isPropertyAssignment(node) && node.name.getText(sf) === 'hint' && ts.isStringLiteral(node.initializer) && !path.endsWith('chat-showcase.ts')) {
          const text = node.initializer.text.trim();
          if (text && !known.has(text)) missing.push(`${path}: hint ${text}`);
        }
        ts.forEachChild(node, visit);
      };
      visit(sf);
    }
  };
  await scan('src');
  assert.deepEqual(missing, []);
});

test('the fix catalogs have the same keys and placeholders in all nine languages', async () => {
  const en = await read('fix-en');
  for (const language of languages) {
    const catalog = await read(`fix-${language}`);
    assert.deepEqual(Object.keys(catalog), Object.keys(en), language);
    for (const [id, text] of Object.entries(catalog)) assert.deepEqual((text.match(/\{\d+\}/g) ?? []).sort(), (en[id].match(/\{\d+\}/g) ?? []).sort(), `${language} ${id}`);
  }
});

test('the /plan, /research and /new hints and the four "Open ..." hints are translated (#3479, #3825)', () => {
  for (const language of ['zh', 'ru']) {
    setTranslationLanguage(language);
    for (const hint of ['Have the assistant write a plan first; nothing runs until you approve it', 'Search the web, read the pages and answer with sources', 'Start a new conversation; keep this one'])
      assert.notEqual(tr(hint), hint, `${language}: ${hint}`);
    for (const label of ['Model settings', 'Serving', 'Models', 'Skills']) {
      const hint = tr(`Open ${label}`);
      assert.notEqual(hint, `Open ${label}`, `${language}: ${label} template`);
      assert.ok(!hint.includes(label), `${language}: the label is translated too: ${hint}`);
    }
  }
  setTranslationLanguage('zh');
  assert.equal(tr('Open Skills'), '打开 技能');
  setTranslationLanguage('en');
  assert.equal(tr('Open Skills'), 'Open Skills');
});
