// One-off: wire the gtui-<lang>.json family (lane GUI-TUI, 0.2.17) into the catalogs. Safe to run twice.
import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
for (const lang of ['zh', 'ja', 'de', 'fr', 'es', 'ru', 'pt', 'ko', 'he']) {
  const file = join(root, 'src', 'lib', `catalog-${lang}.ts`);
  let text = readFileSync(file, 'utf8');
  if (text.includes('gtui')) continue;
  text = text.replace(/\.\.\.fixa\}/, '...fixa,...gtui}');
  text = text.replace(/(import fixa from '[^']+';)/, `$1\nimport gtui from '../locales/gtui-${lang}.json';`);
  writeFileSync(file, text);
}
const i18n = join(root, 'src', 'lib', 'i18n.ts');
let t = readFileSync(i18n, 'utf8');
if (!t.includes('gtui_en')) {
  t = t.replace(/(import fixa_en from '[^']+';)/, `$1\n/* The 0.2.17 strings of lane GUI-TUI (gtui-<lang>.json). */\nimport gtui_en from '../locales/gtui-en.json';`);
  t = t.replace(/\.\.\.fixa_en\}/, '...fixa_en,...gtui_en}');
  writeFileSync(i18n, t);
}
const gaps = join(root, 'tests', 'locale-gaps.test.mjs');
let g = readFileSync(gaps, 'utf8');
if (!g.includes('gtui-en')) { g = g.replace("'fixa-en']", "'fixa-en', 'gtui-en']"); writeFileSync(gaps, g); }
console.log('wired');
