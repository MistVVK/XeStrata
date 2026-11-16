// #5919 (owner's decision, 11-16): Qwen3.6-35B-A3B has no KV-format choice (its engine path ignores --kv, #3303; setup records kv None
// for it), so the Models screen's "Context memory" select is not shown for it; it showed INT8 whatever was picked. The Qwen3.8-Flash-Next
// family keeps the select, unchanged, and its saved --kv still comes back to the page and goes to setup.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const read = p => readFileSync(fileURLToPath(new URL(p, import.meta.url)), 'utf8').replace(/\r\n/g, '\n');
const screen = read('../src/components/NativeModels.tsx');
const built = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/native-catalog.ts', import.meta.url))], bundle: true, write: false, platform: 'node', format: 'esm', loader: { '.json': 'json' } });
const catalog = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

const size = (id, extra = {}) => ({ id, model: id.toUpperCase(), ...extra });
const FAMILIES = [
  { family: 'qwen', title: 'Qwen3.8-Flash-Next', images: true, sizes: [size('iq3_s')] },
  { family: 'swift', title: 'Swift 1.5', images: true, sizes: [size('swift-iq3_xxs')] },
  { family: 'qwen36', title: 'Qwen3.6-35B-A3B', images: true, sizes: [size('qwen36-ud-q4_k_xl')] },
];

test('the KV choice applies to the Qwen3.8-Flash-Next group (Swift included), not to Qwen3.6', () => {
  const groups = catalog.nativeCatalogModels(FAMILIES);
  const of = id => groups.find(g => g.sizes.some(s => s.id === id))?.family;
  assert.equal(catalog.hasKvChoice(of('iq3_s')), true);
  assert.equal(catalog.hasKvChoice(of('swift-iq3_xxs')), true, 'Swift 1.5 (#5919 reporter) keeps the select');
  assert.equal(catalog.hasKvChoice(of('qwen36-ud-q4_k_xl')), false);
  assert.equal(catalog.hasKvChoice(undefined), true, 'nothing selected yet: the page as before');
  assert.equal(catalog.hasKvChoice('some-feed-family'), true, 'feed families keep the choice they had');
});

test('the select is drawn only where the choice applies, and is the same select as before', () => {
  assert.match(screen, /\{tr\(hasKvChoice\(selectedFamily\?\.family\) \? <label className="input-field"><span>\{tr\("Context memory"\)\}<\/span><Select ariaLabel=\{tr\("Native KV format"\)\} value=\{kv\} options=\{KV_CHOICES\.map\(c => \(\{ value: c\.value, label: c\.label, title: c\.hint \}\)\)\} onChange=\{setKv\}\/>/);
  assert.equal(screen.split('ariaLabel={tr("Native KV format")}').length, 2, 'one KV select');
});

test('a saved --kv of a Qwen3.8 model still comes back to the page and goes to setup', () => {
  assert.match(screen, /setKv\(installed\.kv \?\? 'int8'\);/, 'the saved format is shown when the model opens');
  assert.match(screen, /const values = useMemo\(\(\) => \(\{[^}]*\bkv,/, 'the chosen format goes to the plan / setup');
  const control = read('../../../tools/strata_mcp.py');
  assert.match(control, /"kv": val\("--kv"\)/, 'the installed model reports the --kv of its run config');
});
