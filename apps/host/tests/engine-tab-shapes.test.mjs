// Settings > Engine with the engine of a Qwen4 model shape (APP-WIN item 6): Control lists it as its own family ("shape:<shape>", title null); the page titles
// the group with the model's catalog title (the shape's name when the catalog has none), never the family id; switch and delete pass the family through;
// an install with no shape engine renders exactly as before (the page source still titles Control's groups with Control's title).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync as read } from 'node:fs';
import { build } from 'esbuild';
import { changeEngine } from '../server/engine-library.mjs';

const built = await build({ stdin: { contents: "export * from './src/lib/engine-list';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { engineGroupTitle } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

const catalog = [{ family: 'qwen', title: 'Qwen3.8-Flash-Next' }, { family: 'qwen4exp_q4_lite', title: 'Qwen4 Lite' }];

test('the group heading: Control\'s title for Qwen and DeepSeek, the model\'s catalog title for a shape, the shape\'s name without one; never the family id', () => {
  assert.equal(engineGroupTitle({ family: 'qwen', title: 'Qwen and other models', engines: [] }, catalog), 'Qwen and other models');
  assert.equal(engineGroupTitle({ family: 'deepseek', title: 'DeepSeek', engines: [] }, catalog), 'DeepSeek');
  assert.equal(engineGroupTitle({ family: 'shape:q4_lite', shape: 'q4_lite', title: null, engines: [] }, catalog), 'Qwen4 Lite');
  assert.equal(engineGroupTitle({ family: 'shape:q4_mid', shape: 'q4_mid', title: null, engines: [] }, catalog), 'q4_mid');
  assert.equal(engineGroupTitle({ family: 'shape:q4_mid', title: null, engines: [] }, null), 'q4_mid', 'no shape field: from the family, no catalog');
  for (const g of [{ family: 'shape:q4_lite', shape: 'q4_lite', title: null, engines: [] }, { family: 'shape:q4_mid', title: null, engines: [] }])
    assert.doesNotMatch(engineGroupTitle(g, catalog), /shape:/);
});

test('switch and delete take a shape family and pass it to Control; anything else is still refused', async () => {
  const calls = [];
  const control = { useEngine: async a => { calls.push(['use', a]); return { status: 'switched' }; }, removeEngine: async a => { calls.push(['remove', a]); return { status: 'deleted' }; } };
  assert.equal((await changeEngine(control, 'use', { family: 'shape:q4_lite', folder: '0.1.59-windows-x86_64-cuda-cuda13' })).status, 200);
  assert.equal((await changeEngine(control, 'remove', { family: 'shape:q4_lite', folder: '0.1.58-windows-x86_64-cuda-cuda13' })).status, 200);
  assert.deepEqual(calls.map(c => c[1].family), ['shape:q4_lite', 'shape:q4_lite']);
  for (const family of ['shape:', 'shape:Q4', 'shape:../x', 'shapes:q4_lite', 'q4_lite', 'strata-q4-lite'])
    assert.equal((await changeEngine(control, 'use', { family, folder: 'a' })).status, 400, family);
  assert.equal(calls.length, 2);
});

test('the page: Control\'s title when it has one (no change for an install without shape engines), the catalog-derived one otherwise', () => {
  const page = read(new URL('../src/components/SettingsEngine.tsx', import.meta.url), 'utf8');
  assert.match(page, /\{groups\.length > 1 \? <h3 className="engine-group__title">\{group\.title \? tr\(group\.title\) : engineGroupTitle\(group, native\?\.catalog\?\.families\)\}<\/h3> : null\}/);
});

test('the Use dialog (owner 11-09): a shape group says "{model} runs on engine {version} from its next start." with the group title; Strata and DeepSeek keep their sentence', () => {
  const page = read(new URL('../src/components/SettingsEngine.tsx', import.meta.url), 'utf8');
  assert.match(page, /const useGroup = use \? groups\.find\(g => g\.family === use\.family\) : undefined;/);
  assert.match(page, /useGroup\?\.family\.startsWith\('shape:'\) \? tr\("\{0\} runs on engine \{1\} from its next start\. Your models and settings stay as they are\. You can switch back at any time\.", \[engineGroupTitle\(useGroup, native\?\.catalog\?\.families\), use\?\.version \?\? ''\]\) : tr\("Models you start from now on run on engine \{0\}\. Your models and settings stay as they are\. You can switch back at any time\."/);
  // the words as the person reads them, with the owner's example
  const en = JSON.parse(read(new URL('../src/locales/engine-en.json', import.meta.url), 'utf8'));
  const key = Object.keys(en).find(k => en[k] === '{0} runs on engine {1} from its next start. Your models and settings stay as they are. You can switch back at any time.');
  assert.ok(key, 'in the English catalog');
  const title = engineGroupTitle({ family: 'shape:q4_lite', shape: 'q4_lite', title: null, engines: [] }, [{ family: 'qwen4exp_q4_lite', title: 'Qwen4' }]);
  assert.equal(en[key].replace('{0}', title).replace('{1}', '0.1.60'), 'Qwen4 runs on engine 0.1.60 from its next start. Your models and settings stay as they are. You can switch back at any time.');
  for (const lang of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr']) {
    const t = JSON.parse(read(new URL(`../src/locales/engine-${lang}.json`, import.meta.url), 'utf8'))[key];
    assert.ok(t && t.includes('{0}') && t.includes('{1}'), lang);
    // only the first sentence is new: the reassurance is the existing translation of the Strata engine's sentence, word for word
    const use = JSON.parse(read(new URL(`../src/locales/engine-${lang}.json`, import.meta.url), 'utf8')).et15;
    const rest = use.slice(use.search(/[。.]\s*/) + use.match(/[。.]\s*/)[0].length);
    assert.ok(rest.length > 5 && t.endsWith(rest), lang + ': ' + t);
  }
  assert.ok(Object.values(en).includes('Models you start from now on run on engine {0}. Your models and settings stay as they are. You can switch back at any time.'), 'the Strata / DeepSeek sentence is unchanged');
});
