/** Models > This computer: free graphics memory and RAM first, the SSD only when the model does not fit, and a loaded model changes the card. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { mkdtempSync, writeFileSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const stdin = { contents: "import { createElement } from 'react'; import { renderToStaticMarkup } from 'react-dom/server'; import { PcFitCard } from './src/components/PcFitCard'; export { setTranslationLanguage } from './src/lib/i18n'; export const render = props => renderToStaticMarkup(createElement(PcFitCard, props));", resolveDir: fileURLToPath(new URL('..', import.meta.url)), loader: 'ts' };
const built = await build({ stdin, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json', '.tsx': 'tsx' }, jsx: 'automatic', logLevel: 'silent', banner: { js: "import { createRequire } from 'node:module'; const require = createRequire(import.meta.url);" } });
const dir = mkdtempSync(join(tmpdir(), 'strata-pc-test-'));
writeFileSync(join(dir, 'card.mjs'), built.outputFiles[0].text);
const { render, setTranslationLanguage } = await import(pathToFileURL(join(dir, 'card.mjs')).href);
rmSync(dir, { recursive: true, force: true });
const fit = (spillGb, vramUseGb = 12, ramUseGb = 44) => ({ memoryGb: vramUseGb + ramUseGb + spillGb, vramUseGb, ramUseGb, spillGb, known: true });
const card = props => render({ gpuName: 'RTX 5070', ramGb: 64, vramGb: 12, ...props });
const live = { vramTotalGb: 12, freeVramGb: 10.3, freeRamGb: 49.8, diskReadGbs: null };

test('free graphics memory and free RAM come first, as real readings', () => {
  setTranslationLanguage('en');
  const html = card({ fit: fit(0), live });
  assert.match(html, /Free right now/);
  assert.match(html, /Graphics memory<\/span><b>10\.3 GB free<\/b>/);
  assert.match(html, /RAM<\/span><b>49\.8 GB free<\/b>/);
  assert.match(html, /of 12 GB/);
  assert.doesNotMatch(card({ fit: fit(0), live: null }), /Free right now/, 'no readings: no invented numbers');
});

test('the card does not repeat the SSD warning the version picker already gives; a version that fits says nothing either', () => {
  setTranslationLanguage('en');
  assert.doesNotMatch(card({ fit: fit(0), live }), /SSD/);
  assert.doesNotMatch(card({ fit: fit(0), live }), /Fits fully in memory/, 'no "Fits" words (owner, 11-01)');
  assert.doesNotMatch(card({ fit: fit(30), live }), /read from the SSD/, 'said once, under the picker');
  assert.match(card({ fit: fit(30), live }), /This version would take about 12 GB of graphics memory and 44 GB of RAM\./);
});

test('a loaded model leaves the free memory and nothing else: no hardware line, no name, no Loaded, no "It takes about", no numbers, no SSD line', () => {
  setTranslationLanguage('en');
  const fits = card({ fit: fit(0), live, loaded: [{ id: 'iq3_s', label: 'Qwen3.8-Flash-Next · IQ3_S', fit: fit(0) }], selectedLoaded: true });
  assert.match(fits, /This computer/);
  assert.match(fits, /Free right now/);
  assert.match(fits, /Graphics memory<\/span><b>10\.3 GB free<\/b>/);
  for (const gone of [/RTX 5070/, /graphics memory · /, /Qwen3\.8-Flash-Next/, /Loaded/, /It takes about/, /<dl/, /<dt>/, /SSD/, /pc-loaded/]) assert.doesNotMatch(fits, gone, String(gone));
  // A loaded model that really streams from the SSD keeps how fast it reads right now; nothing else comes back.
  const spills = card({ fit: fit(18), live: { ...live, diskReadGbs: 1.2 }, loaded: [{ id: 'q', label: 'Qwen3.8-Flash-Next · UD-Q4_K_XL', fit: fit(18, 12, 53) }], selectedLoaded: true });
  assert.doesNotMatch(spills, /About 18 GB will be read/);
  assert.match(spills, /Reading from your SSD now: 1\.2 GB\/s/);
  for (const gone of [/<dl/, /<dt>/, /Loaded/, /It takes about/, /RTX 5070/]) assert.doesNotMatch(spills, gone, String(gone));
  assert.doesNotMatch(spills, /This version would take/);
});

test('before anything is loaded the card says what the chosen version would take', () => {
  setTranslationLanguage('en');
  assert.match(card({ fit: fit(0), live }), /This version would take about 12 GB of graphics memory and 44 GB of RAM\./);
});

test('the numbers come from the host\'s readings, polled while the page is open and read without redrawing the Models page', () => {
  const wrapper = readFileSync(fileURLToPath(new URL('../src/components/PcComputer.tsx', import.meta.url)), 'utf8');
  assert.match(wrapper, /api\.get\('\/api\/monitor'\)/);
  assert.match(wrapper, /useStore\(s => tenth\(s\.monitor\?\.hardware\?\.vramUsedGb\)\)/);
  assert.match(wrapper, /useStore\(s => tenth\(s\.monitor\?\.hardware\?\.ramUsedGb\)\)/);
});

test('the new words are in all nine languages', () => {
  const en = JSON.parse(readFileSync(fileURLToPath(new URL('../src/locales/pick-en.json', import.meta.url)), 'utf8'));
  for (const lang of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he']) {
    const cat = JSON.parse(readFileSync(fileURLToPath(new URL(`../src/locales/pick-${lang}.json`, import.meta.url)), 'utf8'));
    for (const id of Object.keys(en)) {
      assert.ok(cat[id], `${lang} ${id}`);
      assert.deepEqual((cat[id].match(/\{\d+\}/g) ?? []).sort(), (en[id].match(/\{\d+\}/g) ?? []).sort(), `${lang} ${id} slots`);
    }
  }
});
