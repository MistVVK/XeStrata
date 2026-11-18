// FIX-APP-0216 A2 (GUI-VIS-1117 Run B): for about 8 s the first-run "Your model" step showed only "Connect engine" with Continue disabled, before the version list loaded.
// The page now says it is checking what the computer can run; the Connect block stays for the real "no installation" answer.
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const here = p => fileURLToPath(new URL(p, import.meta.url));
const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const LANGS = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
const nativeBuild = await build({ entryPoints: [here('../src/lib/native.ts')], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent', alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' } });
const { catalogLoading } = await import(`data:text/javascript;base64,${Buffer.from(nativeBuild.outputFiles[0].text).toString('base64')}`);

test('loading is not "no installation": the version list is awaited with a status line', () => {
  assert.equal(catalogLoading(null), true, 'nothing read yet');
  assert.equal(catalogLoading(undefined), true);
  assert.equal(catalogLoading({ available: false, catalog: null, error: null, checking: true }), true, 'Control still starting');
  assert.equal(catalogLoading({ available: true, catalog: null, error: null, checking: false }), true, 'engine answered, catalog not yet');
  assert.equal(catalogLoading({ available: true, catalog: { families: [] }, error: null, checking: false }), false, 'the list is here');
  assert.equal(catalogLoading({ available: true, catalog: null, error: 'x', checking: false }), false, 'an error is shown, not waited for');
  assert.equal(catalogLoading({ available: false, catalog: null, error: 'Connect a local Strata installation.', checking: false }), false, 'a real no-installation answer keeps Connect engine');
});

test('the page draws the status line before the Connect block, with a live region', () => {
  const w = src('components/GuidedSetup.tsx');
  assert.match(w, /<\/> : catalogLoading\(native\) \? <div className="onboard-engine-connect" role="status"/);
  assert.ok(w.indexOf('catalogLoading(native)') < w.indexOf('tr("Connect your Strata installation")'), 'the loading line comes before the Connect block');
  assert.match(w, /\{tr\("Checking what your computer can run…"\)\}/);
  assert.match(w, /\{tr\("This takes a few seconds\."\)\}/);
});

test('the words of this lane (A1, A2, A3) are in the catalog in all ten languages', () => {
  const cat = Object.fromEntries(LANGS.map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/fixa-${l}.json`, import.meta.url), 'utf8'))]));
  for (const text of ['Checking what your computer can run…', 'This takes a few seconds.', 'Left to download', 'Downloading a small helper that makes answers faster. This happens once.']) {
    const key = Object.keys(cat.en).find(k => cat.en[k] === text);
    assert.ok(key, `${text} is in fixa-en.json`);
    for (const l of LANGS) assert.ok(cat[l][key] && cat[l][key].trim(), `${l}: ${text}`);
  }
});
