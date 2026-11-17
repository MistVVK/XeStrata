/**
 * QAGUI 0213 #2: in the first seconds after a start the engine is there but Control's installed list has not arrived, so a downloaded
 * model must not be offered as "Download & load". The per-model buttons wait (existing "Checking…" label, disabled) until the list is in.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const result = await build({
  entryPoints: [here('../src/lib/native.ts')], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
  alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' },
});
const { installedListPending } = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);

test('the list is pending only while the engine answered and the installed list is still missing', () => {
  assert.equal(installedListPending(null), false, 'nothing read yet: the buttons are already off (engine not available)');
  assert.equal(installedListPending({ available: false, installed: null, error: null }), false);
  assert.equal(installedListPending({ available: true, installed: null, error: null }), true, 'the flash: engine up, list not here');
  assert.equal(installedListPending({ available: true, installed: { models: [] }, error: null }), false, 'an empty list is an answer: Download & load is right');
  assert.equal(installedListPending({ available: true, installed: { models: [{ model: 'a', ready: true }] }, error: null }), false);
  assert.equal(installedListPending({ available: true, installed: null, error: 'boom' }), false, 'a failed read does not hold the buttons for ever');
});

test('the Models page waits on it for the download size, Tune model and the primary button, and uses the existing Checking label', () => {
  const src = readFileSync(here('../src/components/NativeModels.tsx'), 'utf8');
  assert.match(src, /listPending = installedListPending\(native\)/);
  assert.match(src, /!loadedSelected && !listPending && !installed\?\.ready && selectedSize\?\.download_gb/);
  assert.match(src, /<Btn icon="gauge" disabled=\{listPending \|\|/);
  assert.match(src, /<Btn variant="primary" icon="models" disabled=\{pending \|\| listPending \|\|/);
  assert.match(src, /tr\(pending \|\| listPending \? 'Checking…' : installed\?\.ready \? 'Load model' : 'Download & load'\)/);
});
