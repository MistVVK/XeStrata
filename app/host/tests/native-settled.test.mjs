import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const compiled = await build({ stdin: { contents: "export * from './src/lib/native';", resolveDir: process.cwd() }, bundle: true, write: false, platform: 'node', format: 'esm' });
const lib = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

test('the Models page waits for the first native answer before it picks a model (#3740)', async () => {
  let release;
  globalThis.fetch = () => new Promise(resolve => { release = () => resolve({ ok: true, status: 200, statusText: 'OK', text: async () => JSON.stringify({ available: true, jobs: [], instances: [] }) }); });
  let heard = 0;
  const off = lib.subscribeNative(() => { heard++; });
  assert.equal(lib.nativeSettled(), false, 'nothing read yet: show a placeholder, not the Starter');
  const reading = lib.refreshNative();
  assert.equal(lib.nativeSettled(), false);
  release();
  await reading;
  assert.equal(lib.nativeSettled(), true);
  assert.ok(heard >= 1, 'the page is told');
  off();
});
