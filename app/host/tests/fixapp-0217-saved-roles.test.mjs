// 0217 lane FIX-APP, #6117: the Graphics cards box showed "Recommended" (both cards) while the Load kept the one card the model's run config named, and the box
// flipped to "Your choice" afterwards. The page now shows the roles the run config holds (gpu_roles); the server hands them over with the installed model.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { transform } from 'esbuild';
import { nativeFixture } from './helpers/native-fixture.mjs';

const { code } = await transform(await readFile(new URL('../src/lib/gpu-roles.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const R = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const CARDS = [{ index: 0, name: 'RTX 3060', vram_gb: 12 }, { index: 1, name: 'RTX 5060 Ti', vram_gb: 16 }];

test('savedRoles reads the roles the run config holds, for the cards this PC has', () => {
  assert.deepEqual(R.savedRoles({ 0: 'main', 1: 'off' }, CARDS), { 0: 'main', 1: 'off' });
  assert.deepEqual(R.savedRoles({ 0: 'helper', 1: 'main' }, CARDS), { 0: 'helper', 1: 'main' });
});

test('savedRoles is null when the file has none, names the wrong cards, or has no single Main', () => {
  assert.equal(R.savedRoles(undefined, CARDS), null);
  assert.equal(R.savedRoles({ 0: 'main' }, CARDS), null);                          // a card of this PC is missing from the file
  assert.equal(R.savedRoles({ 0: 'main', 1: 'main' }, CARDS), null);               // two Mains
  assert.equal(R.savedRoles({ 0: 'helper', 1: 'off' }, CARDS), null);              // no Main
  assert.equal(R.savedRoles({ 0: 'main', 1: 'sideways' }, CARDS), null);
  assert.equal(R.savedRoles({ 0: 'main' }, CARDS.slice(0, 1)), null);              // one card: no roles box
});

test('the installed model carries the roles of its run config', async () => {
  const f = nativeFixture();
  try {
    const path = join(f.repo, 'strata-iq2_xs.json');
    const config = JSON.parse(await readFile(path, 'utf8'));
    await writeFile(path, JSON.stringify({ ...config, gpu: 1, gpu_roles: { 0: 'off', 1: 'main' } }));
    await f.control.refresh(true);
    const entry = f.control.state().installed.models.find(m => m.model === 'iq2_xs');
    assert.deepEqual(entry.gpuRoles, { 0: 'off', 1: 'main' });
    await writeFile(path, JSON.stringify({ ...config }));
    await f.control.refresh(true);
    assert.equal(f.control.state().installed.models.find(m => m.model === 'iq2_xs').gpuRoles, undefined);
  } finally { await f.cleanup(); }
});
