// Owner 0.2.16: STRATA_ROUTE_TAIL_SKIP in Expert settings as a choice, on CUDA and gfx1200 / gfx1201 only.
// Owner 0.2.17 (11-19): Off is the default (no entry; every engine start sets STRATA_ROUTE_TAIL_SKIP=0), On is 7 and writes STRATA_ROUTE_TAIL_SKIP=7, the other ranks are numbers.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { SPEED_SWITCHES, speedSwitchApplies, speedSwitchIsOn, speedSwitchRows } from '../server/native-settings.mjs';
import { nativeFixture } from './helpers/native-fixture.mjs';

const { code } = await transform(await readFile(new URL('../src/lib/speed-choice.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { choiceValue, choiceOptions, isNumberChoice, withChoice } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const sw = SPEED_SWITCHES.find(s => s.id === 'route_tail_skip');

test('the switch comes from the shared table with its choices and note', () => {
  assert.ok(sw);
  assert.equal(sw.default, 'off');
  assert.deepEqual(sw.env, { STRATA_ROUTE_TAIL_SKIP: '7' });
  assert.deepEqual(sw.words, { on: 'On', off: 'Off (default)' });
  assert.deepEqual(choiceOptions(sw).map(o => [o.value, o.label]), [['', 'Off (default)'], ['7', 'On'], ...['1', '2', '3', '4', '5', '6', '8', '9'].map(v => [v, v])]);
  assert.equal(speedSwitchIsOn({}, 'route_tail_skip'), false);                    // no entry: Off
  assert.equal(speedSwitchIsOn({ STRATA_ROUTE_TAIL_SKIP: '0' }, 'route_tail_skip'), false);
  assert.equal(speedSwitchIsOn({ STRATA_ROUTE_TAIL_SKIP: '7' }, 'route_tail_skip'), true);
  assert.equal(sw.choiceNote, 'Lower numbers skip more parts and are faster; answers can change more. Only 7 has been measured.');
});

test('shown on CUDA and on a Radeon RX 9000 / AI PRO R9700 only', () => {
  assert.equal(speedSwitchApplies('route_tail_skip', { backend: 'cuda', gpus: [{ name: 'NVIDIA GeForce RTX 5070', vendor: 'nvidia', vram_gb: 12 }] }), true);
  assert.equal(speedSwitchApplies('route_tail_skip', { backend: 'hip', gpus: [{ name: 'AMD Radeon AI PRO R9700', vendor: 'amd', arch: 'gfx1201' }] }), true);
  assert.equal(speedSwitchApplies('route_tail_skip', { backend: 'hip', gpus: [{ name: 'AMD Radeon RX 9070 XT', vendor: 'amd' }] }), true);
  assert.equal(speedSwitchApplies('route_tail_skip', { backend: 'hip', gpus: [{ name: 'AMD Radeon RX 7900 XTX', vendor: 'amd', arch: 'gfx1100' }] }), false);
  for (const backend of ['metal', 'sycl', 'cpu', null]) assert.equal(speedSwitchApplies('route_tail_skip', { backend, gpus: [] }), false, String(backend));
  assert.ok(speedSwitchRows({}, { backend: 'cuda', gpus: [] }).some(r => r.id === 'route_tail_skip' && r.choices));
});

test('Off (the default) writes nothing, On writes 7, a rank writes itself; setup\'s 0 reads as Off; other entries stay', () => {
  assert.equal(choiceValue(sw, {}), '');
  assert.equal(choiceValue(sw, { STRATA_ROUTE_TAIL_SKIP: '0' }), '');             // setup's 0 (and a 0 the person wrote) is Off, the default
  assert.equal(choiceValue(sw, { STRATA_ROUTE_TAIL_SKIP: '7' }), '7');            // On
  assert.equal(choiceValue(sw, { STRATA_ROUTE_TAIL_SKIP: 12 }), '');              // a value the list does not know shows as Off
  assert.deepEqual(withChoice(sw, { STRATA_KVA: '1' }, '7'), { STRATA_KVA: '1', STRATA_ROUTE_TAIL_SKIP: '7' });
  assert.deepEqual(withChoice(sw, { STRATA_KVA: '1' }, '5'), { STRATA_KVA: '1', STRATA_ROUTE_TAIL_SKIP: '5' });
  assert.deepEqual(withChoice(sw, { STRATA_KVA: '1', STRATA_ROUTE_TAIL_SKIP: '7' }, ''), { STRATA_KVA: '1' });
  assert.deepEqual(withChoice(sw, { STRATA_KVA: '1', STRATA_ROUTE_TAIL_SKIP: '0' }, ''), { STRATA_KVA: '1' });
  assert.deepEqual(withChoice(sw, {}, '0'), {});                                    // 0 is not a choice any more: Off is the default, which writes nothing
  assert.deepEqual(withChoice(sw, {}, '42'), {});                                   // never writes a value outside the list
  assert.equal(isNumberChoice(sw, '5'), true);
  assert.equal(isNumberChoice(sw, '7'), false);                                     // On is a word
  assert.equal(isNumberChoice(sw, '0'), false);
  assert.equal(isNumberChoice(sw, ''), false);
});

test('the Expert settings row uses the choice helpers', async () => {
  const source = await readFile(new URL('../src/components/NativeSettings.tsx', import.meta.url), 'utf8');
  assert.match(source, /sw\.choices\?\.length \? choiceRow\(sw\)/);
  assert.match(source, /withChoice\(sw, values\.env, v\)/);
});

test('a model\'s settings carry the choice, and saving a rank writes it into the model\'s settings file (fixture host, RTX 4090, CUDA)', async t => {
  const f = nativeFixture();
  t.after(() => f.cleanup());
  await f.control.refresh();
  const model = 'iq2_xs', file = join(f.repo, `strata-${model}.json`);
  const config = JSON.parse(readFileSync(file, 'utf8'));
  config.backend = 'cuda';
  (await import('node:fs')).writeFileSync(file, JSON.stringify(config));
  const before = f.control.settings(model);
  const row = before.speedSwitches.find(r => r.id === 'route_tail_skip');
  assert.ok(row && row.choices, JSON.stringify(before.speedSwitches.map(r => r.id)));
  const env = withChoice(row, before.values.env, '5');
  f.control.save(model, { values: { ...before.values, env }, mcp: before.mcp ?? {} });
  const saved = JSON.parse(readFileSync(file, 'utf8')).env;
  assert.equal(saved.STRATA_ROUTE_TAIL_SKIP, '5');
  assert.equal(saved.PATH, '/native/bin');                                       // nothing else in the file's environment changes
  assert.equal(choiceValue(row, f.control.settings(model).values.env), '5');
  const on = withChoice(row, f.control.settings(model).values.env, '7');
  f.control.save(model, { values: { ...f.control.settings(model).values, env: on }, mcp: before.mcp ?? {} });
  assert.equal(JSON.parse(readFileSync(file, 'utf8')).env.STRATA_ROUTE_TAIL_SKIP, '7');   // On is written explicitly
  const off = withChoice(row, f.control.settings(model).values.env, '');
  f.control.save(model, { values: { ...f.control.settings(model).values, env: off }, mcp: before.mcp ?? {} });
  assert.equal('STRATA_ROUTE_TAIL_SKIP' in JSON.parse(readFileSync(file, 'utf8')).env, false);   // Off, the default, writes no entry
});

test('a model with no entry shows Off (default) in its settings, a stored 7 shows On', async t => {
  const f = nativeFixture();
  t.after(() => f.cleanup());
  await f.control.refresh();
  const model = 'iq2_xs', file = join(f.repo, `strata-${model}.json`);
  const config = JSON.parse(readFileSync(file, 'utf8'));
  config.backend = 'cuda';
  (await import('node:fs')).writeFileSync(file, JSON.stringify(config));
  const row = f.control.settings(model).speedSwitches.find(r => r.id === 'route_tail_skip');
  assert.equal(choiceValue(row, f.control.settings(model).values.env), '');
  assert.equal(choiceOptions(row)[0].label, 'Off (default)');
  config.env = { ...(config.env ?? {}), STRATA_ROUTE_TAIL_SKIP: '7' };
  (await import('node:fs')).writeFileSync(file, JSON.stringify(config));
  assert.equal(choiceValue(row, f.control.settings(model).values.env), '7');
});
