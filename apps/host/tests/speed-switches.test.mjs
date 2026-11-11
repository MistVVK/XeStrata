// The 0.2.7 speed features have a switch in Expert settings (the model's "Native engine settings"). "Faster long prompts" (Qwen KVA) is OFF by default (owner
// decision, 26 October) and STRATA_KVA=1 turns it on; the Radeon RX 9000 and DeepSeek ones are on by default and turning them off writes their entries.
// Back in its default a switch removes its entries, so a default launch carries none of them. strata_control/speed_switches.py says the same for the
// Terminal (tests/terminal/test_speed_switches.py).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SPEED_SWITCHES, speedSwitchApplies, speedSwitchIsSet, speedSwitchIsOn, speedSwitchRows, setSpeedSwitch, validateNativeSettings, publicNativeEnv, restoreNativeEnv } from '../server/native-settings.mjs';
import { nativeFixture } from './helpers/native-fixture.mjs';

const read = rel => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8').replace(/\r\n/g, '\n');
const ENV = {      // what each switch writes outside its default
  faster_long_prompts: { STRATA_KVA: '1' },
  faster_12gb_nvidia: { STRATA_ROUTE_PRIOR: '1.5' },
  faster_radeon_9000: { STRATA_PCIE_FRAC_DEFAULT: 'old' },
  deepseek_speedups: { DS4_DEFAULTS: '026', DS4_FAST_ATTN: '0', DS4_ATTN_SLAB: '0', DS4_COPY_PIN: '0', DS4_COPY_HUGETLB: '0', DS4_STORE_SPARE: '0', DS4_STORE_PUT: '0' },
  picture_reader: { STRATA_VISION_TOWER: 'mtmd' },
};
const SIX = ['DS4_FAST_ATTN', 'DS4_ATTN_SLAB', 'DS4_COPY_PIN', 'DS4_COPY_HUGETLB', 'DS4_STORE_SPARE', 'DS4_STORE_PUT'];
const DEFAULT = { faster_long_prompts: 'off', faster_12gb_nvidia: 'off', faster_radeon_9000: 'on', deepseek_speedups: 'on', picture_reader: 'on' };
const RTX3060 = { index: 0, name: 'NVIDIA GeForce RTX 3060', vendor: 'nvidia', arch: '86', vram_gb: 12 };
const R9700 = { index: 0, name: 'AMD Radeon AI PRO R9700', vendor: 'amd', arch: 'gfx1201', vram_gb: 32 };
const RX7900 = { index: 0, name: 'AMD Radeon RX 7900 XTX', vendor: 'amd', arch: 'gfx1100', vram_gb: 24 };
const RTX = { index: 0, name: 'NVIDIA GeForce RTX 4090', vendor: 'nvidia', arch: '89', vram_gb: 24 };
const STRIX = { index: 0, name: 'AMD Radeon 8060S', vendor: 'amd', arch: 'gfx1151', vram_gb: 96, uma: true };

test('there are exactly five switches in the app, the picture reader among them (the Terminal has one more); each writes the engine names of the contract and has its default (KVA and the 12 GB routing off, the other two on)', () => {
  assert.deepEqual(SPEED_SWITCHES.map(s => s.id), Object.keys(ENV));
  for (const sw of SPEED_SWITCHES) {
    assert.deepEqual(sw.env, ENV[sw.id], sw.id);
    assert.equal(sw.default, DEFAULT[sw.id], sw.id);
  }
});

test('the Terminal\'s Qwen "0.2.6 behaviour" switch is not in the app', () => {
  assert.ok(!SPEED_SWITCHES.some(s => s.env.STRATA_DEFAULTS || /0\.2\.6 behaviour/.test(s.label)));
  assert.ok(!Object.values(SPEED_SWITCHES.flatMap(s => Object.keys(s.env))).includes('STRATA_DEFAULTS'));
});

test('Faster answers on 12 GB NVIDIA cards: only for Qwen on a discrete CUDA card of 13 GiB or less; off by default since 0.2.8 (#4402), On writes STRATA_ROUTE_PRIOR=1.5', () => {
  const ids = context => speedSwitchRows({}, context).map(r => r.id);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX3060] }), ['faster_long_prompts', 'faster_12gb_nvidia']);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [{ ...RTX3060, vram_gb: 13 }] }), ['faster_long_prompts', 'faster_12gb_nvidia'], '13 GiB counts');
  assert.deepEqual(ids({ backend: 'cuda', gpus: [{ ...RTX3060, vram_gb: 13.5 }] }), ['faster_long_prompts']);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [{ ...RTX3060, vram_gb: 16 }] }), ['faster_long_prompts'], 'a 16 GB card is not a 12 GB card');
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX] }), ['faster_long_prompts']);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX, RTX3060] }), ['faster_long_prompts', 'faster_12gb_nvidia'], 'one small card among others');
  assert.deepEqual(ids({ backend: 'cuda', gpus: [{ ...RTX3060, uma: true }] }), ['faster_long_prompts'], 'not a chip that shares system memory');
  assert.deepEqual(ids({ backend: 'cuda', gpus: [{ ...RTX3060, integrated: true }] }), ['faster_long_prompts']);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [{ ...RTX3060, usable: false }] }), ['faster_long_prompts']);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [{ name: 'GeForce RTX 3060', vram_gb: 12 }] }), ['faster_long_prompts', 'faster_12gb_nvidia'], 'no vendor reported: the name decides');
  assert.deepEqual(ids({ backend: 'cuda', gpus: [{ name: 'Mystery card', vram_gb: 12 }] }), ['faster_long_prompts']);
  assert.deepEqual(ids({ backend: 'hip', gpus: [{ name: 'AMD Radeon RX 7700 XT', vendor: 'amd', arch: 'gfx1101', vram_gb: 12 }] }), ['faster_long_prompts'], 'not a Radeon');
  assert.deepEqual(ids({ backend: 'sycl', gpus: [RTX3060] }), [], 'not a CUDA build');
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX3060], arch: 'qwen35moe' }), [], 'Qwen3.6-35B-A3B: the routing needs 512 experts');
  assert.deepEqual(ids({ deepseek: true, backend: 'cuda', gpus: [RTX3060] }), ['deepseek_speedups']);
  // a default launch carries nothing: the engine then runs as 0.2.6 did (STRATA_ROUTE_PRIOR unset = 0)
  assert.equal(speedSwitchIsOn({}, 'faster_12gb_nvidia'), false);
  assert.equal(speedSwitchRows({}, { backend: 'cuda', gpus: [RTX3060] }).find(r => r.id === 'faster_12gb_nvidia').on, false);
  assert.deepEqual(setSpeedSwitch({ PATH: '/bin' }, 'faster_12gb_nvidia', false), { PATH: '/bin' }, 'Off is the default: it writes nothing');
  // the explicit opt-in the row writes
  const on = setSpeedSwitch({ PATH: '/bin' }, 'faster_12gb_nvidia', true);
  assert.deepEqual(on, { PATH: '/bin', STRATA_ROUTE_PRIOR: '1.5' });
  assert.equal(speedSwitchIsOn(on, 'faster_12gb_nvidia'), true);
  assert.deepEqual(setSpeedSwitch(on, 'faster_12gb_nvidia', false), { PATH: '/bin' }, 'back to Off removes the entry');
  // the 0.2.7 "Off" wrote STRATA_ROUTE_PRIOR=0: that file is in the default state now, and On replaces it with 1.5
  assert.equal(speedSwitchIsOn({ STRATA_ROUTE_PRIOR: '0' }, 'faster_12gb_nvidia'), false);
  assert.deepEqual(setSpeedSwitch({ STRATA_ROUTE_PRIOR: '0' }, 'faster_12gb_nvidia', true), { STRATA_ROUTE_PRIOR: '1.5' });
  assert.deepEqual(setSpeedSwitch({ STRATA_ROUTE_PRIOR: '0' }, 'faster_12gb_nvidia', false), {}, 'and Off clears it');
  // the words say both sides
  const hint = SPEED_SWITCHES.find(s => s.id === 'faster_12gb_nvidia').hint;
  assert.match(hint, /faster on 12 GB cards/);
  assert.match(hint, /Off by default/);
  assert.match(hint, /long agent sessions it can repeat itself/);
});

test('DeepSeek speed-ups Off writes DS4_DEFAULTS=026 and the six; On removes all seven; the six alone (an earlier test build) still read as Off', () => {
  const env = { DS4_ASYNC_T: '4' };
  const off = setSpeedSwitch(env, 'deepseek_speedups', false);
  assert.deepEqual(Object.keys(off).slice(0, 2), ['DS4_ASYNC_T', 'DS4_DEFAULTS'], 'the master switch first');
  assert.deepEqual(off, { DS4_ASYNC_T: '4', DS4_DEFAULTS: '026', ...Object.fromEntries(SIX.map(k => [k, '0'])) });
  assert.equal(speedSwitchIsOn(off, 'deepseek_speedups'), false);
  assert.deepEqual(setSpeedSwitch(off, 'deepseek_speedups', true), env);
  const six = { DS4_ASYNC_T: '4', ...Object.fromEntries(SIX.map(k => [k, '0'])) };
  assert.equal(speedSwitchIsOn(six, 'deepseek_speedups'), false);
  assert.deepEqual(setSpeedSwitch(six, 'deepseek_speedups', true), env, 'turning it on removes the six too');
  assert.equal(speedSwitchIsOn({ DS4_DEFAULTS: '026' }, 'deepseek_speedups'), true, 'the master switch alone is a hand-edited file: left alone');
  assert.deepEqual(speedSwitchRows(six, { deepseek: true }).map(r => [r.id, r.on, Object.keys(r.legacy).length]), [['deepseek_speedups', false, 6]], 'the client gets the six to recognise and remove');
});

test('a default launch has none of the new environment entries: KVA reads as off, the others as on, and an environment in its default stays as it is', () => {
  for (const sw of SPEED_SWITCHES) assert.equal(speedSwitchIsSet({}, sw.id), false);
  const env = { PATH: '/native/bin', STRATA_PREFILL: '2' };
  for (const sw of SPEED_SWITCHES) assert.deepEqual(setSpeedSwitch(env, sw.id, sw.default === 'on'), env, 'the default state removes nothing that is not the switch\'s');
  const rows = speedSwitchRows({}, { backend: 'hip', gpus: [R9700] });
  assert.deepEqual(rows.map(r => [r.id, r.on]), [['faster_long_prompts', false], ['faster_radeon_9000', true]]);
  assert.deepEqual(rows.map(r => r.default), ['off', 'on']);
});

test('Faster long prompts: On writes STRATA_KVA=1, Off removes it, and an earlier test build\'s STRATA_KVA=0 shows as Off', () => {
  const env = { PATH: '/native/bin' };
  assert.deepEqual(setSpeedSwitch(env, 'faster_long_prompts', true), { PATH: '/native/bin', STRATA_KVA: '1' });
  assert.deepEqual(setSpeedSwitch({ ...env, STRATA_KVA: '1' }, 'faster_long_prompts', false), env);
  assert.equal(speedSwitchIsOn({ STRATA_KVA: '1' }, 'faster_long_prompts'), true);
  assert.equal(speedSwitchIsOn({ STRATA_KVA: '0' }, 'faster_long_prompts'), false, 'stored by the off switch of an earlier build: harmless, and Off is what it is now');
  assert.equal(speedSwitchIsOn({}, 'faster_long_prompts'), false);
  assert.deepEqual(setSpeedSwitch({ ...env, STRATA_KVA: '0' }, 'faster_long_prompts', false), env, 'choosing Off again cleans it up');
  assert.deepEqual(speedSwitchRows({ STRATA_KVA: '0' }, { backend: 'cuda', gpus: [RTX] }).map(r => [r.id, r.on]), [['faster_long_prompts', false]]);
});

test('each switch outside its default yields exactly its entries and nothing else; back in the default gives back the same environment', () => {
  const env = { PATH: '/native/bin', STRATA_PREFILL: '2', DS4_ASYNC_T: '4' };
  for (const sw of SPEED_SWITCHES) {
    const other = setSpeedSwitch(env, sw.id, sw.default !== 'on');          // the non-default state: KVA on, the others off
    assert.deepEqual(other, { ...env, ...ENV[sw.id] }, sw.id);
    assert.equal(speedSwitchIsSet(other, sw.id), true);
    assert.equal(speedSwitchIsOn(other, sw.id), sw.default !== 'on');
    for (const another of SPEED_SWITCHES) if (another.id !== sw.id) assert.equal(speedSwitchIsSet(other, another.id), false, `${sw.id} does not touch ${another.id}`);
    const back = setSpeedSwitch(other, sw.id, sw.default === 'on');
    assert.deepEqual(back, env, sw.id);
    assert.equal(JSON.stringify(back), JSON.stringify(env), 'byte for byte, key order included');
  }
  assert.throws(() => setSpeedSwitch({}, 'nothing', false), /Unknown speed switch/);
});

test('a hand-edited file with only some of a switch\'s entries reads as its default and is left alone by an unchanged save', () => {
  const partial = { DS4_FAST_ATTN: '0' };
  assert.equal(speedSwitchIsOn(partial, 'deepseek_speedups'), true);
  assert.deepEqual(setSpeedSwitch(partial, 'faster_long_prompts', false), partial);
  assert.equal(speedSwitchIsOn({ STRATA_KVA: 'yes' }, 'faster_long_prompts'), false, 'only 1 turns it on');
});

test('a switch shows only where it applies: Qwen on CUDA or HIP, Qwen on a Radeon RX 9000 / R9700, DeepSeek', () => {
  const ids = context => speedSwitchRows({}, context).map(r => r.id);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX] }), ['faster_long_prompts']);
  assert.deepEqual(ids({ backend: 'hip', gpus: [RX7900] }), ['faster_long_prompts']);
  assert.deepEqual(ids({ backend: 'hip', gpus: [STRIX] }), ['faster_long_prompts'], 'Strix Halo (gfx1151) is not gfx12');
  assert.deepEqual(ids({ backend: 'hip', gpus: [R9700] }), ['faster_long_prompts', 'faster_radeon_9000']);
  assert.deepEqual(ids({ backend: 'hip', gpus: [{ name: 'AMD Radeon RX 9070 XT', vendor: 'amd' }] }), ['faster_long_prompts', 'faster_radeon_9000'], 'no arch reported: the card name decides');
  assert.deepEqual(ids({ backend: 'hip', gpus: [{ name: 'AMD Radeon RX 7900 XT', vendor: 'amd' }] }), ['faster_long_prompts']);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX, R9700] }), ['faster_long_prompts'], 'the Radeon switch needs the HIP engine');
  for (const backend of ['sycl', 'metal', 'cpu', null]) assert.deepEqual(ids({ backend, gpus: [R9700] }), [], `Qwen on ${backend}`);
  assert.deepEqual(ids({ deepseek: true, backend: 'hip', gpus: [R9700] }), ['deepseek_speedups'], 'DeepSeek has its own switch only');
  assert.deepEqual(ids({ deepseek: true, backend: 'cuda', gpus: [RTX] }), ['deepseek_speedups']);
  assert.equal(speedSwitchApplies('deepseek_speedups', { deepseek: false, backend: 'cuda', gpus: [RTX] }), false);
  assert.deepEqual(ids({}), [], 'nothing is known: nothing is offered');
});

test('Faster long prompts is for Qwen3.8-Flash-Next only: no row for Qwen3.6-35B-A3B (a run config with an arch) or DeepSeek', () => {
  const ids = context => speedSwitchRows({}, context).map(r => r.id);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX], arch: null }), ['faster_long_prompts'], 'Flash-Next files have no arch');
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX], arch: 'qwen35moe' }), [], 'Qwen3.6-35B-A3B on CUDA');
  assert.deepEqual(ids({ backend: 'hip', gpus: [RX7900], arch: 'qwen35moe' }), [], 'and on HIP');
  assert.deepEqual(ids({ backend: 'hip', gpus: [R9700], arch: 'qwen35moe' }), ['faster_radeon_9000'], 'the Radeon switch is not tied to the model\'s shape');
  assert.equal(speedSwitchApplies('faster_long_prompts', { backend: 'cuda', gpus: [RTX], arch: 'qwen35moe' }), false);
});

test('the picture reader: offered for a Qwen model that reads pictures, in the Image reading group, with Strata\'s own and llama.cpp\'s as the two options', () => {
  const ids = context => speedSwitchRows({}, context).map(r => r.id);
  for (const backend of ['cuda', 'hip', 'sycl', 'metal', 'cpu', null]) assert.ok(ids({ backend, gpus: [R9700], vision: true }).includes('picture_reader'), `${backend}`);
  assert.ok(!ids({ backend: 'cuda', gpus: [RTX] }).includes('picture_reader'), 'pictures off');
  assert.ok(!ids({ deepseek: true, backend: 'cuda', gpus: [RTX], vision: true }).includes('picture_reader'), 'DeepSeek reads pictures in its own engine');
  const sw = SPEED_SWITCHES.find(s => s.id === 'picture_reader');
  assert.deepEqual([sw.group, sw.default, sw.words], ['Image reading', 'on', { on: 'Strata\'s own (recommended)', off: 'llama.cpp\'s' }]);
  assert.deepEqual(setSpeedSwitch({ PATH: '/bin' }, 'picture_reader', false), { PATH: '/bin', STRATA_VISION_TOWER: 'mtmd' });
  assert.deepEqual(setSpeedSwitch({ PATH: '/bin', STRATA_VISION_TOWER: 'mtmd' }, 'picture_reader', true), { PATH: '/bin' });
  const screen = read('../src/components/NativeSettings.tsx');
  assert.match(screen, /options=\{sw\.words \? \[\{ value: sw\.default, label: sw\.words\[sw\.default\] \}/, 'the two option names, the default first');
  assert.match(screen, /\{tr\(switchRows\.filter\(sw => switchGroup\(sw\) === group\)\.map\(switchRow\)\)\}<\/section>/, 'a switch of a field group is shown in that group');
});

test('the environment names round-trip through the validator, the public view and the restore, and other names stay untouched', () => {
  const all = Object.assign({}, ...Object.values(ENV));
  assert.deepEqual(validateNativeSettings({ env: all }), { env: all });
  assert.throws(() => validateNativeSettings({ env: { HOME: '/tmp' } }));
  assert.throws(() => validateNativeSettings({ env: { DS4_fast: '0' } }));
  const previous = { PATH: '/native/bin', HIP_VISIBLE_DEVICES: '0', ...all };
  assert.deepEqual(publicNativeEnv(previous), all);
  assert.deepEqual(restoreNativeEnv({}, previous), { PATH: '/native/bin', HIP_VISIBLE_DEVICES: '0' }, 'every switch back in its default removes only theirs');
  assert.deepEqual(restoreNativeEnv(all, { PATH: '/native/bin' }), { PATH: '/native/bin', ...all });
});

function setConfig(f, model, patch) {
  const path = join(f.repo, `strata-${model}.json`);
  writeFileSync(path, JSON.stringify({ ...JSON.parse(readFileSync(path, 'utf8')), ...patch }));
  return path;
}

test('through the host: the settings answer lists the switches that apply, and KVA on then off gives back the same file', async () => {
  const f = nativeFixture();
  try {
    f.setHardware({ ram_gb: 64, gpus: [R9700] });
    const path = setConfig(f, 'iq2_xs', { backend: 'hip' });
    await f.control.refresh();
    const before = readFileSync(path, 'utf8');
    const first = f.control.settings('iq2_xs');
    assert.deepEqual(first.speedSwitches.map(s => [s.id, s.on]), [['faster_long_prompts', false], ['faster_radeon_9000', true]]);
    assert.deepEqual(first.speedSwitches.map(s => [s.default, s.env]), [['off', ENV.faster_long_prompts], ['on', ENV.faster_radeon_9000]], 'the client writes what the host says');
    assert.equal('STRATA_KVA' in first.values.env, false, 'default: nothing in the environment');

    // the client turns "Faster long prompts" on: it sends the environment with the entry added
    const off = f.control.save('iq2_xs', { values: { env: { ...first.values.env, ...ENV.faster_long_prompts } } });
    const file = JSON.parse(readFileSync(path, 'utf8'));
    assert.equal(file.env.STRATA_KVA, '1');
    assert.equal(file.env.PATH, '/native/bin');
    assert.equal(file.env.STRATA_TOKEN, 'private', 'a credential is kept, not lost behind its mask');
    assert.equal('STRATA_PCIE_FRAC_DEFAULT' in file.env, false);
    assert.deepEqual(off.speedSwitches.map(s => [s.id, s.on]), [['faster_long_prompts', true], ['faster_radeon_9000', true]]);

    // and Off again: the entry is gone and the file is as it was
    const env = { ...off.values.env };
    delete env.STRATA_KVA;
    const on = f.control.save('iq2_xs', { values: { env } });
    assert.deepEqual(on.speedSwitches.map(s => s.on), [false, true]);
    const after = JSON.parse(readFileSync(path, 'utf8'));
    assert.deepEqual(after, JSON.parse(before));
    assert.equal(JSON.stringify(after.env), JSON.stringify(JSON.parse(before).env), 'the same environment, same order');
  } finally { await f.cleanup(); }
});

test('through the host: a DeepSeek model gets its own switch only, and the Strix async entry is kept beside it', async () => {
  const f = nativeFixture();
  try {
    f.setHardware({ ram_gb: 128, gpus: [STRIX] });
    const path = setConfig(f, 'iq2_xs', { arch: 'deepseek41', backend: 'hip', env: { DS4_ASYNC_T: '4' } });
    await f.control.refresh();
    const first = f.control.settings('iq2_xs');
    assert.deepEqual(first.speedSwitches.map(s => [s.id, s.on]), [['deepseek_speedups', true]]);
    assert.equal(first.values.env.DS4_ASYNC_T, '4', 'setup\'s own entry is visible and kept');
    const off = f.control.save('iq2_xs', { values: { env: { ...first.values.env, ...ENV.deepseek_speedups } } });
    assert.deepEqual(JSON.parse(readFileSync(path, 'utf8')).env, { DS4_ASYNC_T: '4', ...ENV.deepseek_speedups });
    assert.equal(off.speedSwitches[0].on, false);
    const env = Object.fromEntries(Object.entries(off.values.env).filter(([name]) => !(name in ENV.deepseek_speedups)));
    f.control.save('iq2_xs', { values: { env } });
    assert.deepEqual(JSON.parse(readFileSync(path, 'utf8')).env, { DS4_ASYNC_T: '4' });
  } finally { await f.cleanup(); }
});

test('through the host: Qwen3.6-35B-A3B (its run config has an arch) gets no Faster long prompts row, a Flash-Next model does', async () => {
  const f = nativeFixture();
  try {
    f.setHardware({ ram_gb: 64, gpus: [RTX] });
    setConfig(f, 'iq2_xs', { backend: 'cuda' });
    setConfig(f, 'iq3_s', { backend: 'cuda', arch: 'qwen35moe', model_name: 'qwen3.6-35b-a3b-ud-q4_k_xl' });
    await f.control.refresh();
    assert.deepEqual(f.control.settings('iq2_xs').speedSwitches.map(s => s.id), ['faster_long_prompts']);
    assert.deepEqual(f.control.settings('iq3_s').speedSwitches, []);
  } finally { await f.cleanup(); }
});

test('through the host: a Qwen model on a PC the engine cannot speed up offers no switch (SYCL, a Mac, no card)', async () => {
  const f = nativeFixture();
  try {
    f.setHardware({ ram_gb: 32, gpus: [{ index: 0, name: 'Intel Arc A750', vendor: 'intel', vram_gb: 8 }] });
    setConfig(f, 'iq2_xs', { backend: 'sycl' });
    await f.control.refresh();
    assert.deepEqual(f.control.settings('iq2_xs').speedSwitches, []);
  } finally { await f.cleanup(); }
});

test('the Expert settings show the rows in the existing row layout with an On / Off choice (Off recommended for KVA), and the words are in all nine languages', () => {
  const screen = read('../src/components/NativeSettings.tsx');
  assert.match(screen, /<h3 className="h4">\{tr\("Speed"\)\}<\/h3>/);
  assert.match(screen, /<label className="native-setting-row" key=\{sw\.id\}><span><strong>\{tr\(sw\.label\)\}<\/strong><small>\{tr\(sw\.hint\)\}<\/small><\/span><Select /, 'the same row, the same Select');
  assert.match(screen, /\[\{ value: 'on', label: 'On \(recommended\)' \}, \{ value: 'off', label: 'Off' \}\] : \[\{ value: 'off', label: 'Off \(recommended\)' \}, \{ value: 'on', label: 'On' \}\]/);
  assert.match(screen, /if \(write\) env\[name\] = value; else delete env\[name\];/, 'the default removes the entries');
  const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr'];
  const catalog = lang => JSON.parse(read(`../src/locales/fix-${lang}.json`));
  const words = [...SPEED_SWITCHES.flatMap(s => [s.label, s.hint, ...Object.values(s.words ?? {})]), 'On (recommended)', 'Off (recommended)'];
  const english = Object.values(catalog('en'));
  for (const word of words) assert.ok(english.includes(word), `English catalog: ${word}`);
  for (const lang of languages.slice(1)) {
    const translated = Object.values(catalog(lang));
    const en = catalog('en');
    for (const [id, text] of Object.entries(en)) if (words.includes(text)) {
      assert.ok(catalog(lang)[id] && catalog(lang)[id] !== text, `${lang} ${id} is translated`);
      assert.ok(translated.includes(catalog(lang)[id]));
      assert.ok(!/\{\d+\}/.test(catalog(lang)[id]));
    }
  }
  for (const lang of ['zh', 'ja', 'ko']) assert.ok(Object.values(catalog(lang)).some(v => v.includes('0.2.6')), 'the version number is kept in the DeepSeek text');
});

test('the effect texts say what the feature does and what the other choice costs, in plain words', () => {
  for (const sw of SPEED_SWITCHES) {
    assert.ok(sw.words ? sw.hint.includes(`Choose ${sw.words.off}`) : sw.default === 'off' ? /Off by default/.test(sw.hint) : /[Tt]urn off/.test(sw.hint), sw.id);
    assert.ok(sw.hint.length < (sw.id === 'picture_reader' ? 300 : 260), sw.id);   // the picture reader: the owner's approved help line (11-11), 275 characters
    assert.ok(!/STRATA_|DS4_|env|--/.test(sw.hint), 'no flags or variable names for a non-technical reader');
  }
  assert.equal(SPEED_SWITCHES[0].hint, 'Reads very long prompts faster by approximating part of the work. Off by default: answers about long documents can miss or mix up details.');
});
