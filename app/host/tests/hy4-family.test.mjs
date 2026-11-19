// Tencent Hy4-Flash in the app (strata-hy4 is its own engine program): the catalog shows it as its own group, the version fit follows the author's rank-cut verdict that the
// catalog carries (strata_control/hy4_fit.py, pinned to the author's cases by terminal/tests/terminal/test_hy4_fit.py), and no Flash-Next option reaches its engine.
import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { isHy4, isHy4Family } from '../server/hy4.mjs';
import { resourceSettings, saferStart } from '../server/native-resources.mjs';
import { speedSwitchApplies } from '../server/native-settings.mjs';

const bundle = async file => {
  const built = await build({ entryPoints: [fileURLToPath(new URL(file, import.meta.url))], bundle: true, write: false, platform: 'node', format: 'esm', loader: { '.json': 'json' } });
  return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
};
const catalog = await bundle('../src/lib/native-catalog.ts');
const fit = await bundle('../src/lib/native-fit.ts');
const names = await bundle('../src/lib/native-names.ts');

const size = (build, mode, extra = {}) => ({ id: `hy4-flash-${build.toLowerCase()}`, model: build, about: '', download_gb: 130, ram_needed_gb: 62, experts_gb: 56, on_this_pc: 'fits', installed: false,
  hy4: { placement: { vram_gb: 5.08, ram_gb: 50.96, ssd_gb: 0 }, mode, recommended: false, unified: false, engram_gb: 67.58 }, ...extra });
const FAMILIES = [
  { family: 'qwen', title: 'Qwen3.8-Flash-Next', images: true, sizes: [{ id: 'iq3_s', model: 'IQ3_S' }] },
  { family: 'hy4-flash', title: 'Tencent Hy4-Flash 220B-A12B', about: 'x', images: false, sizes: [size('SQ-64', 'ram_vram_zero_ssd'), size('SQ-128', 'cross_tier_ssd')] },
];

test('the run config says it is Hy4-Flash by its arch or its program', () => {
  assert.equal(isHy4({ arch: 'hy4flash' }), true);
  assert.equal(isHy4({ exe: 'component:strata-hy4' }), true);
  assert.equal(isHy4({ exe: 'C:\\x\\strata-hy4.exe' }), true);
  assert.equal(isHy4({ exe: 'component:strata-engine', arch: 'qwen4exp' }), false);
  assert.equal(isHy4(null), false);
  assert.equal(isHy4Family('hy4-flash'), true);
  assert.equal(isHy4Family('qwen'), false);
});

test('the catalog lists Hy4-Flash as a group of its own, with no KV choice and no pictures', () => {
  const groups = catalog.nativeCatalogModels(FAMILIES);
  const hy4 = groups.find(g => g.family === 'hy4-flash');
  assert.ok(hy4, 'the group exists (a family nobody whitelisted used to be dropped)');
  assert.deepEqual(hy4.sizes.map(s => s.model), ['SQ-64', 'SQ-128']);
  assert.equal(hy4.sizes[0].sourceFamily, 'hy4-flash');
  assert.equal(hy4.images, false);
  assert.equal(catalog.hasKvChoice('hy4-flash'), false);
  assert.equal(catalog.hasKvChoice('qwen3.8'), true, 'the Qwen3.8 group keeps its select');
  assert.equal(groups.find(g => g.family === 'qwen3.8').sizes.length, 1, 'the Qwen group is unchanged');
});

test('the short name and the version labels', () => {
  assert.equal(names.nativeShortName('hy4-flash-sq-64'), 'Hy4-Flash SQ-64');
  assert.equal(catalog.nativeVersionLabel('SQ-64'), 'Flagship');
});

test('the version fit is the catalog verdict of the rank-cut rule', () => {
  const hw = { ram_gb: 64, vram_gb: 12 };
  assert.equal(fit.versionFit(size('SQ-64', 'ram_vram_zero_ssd'), hw), 'fits');
  assert.equal(fit.versionFit(size('SQ-32', 'near_resident_ssd'), hw), 'fits');
  assert.equal(fit.versionFit(size('SQ-128', 'cross_tier_ssd'), hw), 'ssd');
});

test('no Flash-Next option is written for the Hy4-Flash engine', () => {
  const config = { arch: 'hy4flash', exe: 'component:strata-hy4', args: ['--native', 'm.strata', '--engram', 'e.strata', '--max-context', '32768'] };
  const out = resourceSettings(config, { cpu: { cores: 16 }, ram_gb: 64 }, 'work', { arenaGb: 56 }, {});
  assert.deepEqual(out.args, config.args);
  for (const flag of ['--vram-reserve-mib', '--pool-workers', '--resident-budget-gib']) assert.ok(!out.args.includes(flag), flag);
  assert.equal(out.env, undefined);
  assert.equal(saferStart(config, { ram_gb: 64 }, {}, 'out of memory'), null);
  const qwen = resourceSettings({ args: ['--native', 'q.gguf'] }, { cpu: { cores: 16 }, ram_gb: 64 }, 'work', { arenaGb: 56 }, {});
  assert.ok(qwen.args.includes('--vram-reserve-mib'), 'the Strata engine still gets them');
});

test('the environment speed switches do not apply to it', () => {
  for (const id of ['faster_long_prompts', 'faster_12gb_nvidia', 'route_tail_skip', 'faster_radeon_9000']) {
    assert.equal(speedSwitchApplies(id, { backend: 'cuda', arch: 'hy4flash', gpus: [{ vram_gb: 12, vendor: 'nvidia' }] }), false, id);
  }
  assert.equal(speedSwitchApplies('route_tail_skip', { backend: 'cuda', gpus: [] }), true, 'the Strata engine keeps its switch');
});
