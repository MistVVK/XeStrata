/**
 * Unit tests for Inkling-2-Mini Desktop GUI / Host integration:
 * - server/inkling.mjs helpers
 * - src/lib/native-catalog.ts, native-fit.ts, native-names.ts
 * - server/native-resources.mjs & server/native-settings.mjs
 * - server/run-config-check.mjs & server/native-vision.mjs
 * - server/agent/profile.mjs 6-level INKLING_LOCAL_EFFORTS (default high)
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { isInkling, isInklingFamily, inklingVisionShard, withoutInklingVision, INKLING_ARCH, INKLING_FAMILY } from '../server/inkling.mjs';
import { resourceSettings, saferStart } from '../server/native-resources.mjs';
import { speedSwitchApplies } from '../server/native-settings.mjs';
import { checkRunConfig } from '../server/run-config-check.mjs';
import { markPictures, runtimeVision } from '../server/native-vision.mjs';

const bundle = async file => {
  const built = await build({
    entryPoints: [fileURLToPath(new URL(file, import.meta.url))],
    bundle: true,
    write: false,
    platform: 'node',
    format: 'esm',
    loader: { '.json': 'json' },
  });
  return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
};
const catalog = await bundle('../src/lib/native-catalog.ts');
const fit = await bundle('../src/lib/native-fit.ts');
const names = await bundle('../src/lib/native-names.ts');

const size = (buildName, mode, extra = {}) => ({
  id: `inkling-2-mini-${buildName.toLowerCase()}`,
  model: buildName,
  about: '',
  download_gb: buildName === 'SQ-20' ? 19.35 : buildName === 'SQ-15' ? 14.35 : 10.47,
  ram_needed_gb: buildName === 'SQ-20' ? 20 : buildName === 'SQ-15' ? 15 : 11,
  experts_gb: buildName === 'SQ-20' ? 15.68 : buildName === 'SQ-15' ? 12.28 : 8.6,
  images: true,
  on_this_pc: 'fits',
  installed: false,
  inkling: { placement: { vram_gb: 15.68, ram_gb: 0, ssd_gb: 0, full_vram: mode === 'full_vram' }, mode, recommended: buildName === 'SQ-20', unified: false },
  ...extra,
});

const FAMILIES = [
  { family: 'qwen', title: 'Qwen3.8-Flash-Next', images: true, sizes: [{ id: 'iq3_s', model: 'IQ3_S' }] },
  {
    family: 'inkling-2-mini',
    title: 'Inkling-2-Mini',
    about: 'Thinking Machines Inkling-2-Mini',
    images: true,
    sizes: [size('SQ-20', 'full_vram'), size('SQ-15', 'ram_vram_zero_ssd'), size('SQ-10', 'cross_tier_ssd')],
  },
];

test('inkling.mjs detects Inkling-2-Mini by arch or executable', () => {
  assert.equal(isInkling({ arch: INKLING_ARCH }), true);
  assert.equal(isInkling({ exe: 'component:strata-inkling' }), true);
  assert.equal(isInkling({ exe: '/opt/strata/strata-inkling' }), true);
  assert.equal(isInkling({ exe: 'C:\\Strata\\strata-inkling.exe' }), true);
  assert.equal(isInkling({ exe: 'component:strata-hy4', arch: 'hy4flash' }), false);
  assert.equal(isInkling(null), false);
  assert.equal(isInklingFamily(INKLING_FAMILY), true);
  assert.equal(isInklingFamily('hy4-flash'), false);
});

test('catalog exposes Inkling-2-Mini as its own group with default-on Vision and version labels', () => {
  const groups = catalog.nativeCatalogModels(FAMILIES);
  const ink = groups.find(g => g.family === 'inkling-2-mini');
  assert.ok(ink, 'Inkling-2-Mini group exists');
  assert.deepEqual(ink.sizes.map(s => s.model), ['SQ-20', 'SQ-15', 'SQ-10']);
  assert.equal(ink.images, true);
  assert.equal(catalog.hasKvChoice('inkling-2-mini'), false);
  assert.equal(catalog.PICTURE_FETCH_FAMILIES.includes('inkling-2-mini'), true);
  assert.equal(catalog.fetchesPictures({ sourceFamily: 'inkling-2-mini' }, true), true);
  assert.equal(names.nativeShortName('inkling-2-mini-sq-20'), 'Inkling-2-Mini SQ-20');
  assert.equal(names.nativeShortName('inkling-2-mini-sq-15'), 'Inkling-2-Mini SQ-15');
  assert.equal(names.nativeShortName('inkling-2-mini-sq-10'), 'Inkling-2-Mini SQ-10');
});

test('versionFit honors inkling placement verdicts (full_vram / hybrid -> fits, cross_tier_ssd -> ssd)', () => {
  const hw = { ram_gb: 32, vram_gb: 24 };
  assert.equal(fit.versionFit(size('SQ-20', 'full_vram'), hw), 'fits');
  assert.equal(fit.versionFit(size('SQ-15', 'ram_vram_zero_ssd'), hw), 'fits');
  assert.equal(fit.versionFit(size('SQ-10', 'cross_tier_ssd'), hw), 'ssd');
});

test('no Flash-Next resource flags or speed switches touch Inkling-2-Mini', () => {
  const config = {
    arch: 'inkling2_loop',
    exe: 'component:strata-inkling',
    args: ['--native', 'Inkling-2-Mini-SQ-20.strata', '--max-context', '32768', '--parallel', '8'],
  };
  const out = resourceSettings(config, { cpu: { cores: 16 }, ram_gb: 64 }, 'work', { arenaGb: 20 }, {});
  assert.deepEqual(out.args, config.args);
  assert.equal(saferStart(config, { ram_gb: 64 }, {}, 'out of memory'), null);
  for (const id of ['faster_long_prompts', 'faster_12gb_nvidia', 'route_tail_skip', 'faster_radeon_9000']) {
    assert.equal(speedSwitchApplies(id, { backend: 'cuda', arch: 'inkling2_loop', gpus: [{ vram_gb: 24, vendor: 'nvidia' }] }), false, id);
  }
});

test('run-config-check accepts Inkling-2-Mini drafting: dspark and rejects it for Qwen', () => {
  const cfg = {
    exe: 'component:strata-inkling',
    arch: 'inkling2_loop',
    model_name: 'inkling-2-mini-sq-20',
    port: 8096,
    drafting: 'dspark',
    args: ['--native', 'C:\\m\\Inkling-2-Mini-SQ-20.strata', '--vision', 'C:\\m\\Inkling-2-Mini-Vision.strata', '--max-context', '32768', '--parallel', '8'],
  };
  const checked = checkRunConfig(JSON.stringify(cfg, null, 2));
  assert.equal(checked.ok, true, JSON.stringify(checked.issues));
  const qwen = checkRunConfig(JSON.stringify({ exe: 'component:strata-engine', drafting: 'dspark' }, null, 2));
  assert.equal(qwen.ok, false);
});

test('inklingVisionShard, withoutInklingVision, runtimeVision, and markPictures work cleanly', () => {
  const cfg = {
    arch: 'inkling2_loop',
    exe: 'component:strata-inkling',
    args: ['--native', 'C:\\m\\Inkling-2-Mini-SQ-20.strata', '--vision', 'C:\\m\\Inkling-2-Mini-Vision.strata', '--max-context', '32768'],
  };
  assert.equal(inklingVisionShard(cfg), 'C:\\m\\Inkling-2-Mini-Vision.strata');
  assert.deepEqual(withoutInklingVision(cfg.args), ['--native', 'C:\\m\\Inkling-2-Mini-SQ-20.strata', '--max-context', '32768']);
  assert.equal(runtimeVision(cfg, 'yes'), undefined);
  assert.equal(runtimeVision(cfg, 'no'), undefined);
  const noShard = { arch: 'inkling2_loop', exe: 'component:strata-inkling2', args: ['--native', 'C:\\m\\Inkling-2-Mini-SQ-20.strata'] };
  assert.equal(inklingVisionShard(noShard), null);
  assert.throws(() => runtimeVision(noShard, 'yes'), /Prepare this model/);
  const marked = markPictures({ families: FAMILIES }, false);
  const inkFam = marked.families.find(f => f.family === 'inkling-2-mini');
  assert.equal(inkFam.images, true);
  assert.equal(inkFam.sizes[0].images, true);
});

test('profile.mjs disables thinking mode for Inkling-2-Mini (off only) and run-config-check allows parallel 64', async () => {
  const { INKLING_LOCAL_EFFORTS, buildProfile } = await import('../server/agent/profile.mjs');
  assert.deepEqual(INKLING_LOCAL_EFFORTS, { off: 'none' });
  const prof = buildProfile({
    local: [{ id: 'ink', label: 'Inkling-2-Mini SQ-20', baseUrl: 'http://127.0.0.1:8080/v1', modelId: 'inkling-2-mini-sq-20', context: 32768, vision: true, slots: 8 }],
  });
  assert.match(prof.patch, /"reasoning": false/);
  assert.doesNotMatch(prof.patch, /"thinkingFormat": "chat-template"/);
  // Also verify run-config-check allows parallel up to 64 for Inkling-2-Mini
  const checked64 = checkRunConfig(JSON.stringify({
    exe: 'component:strata-inkling',
    arch: 'inkling2_loop',
    parallel: 64,
    gpu: [0, 1],
    args: ['--native', 'Inkling-2-Mini-SQ-20.strata', '--gpus', '2', '--parallel', '64'],
  }));
  assert.equal(checked64.ok, true, JSON.stringify(checked64.issues));
});
