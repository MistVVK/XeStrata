// Tencent Hy4-Flash in the app (strata-hy4 is its own engine program): the catalog shows it as its own group, the version fit follows the author's rank-cut verdict that the
// catalog carries (strata_control/hy4_fit.py, pinned to the author's cases by terminal/tests/terminal/test_hy4_fit.py), and no Flash-Next option reaches its engine.
import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { isHy4, isHy4Family, hy4VisionShard, withoutHy4Vision } from '../server/hy4.mjs';
import { resourceSettings, saferStart } from '../server/native-resources.mjs';
import { speedSwitchApplies } from '../server/native-settings.mjs';
import { checkRunConfig } from '../server/run-config-check.mjs';
import { markPictures, runtimeVision } from '../server/native-vision.mjs';

const bundle = async file => {
  const built = await build({ entryPoints: [fileURLToPath(new URL(file, import.meta.url))], bundle: true, write: false, platform: 'node', format: 'esm', loader: { '.json': 'json' } });
  return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
};
const catalog = await bundle('../src/lib/native-catalog.ts');
const fit = await bundle('../src/lib/native-fit.ts');
const names = await bundle('../src/lib/native-names.ts');

const size = (build, mode, extra = {}) => ({ id: `hy4-flash-${build.toLowerCase()}`, model: build, about: '', download_gb: 130, ram_needed_gb: 62, experts_gb: 56, images: true, on_this_pc: 'fits', installed: false,
  hy4: { placement: { vram_gb: 5.08, ram_gb: 50.96, ssd_gb: 0 }, mode, recommended: false, unified: false, engram_gb: 67.58 }, ...extra });
const FAMILIES = [
  { family: 'qwen', title: 'Qwen3.8-Flash-Next', images: true, sizes: [{ id: 'iq3_s', model: 'IQ3_S' }] },
  { family: 'hy4-flash', title: 'Tencent Hy4-Flash', about: 'x', images: true, sizes: [size('SQ-64', 'ram_vram_zero_ssd'), size('SQ-128', 'cross_tier_ssd')] },
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

test('the catalog lists Hy4-Flash as a group of its own, with no KV choice and pictures on', () => {
  const groups = catalog.nativeCatalogModels(FAMILIES);
  const hy4 = groups.find(g => g.family === 'hy4-flash');
  assert.ok(hy4, 'the group exists (a family nobody whitelisted used to be dropped)');
  assert.deepEqual(hy4.sizes.map(s => s.model), ['SQ-64', 'SQ-128']);
  assert.equal(hy4.sizes[0].sourceFamily, 'hy4-flash');
  assert.equal(hy4.images, true, 'images are on by default (owner, 2026-11-19)');
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
  assert.equal(fit.versionFit(size('SQ-96', 'near_resident_ssd'), hw), 'fits');
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

test('run-config validation accepts Hy4-Flash drafting: dfly2 and rejects it for Qwen', () => {
  const cfg = {
    exe: 'component:strata-hy4',
    arch: 'hy4flash',
    model_name: 'hy4-flash-sq-64',
    port: 8095,
    drafting: 'dfly2',
    args: ['--native', 'C:\\m\\Hy4-Flash-SQ-64.strata', '--engram', 'C:\\m\\Hy4-Flash-Engram-Q8.strata', '--max-context', '131072'],
  };
  const checked = checkRunConfig(JSON.stringify(cfg, null, 2));
  assert.equal(checked.ok, true, JSON.stringify(checked.issues));
  const qwen = checkRunConfig(JSON.stringify({ exe: 'component:strata-engine', drafting: 'dfly2' }, null, 2));
  assert.equal(qwen.ok, false);
});

test('hy4VisionShard, withoutHy4Vision, runtimeVision, and markPictures handle built-in Vision shard cleanly', () => {
  const cfg = {
    arch: 'hy4flash',
    exe: 'component:strata-hy4',
    args: ['--native', 'C:\\m\\Hy4-Flash-SQ-64.strata', '--vision', 'C:\\m\\Hy4-Flash-Vision.strata', '--max-context', '131072'],
  };
  assert.equal(hy4VisionShard(cfg), 'C:\\m\\Hy4-Flash-Vision.strata');
  assert.deepEqual(withoutHy4Vision(cfg.args), ['--native', 'C:\\m\\Hy4-Flash-SQ-64.strata', '--max-context', '131072']);
  assert.equal(runtimeVision(cfg, 'yes'), undefined);
  assert.equal(runtimeVision(cfg, 'no'), undefined);
  const noShard = { arch: 'hy4flash', exe: 'component:strata-hy4', args: ['--native', 'C:\\m\\Hy4-Flash-SQ-64.strata'] };
  assert.equal(hy4VisionShard(noShard), null);
  assert.throws(() => runtimeVision(noShard, 'yes'), /Prepare this model/);
  const marked = markPictures({ families: FAMILIES }, false);
  const hy4Fam = marked.families.find(f => f.family === 'hy4-flash');
  const qwenFam = marked.families.find(f => f.family === 'qwen');
  assert.equal(hy4Fam.images, true);
  assert.equal(hy4Fam.sizes[0].images, true);
  assert.equal(qwenFam.images, false, 'Qwen still requires strata-vision reader');
  assert.equal(catalog.PICTURE_FETCH_FAMILIES.includes('hy4-flash'), true, 'Hy4-Flash supports on-demand Vision shard fetch');
  assert.equal(catalog.fetchesPictures({ sourceFamily: 'hy4-flash' }, true), true);
});

test('Hy4-Flash profile uses Off/High/Max efforts (max -> max) while Qwen keeps LOCAL_EFFORTS', async () => {
  const { buildProfile, LOCAL_EFFORTS, HY4_LOCAL_EFFORTS } = await import('../server/agent/profile.mjs');
  const rows = p => JSON.parse(p.patch.slice(p.patch.indexOf('[')));
  const built = buildProfile({
    local: [
      { id: 'h1', modelId: 'hy4-flash-sq-64', label: 'Tencent Hy4-Flash SQ-64', baseUrl: 'http://127.0.0.1:9001/v1', context: 131072, vision: true },
      { id: 'q1', modelId: 'iq3_s', label: 'Qwen3.8-Flash-Next IQ3_S', baseUrl: 'http://127.0.0.1:9002/v1', context: 131072, vision: false },
    ],
    connections: [],
  });
  const providers = rows(built).find(e => e.id === 'llm-pi-ai').config.providers;
  assert.deepEqual(providers['strata-local'].models[0].reasoningEfforts, HY4_LOCAL_EFFORTS);
  assert.equal(providers['strata-local'].models[0].reasoningEfforts.max, 'max');
  assert.deepEqual(providers['strata-local'].models[0].input, ['text', 'image']);
  assert.equal(built.routes[0].vision, true);
  assert.deepEqual(providers['strata-local-q1'].models[0].reasoningEfforts, LOCAL_EFFORTS);
  assert.equal(providers['strata-local-q1'].models[0].reasoningEfforts.max, 'high');
  assert.equal(built.routes[1].vision, false);
});

test('Hy4-Flash on-demand Vision fetch, Step 4 tune() vision preservation, and UI edge controls', async () => {
  const { readFileSync } = await import('node:fs');
  const read = rel => readFileSync(new URL(rel, import.meta.url), 'utf8');
  const nc = read('../server/native-control.mjs');
  const bridge = read('../server/agent/agent-bridge.mjs');
  const composer = read('../src/components/ComposerControls.tsx');
  const models = read('../src/components/NativeModels.tsx');
  const guided = read('../src/components/GuidedSetup.tsx');
  const settings = read('../src/components/NativeSettings.tsx');
  const sidebar = read('../src/app/Sidebar.tsx');
  assert.match(nc, /vision:\(isHy4\(current\.config\)\|\|isInkling\(current\.config\)\|\|current\.config\.vision\)\?args\.vision:'no'/);
  assert.match(nc, /withoutHy4Vision/);
  assert.match(bridge, /known\.vision === false && vision === true/);
  assert.match(bridge, /isHy4Route = route\.kind === 'local' && \/\^hy4-flash\/i\.test/);
  assert.match(composer, /hy4Efforts = \['Off', 'High', 'Max'\]/);
  assert.match(composer, /isHy4Active[\s\S]*'High' : 'Medium'/);
  assert.match(models, /isHy4Selected = selectedFamily\?\.family === HY4_FAMILY/);
  assert.match(models, /!isHy4Selected \? <NativeTuneBadge/);
  assert.match(models, /!isHy4Selected \? <Btn icon="gauge"/);
  assert.match(guided, /isHy4Selected = selectedFamily\?\.family === HY4_FAMILY/);
  assert.match(settings, /HY4_HIDDEN_FIELDS/);
  assert.match(settings, /isHy4 \? \['dfly2', 'off'\]/);
  assert.match(sidebar, /nativeFamily\?\.title \?\? nativeCurrent\?\.label/);
});

test('behavioral: native-control tune() and load on-demand Vision fetch keep Hy4-Flash vision: true', async () => {
  await import('./helpers/isolate-profile.mjs');
  const { EventEmitter } = await import('node:events');
  const { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } = await import('node:fs');
  const { join } = await import('node:path');
  const { tmpdir } = await import('node:os');
  const { createNativeControl } = await import('../server/native-control.mjs');

  const root = mkdtempSync(join(realpathSync(tmpdir()), 'strata-hy4-vis-'));
  const install = join(root, 'install'), data = join(root, 'host-data'), modelDir = join(root, 'data', 'models', 'hy4-flash');
  for (const d of [install, data, modelDir]) mkdirSync(d, { recursive: true });
  const prevInstall = process.env.STRATA_INSTALL_DIR;
  process.env.STRATA_INSTALL_DIR = install;
  const nativeFile = join(modelDir, 'Hy4-Flash-SQ-64.strata');
  const engramFile = join(modelDir, 'Hy4-Flash-Engram-Q8.strata');
  const visionFile = join(modelDir, 'Hy4-Flash-Vision.strata');
  writeFileSync(nativeFile, 'sq64');
  writeFileSync(engramFile, 'engram');
  const configPath = join(install, 'strata-hy4-flash-sq-64.json');
  const baseArgs = ['--native', nativeFile, '--engram', engramFile, '--max-context', '131072'];
  writeFileSync(configPath, JSON.stringify({
    exe: 'component:strata-hy4', arch: 'hy4flash', model_name: 'hy4-flash-sq-64', port: 8095,
    tokenizer: join(root, 'tok'), args: baseArgs, args_written: baseArgs,
  }));

  const savedConfigs = [], fetchCalls = [];
  const control = {
    control: { dir: join(root, 'control'), python: 'python' }, version: '0.2.18',
    async installedModels() {
      return { install_dir: install, models: [{ id: 'hy4-flash-sq-64', name: 'hy4-flash-sq-64', source_config: configPath, context_tokens: 131072, backend: 'hip', component_status: 'present', missing: [], issue_codes: [] }] };
    },
    async engine() { return { usable: true, backend: 'hip', components_dir: join(root, 'c'), engine: { version: '0.1.64', platform: 'win-x64', backend: 'hip', exe: join(root, 'engine', 'strata-hy4.exe'), root: join(root, 'engine') } }; },
    async saveProfile(_name, path) { savedConfigs.push(JSON.parse(readFileSync(path, 'utf8'))); return {}; },
    async start() { return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
    async picturesFetch(opts) {
      fetchCalls.push(opts);
      writeFileSync(visionFile, 'vision-shard');
      return { ok: true, event: 'result', mmproj: visionFile, vision: null, fetched: true, note: '', family: 'hy4-flash' };
    },
  };
  const catalog = {
    families: [{ family: 'hy4-flash', title: 'Tencent Hy4-Flash', about: 'x', images: true, sizes: [size('SQ-64', 'ram_vram_zero_ssd')] }],
    contexts: [131072], recommendation: { family: 'hy4-flash', model: 'SQ-64', context: 131072 }, context_note: '',
  };
  const spawnProcess = () => {
    const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter(); child.exitCode = null;
    child.kill = () => { child.exitCode = 0; queueMicrotask(() => child.emit('exit', 0)); return true; };
    child.stdin = {
      write(line) {
        const msg = JSON.parse(line); if (!msg.id) return;
        let result = {};
        if (msg.method === 'initialize') result = { protocolVersion: '2024-11-05' };
        else if (msg.method === 'tools/list') result = { tools: ['strata_models', 'strata_status'].map(name => ({ name })) };
        else if (msg.params?.name === 'strata_models') result = { structuredContent: catalog };
        else if (msg.params?.name === 'strata_status') result = { structuredContent: { hardware: { ram_gb: 128, cpu: { cores: 16 }, gpus: [] }, summary: 'ok' } };
        queueMicrotask(() => child.stdout.emit('data', Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result }) + '\n')));
      },
    };
    return child;
  };
  const native = createNativeControl({
    control, directory: data, spawnProcess,
    fetcher: async url => url.includes('/chat/completions')
      ? Response.json({ usage: { prompt_tokens: 10, completion_tokens: 20 }, timings: { predicted_per_second: 50, prompt_per_second: 500 } })
      : Response.json({ status: 'ok' }),
    requests: () => [], modelsDirectory: () => join(root, 'data'),
  });
  const finish = async id => {
    for (let i = 0; i < 400; i++) {
      const j = native.state().jobs.find(x => x.id === id);
      if (j && j.state !== 'running') return j;
      await new Promise(r => setTimeout(r, 10));
    }
    throw new Error('job timeout');
  };
  try {
    await native.refresh(true);
    // 1. Load without --vision in args -> ensurePictures fetches Hy4-Flash-Vision.strata, splices --vision, sets instance.vision === true
    const j1 = await finish(native.operation('load', { model: 'hy4-flash-sq-64', vision: 'yes' }).id);
    assert.equal(j1.state, 'done', j1.error);
    assert.equal(fetchCalls.length, 1);
    assert.equal(fetchCalls[0].family, 'hy4-flash');
    assert.ok(existsSync(visionFile));
    assert.equal(hy4VisionShard(savedConfigs[0]), visionFile);
    assert.equal(native.state().instances[0].vision, true);
    const persisted = JSON.parse(readFileSync(configPath, 'utf8'));
    assert.equal(hy4VisionShard(persisted), visionFile);
    assert.ok(persisted.args_written.includes('--vision'));
    // 2. Step 4 tune() on Hy4-Flash with vision: 'yes' must preserve --vision and NOT force 'no'
    const j2 = await finish(native.operation('tune', { model: 'hy4-flash-sq-64', profile: 'work', calibrate: false, vision: 'yes' }).id);
    assert.equal(j2.state, 'done', j2.error);
    assert.equal(hy4VisionShard(savedConfigs.at(-1)), visionFile);
    assert.equal(native.state().instances[0].vision, true);
  } finally {
    await native.close();
    if (prevInstall === undefined) delete process.env.STRATA_INSTALL_DIR; else process.env.STRATA_INSTALL_DIR = prevInstall;
    rmSync(root, { recursive: true, force: true });
  }
});
