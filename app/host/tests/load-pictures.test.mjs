/**
 * Load with "Read pictures" on and the picture files not on this computer (the owner's bug, 11-01): the Load fetches only those files and loads.
 * It must not run the installer (no checking of the model files, no second pack, no second draft layer), must keep the model's pack, draft layer and data folder
 * exactly where its run config has them, and must never rewrite data_dir in the settings. No internet: the model loads without pictures and says so in one line.
 * Control is a fake that records what the host asked it to do; nothing here starts an engine or reaches a network.
 */
import './helpers/isolate-profile.mjs';   // a test run never writes the owner's real Strata profile
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, realpathSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir as osTmpdir } from 'node:os';
// the canonical scratch root: on a Mac the temp folder is /var/folders/..., which resolves to /private/var/...; the paths the host reports are the resolved ones
const tmpdir = () => realpathSync(osTmpdir());
import { createNativeControl } from '../server/native-control.mjs';
import { ControlError } from '../server/control-runtime.mjs';

const finish = async (native, id) => { for (let i = 0; i < 600; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };
const until = async (test, what) => { for (let i = 0; i < 600; i++) { if (test()) return; await new Promise(r => setTimeout(r, 10)); } throw new Error(`never happened: ${what}`); };
const listing = root => { const out = {}; const walk = dir => { for (const e of readdirSync(dir, { withFileTypes: true })) { const p = join(dir, e.name); if (e.isDirectory()) walk(p); else out[p.slice(root.length)] = statSync(p).size; } }; walk(root); return out; };

/**
 * layout 'own'      the model's files in their own folder, its pack and draft layer in a data folder that is not the app's models folder (the owner's case)
 * layout 'normal'   everything under <data>/models/<tag>, <data>/packs, <data>/mtp (what the app's own download makes)
 * layout 'local'    a model added with "Use a local GGUF": files in the person's folder, pack and draft layer in a Strata data folder of that day
 */
function fixture({ layout = 'own', vision = null, arch = undefined, onFetch = null, mmprojFile = false, engineRoot = null } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-lp-test-'));
  const install = join(root, 'install'), data = join(root, 'host-data'), home = join(root, 'home'), modelsDir = join(home, 'models');
  for (const d of [install, data, modelsDir]) mkdirSync(d, { recursive: true });
  const earlier = process.env.STRATA_INSTALL_DIR;
  process.env.STRATA_INSTALL_DIR = install;
  const dataFolder = layout === 'own' ? join(root, 'qwen-own') : layout === 'normal' ? join(root, 'Strata-data') : join(root, 'Strata-data');
  const modelFolder = layout === 'normal' ? join(dataFolder, 'models', 'IQ3_S') : layout === 'local' ? join(root, 'my-files', 'IQ3_S') : join(dataFolder, 'IQ3_S');
  const gguf = join(modelFolder, 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf');
  mkdirSync(modelFolder, { recursive: true });
  writeFileSync(gguf, 'GGUF'); writeFileSync(gguf.replace('00001', '00002'), 'GGUF');
  const pack = join(dataFolder, 'packs', 'iq3_s'), mtp = join(dataFolder, 'mtp', 'rt');
  mkdirSync(join(pack, 'tokenizer'), { recursive: true }); mkdirSync(mtp, { recursive: true });
  writeFileSync(join(pack, 'native_experts.txt'), 'x'); writeFileSync(join(mtp, 'experts.bin'), 'e'); writeFileSync(join(dataFolder, 'mtp', 'mtp-q2_0.gguf'), 'm');
  const mmproj = join(dataFolder, 'models', 'mmproj-Qwen3.8-Flash-Next-BF16.gguf');
  if (mmprojFile) { mkdirSync(join(dataFolder, 'models'), { recursive: true }); writeFileSync(mmproj, 'mm'); }
  const configPath = join(install, 'strata-iq3_s.json');
  const args = ['--pack', pack, '--native', gguf, '--ple-gguf', gguf.replace('00001', '00002'), '--mtp', mtp, '--max-context', '8192', '--kv', 'int8'];
  const config = { exe: 'component:strata-engine', tokenizer: join(pack, 'tokenizer'), args, model_name: 'qwen3.8-flash-next-iq3_s', port: 8080, ...(arch ? { arch } : {}),
    ...(vision ? { vision: { exe: join(root, 'engine', 'strata-vision'), mmproj, model: gguf, gpu: false, max_tokens: 300 } } : {}) };
  writeFileSync(configPath, JSON.stringify(config));
  const settingsFile = process.env.STRATA_SETTINGS_FILE;
  writeFileSync(settingsFile, JSON.stringify({ data_dir: join(root, 'host-owned', 'Strata-data'), data_dir_at: 1700000000 }));
  const calls = [], saved = [], rpc = [];
  const control = {
    control: { dir: join(root, 'control'), python: join(root, 'control', 'python', 'python') }, version: '0.2.9',
    async installedModels() {
      const gone = vision && !existsSync(mmproj);
      return { install_dir: install, models: [{ id: 'iq3_s', name: 'qwen3.8-flash-next-iq3_s', source_config: configPath, context_tokens: 8192, backend: 'cuda',
        component_status: String(arch ?? '').startsWith('deepseek') ? 'retired' : gone ? 'missing_or_invalid' : 'present', ...(String(arch ?? '').startsWith('deepseek') ? { retired: 'deepseek', retired_line: 'DeepSeek - no longer supported.', refusal: 'DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw is a DeepSeek model. This version of Strata no longer runs DeepSeek models, so it was not started. Choose a Qwen model to run instead. If you no longer need the DeepSeek files, you can delete that folder to free the space.' } : {}), ...(gone ? { missing: [{ part: 'vision/mmproj', file: 'mmproj-Qwen3.8-Flash-Next-BF16.gguf', status: 'missing' }], issue_codes: ['COMPONENT_MISSING'] } : { missing: [], issue_codes: [] }) }] };
    },
    async engine() { return { usable: true, backend: 'cuda', components_dir: join(root, 'c'), engine: { version: '0.1.55', platform: 'linux-x86_64', backend: 'cuda', exe: join(root, 'engine', 'strata'), root: engineRoot ?? join(root, 'engine') } }; },
    async saveProfile(name, path) { calls.push(['saveProfile', name]); saved.push(JSON.parse(readFileSync(path, 'utf8'))); return {}; },
    async start() { calls.push(['start']); return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
    async picturesFetch(options) {
      calls.push(['picturesFetch', options]);
      if (onFetch) return onFetch(options, { root, mmproj, gguf, calls });
      options.onEvent?.({ event: 'progress', step: 'Downloading the picture files…', done: 9e8, total: 9e8 });
      mkdirSync(join(options.dataDir, 'models'), { recursive: true });
      writeFileSync(mmproj, 'mmproj');
      return { ok: true, event: 'result', mmproj, vision: { exe: join(root, 'engine', 'strata-vision'), mmproj, model: gguf, gpu: false, max_tokens: 300, threads: 6 }, fetched: true, note: '', data_dir: options.dataFolder };
    },
  };
  const catalog = { families: [{ family: 'qwen', title: 'Qwen3.8-Flash-Next', about: 'x', images: true, experimental: false, sizes: [
    { id: 'iq3_s', model: 'IQ3_S', download_gb: 75.8, ram_needed_gb: 60, about: 'high quality', on_this_pc: 'fits', installed: true, images: true, experimental: false }] }],
    contexts: [8192, 32768], recommendation: { family: 'qwen', model: 'IQ3_S', context: 32768 }, context_note: '' };
  const hardware = { ram_gb: 63, cpu: { name: 'cpu', cores: 12 }, engine_vendor: 'nvidia', gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5070', vendor: 'nvidia', vram_gb: 11.9, arch: 120, usable: true }] };
  const spawnProcess = (command, argv) => {
    if (!String(argv[0]).endsWith('strata_mcp.py')) throw new Error('no process but the installer may be started in packaged mode: ' + command + ' ' + argv.join(' '));
    const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter(); child.exitCode = null;
    child.kill = () => { child.exitCode = 0; queueMicrotask(() => child.emit('exit', 0)); return true; };
    child.stdin = { write(line) {
      const message = JSON.parse(line); if (!message.id) return;
      if (message.method === 'tools/call') rpc.push(message.params);
      let result;
      if (message.method === 'initialize') result = { protocolVersion: '2024-11-05' };
      else if (message.method === 'tools/list') result = { tools: ['strata_models', 'strata_status', 'strata_install', 'strata_remove'].map(name => ({ name })) };
      else if (message.params.name === 'strata_models') result = { structuredContent: catalog };
      else if (message.params.name === 'strata_status') result = { structuredContent: { hardware, recommendation: catalog.recommendation, install: null, running: [], summary: 'x' } };
      else if (Object.keys(message.params.arguments ?? {}).length === 0) result = { structuredContent: { summary: 'Install finished', install: { running: false, exit_code: 0 } } };
      else if (message.params.arguments?.confirm) result = { structuredContent: { summary: 'Install started', plan: {}, install: { running: true, step: '1/7 checking your PC' } } };
      else result = { structuredContent: { summary: 'Plan', plan: { title: 'Qwen3.8-Flash-Next IQ3_S', download_gb: 0, disk_short: null } } };
      queueMicrotask(() => child.stdout.emit('data', Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: message.id, result }) + String.fromCharCode(10))));
    } };
    return child;
  };
  const native = createNativeControl({ control, directory: data, spawnProcess, fetcher: async () => Response.json({ status: 'ok' }), requests: () => [], modelsDirectory: () => modelsDir,
    hardwareProbe: () => [{ index: 0, vram_gb: 11.9 }] });
  const fetches = () => calls.filter(c => c[0] === 'picturesFetch').map(c => c[1]);
  const installs = () => rpc.filter(p => p.name === 'strata_install');
  return { native, calls, saved, rpc, fetches, installs, root, install, data, home, modelsDir, dataFolder, modelFolder, gguf, pack, mtp, mmproj, configPath, args, settingsFile,
    config: () => JSON.parse(readFileSync(configPath, 'utf8')), settings: () => readFileSync(settingsFile, 'utf8'),
    cleanup: () => { if (earlier === undefined) delete process.env.STRATA_INSTALL_DIR; else process.env.STRATA_INSTALL_DIR = earlier; rmSync(root, { recursive: true, force: true }); } };
}
const loadOp = (f, args) => f.native.operation('load', { model: 'iq3_s', ...args });

test('own folder: the host knows the model\'s real data folder from its run config, not from the app\'s models folder', async () => {
  const f = fixture();
  try {
    const state = await f.native.refresh(true);
    const m = state.installed.models[0];
    assert.equal(m.ready, true);
    assert.equal(m.runDataFolder, f.dataFolder, 'the folder that holds packs/iq3_s and mtp/rt');
    assert.notEqual(m.runDataFolder, f.modelsDir);
    assert.equal(m.images, 'off');
    assert.equal(m.visionPrepared, false, 'the page offers the picture files');
    assert.equal(m.ownGgufDir, f.modelFolder);
  } finally { await f.native.close(); f.cleanup(); }
});

test('own folder: Load with Read pictures on downloads only the picture files into that data folder, keeps pack, draft layer and data_dir, and loads', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    const settingsBefore = f.settings();
    const treeBefore = listing(f.dataFolder);
    const configBefore = f.config();
    const job = await finish(f.native, loadOp(f, { vision: 'yes' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(job.kind, 'load');
    assert.deepEqual(f.native.state().jobs.map(j => j.kind), ['load'], 'no install job');
    assert.equal(f.installs().length, 0, 'the installer was not asked to do anything: no re-check, no second pack, no second draft layer');
    const [asked] = f.fetches();
    assert.equal(f.fetches().length, 1);
    assert.equal(asked.dataDir, f.dataFolder, 'the data folder of the model, not <home>/Strata-data');
    assert.equal(asked.configPath, f.configPath);
    assert.equal(asked.vision, 'yes');
    assert.equal(asked.family, 'qwen');
    assert.equal(asked.engineDir, join(f.root, 'engine'));
    assert.deepEqual(asked.hardware, { vram_gb: 11.9, make: 'nvidia', arch: 120 });
    // what changed on disk: the picture file, and the picture section of the run config; nothing else of the model
    const after = listing(f.dataFolder);
    const added = Object.keys(after).filter(k => !(k in treeBefore));
    assert.deepEqual(added.map(k => k.replaceAll('\\', '/')), ['/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf']);
    for (const [k, size] of Object.entries(treeBefore)) assert.equal(after[k], size, `${k} untouched`);
    assert.equal(existsSync(join(f.home, 'Strata-data')), false, 'no second data folder next to the app\'s models folder');
    assert.equal(existsSync(join(f.root, 'host-data', 'mtp')), false);
    const config = f.config();
    assert.deepEqual(config.args, configBefore.args, 'the pack, the draft layer and the other arguments are exactly as they were');
    assert.equal(config.vision.mmproj, f.mmproj);
    assert.equal(config.vision.model, f.gguf);
    assert.equal(f.settings(), settingsBefore, 'data_dir in the settings is not moved or rewritten');
    // the engine was started with the pictures
    assert.equal(f.saved.length, 1);
    assert.equal(f.saved[0].vision.mmproj, f.mmproj);
    assert.equal(f.native.state().instances[0].status, 'ready');
    assert.equal(f.native.state().instances[0].picturesNote, undefined, 'nothing to apologise for');
    assert.equal(f.native.state().installed.models[0].visionPrepared, true, 'the page stops offering the download');
    assert.equal(f.native.state().installed.models[0].images, 'yes');
  } finally { await f.native.close(); f.cleanup(); }
});

test('the progress is in plain words with a bar, and the Load goes on to "Loading the model…" after it', async () => {
  let release;
  const gate = new Promise(r => { release = r; });
  const f = fixture({ onFetch: async (options, ctx) => {
    options.onEvent({ event: 'progress', step: 'Downloading the picture files…', done: 3e8, total: 9e8 });
    await gate;
    writeFileSync(ctx.mmproj, 'x');
    return { ok: true, vision: { exe: 'e', mmproj: ctx.mmproj, model: ctx.gguf, gpu: false, max_tokens: 300 }, fetched: true, note: '' };
  } });
  try {
    await f.native.refresh(true);
    const id = loadOp(f, { vision: 'yes' }).id;
    await until(() => f.native.state().jobs.find(j => j.id === id)?.result?.install?.download, 'the first progress');
    const running = f.native.state().jobs.find(j => j.id === id);
    assert.equal(running.state, 'running');
    assert.equal(running.summary, 'Downloading the picture files (0.9 GB, once)…');
    const d = running.result.install.download;
    assert.deepEqual([d.file, d.percent, +d.done_gb.toFixed(1), +d.total_gb.toFixed(1)], ['picture reader', 33, 0.3, 0.9]);
    release();
    const job = await finish(f.native, id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(job.result?.install, undefined, 'the bar is gone');
  } finally { release(); await f.native.close(); f.cleanup(); }
});

test('no internet: the model loads without pictures, says so in one short line, and nothing is saved about pictures', async () => {
  const f = fixture({ onFetch: async () => { throw Object.assign(new ControlError("There is no internet connection, so the picture files can't be downloaded now."), { code: 'offline' }); } });
  try {
    await f.native.refresh(true);
    const configBefore = readFileSync(f.configPath, 'utf8');
    const started = Date.now();
    const job = await finish(f.native, loadOp(f, { vision: 'yes' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.ok(Date.now() - started < 5000, 'no hang');
    const instance = f.native.state().instances[0];
    assert.equal(instance.status, 'ready');
    assert.equal(instance.picturesNote, 'No internet connection: the model is loaded without picture reading. Load it again when you are online.');
    assert.equal(instance.vision, false);
    assert.equal(f.saved[0].vision, undefined, 'started without the picture reader');
    assert.equal(f.installs().length, 0);
    assert.deepEqual(f.config(), JSON.parse(configBefore), 'the run config says what it said: Read pictures stays on for the next Load');
    assert.deepEqual(f.native.state().jobs.map(j => j.kind), ['load']);
  } finally { await f.native.close(); f.cleanup(); }
});

test('no room on the disk or a download that fails: the same, with the right line', async () => {
  for (const [code, line] of [['disk', /not enough free disk space for the picture files/], ['failed', /could not be downloaded, so the model is loaded without picture reading\. Try again later\./], ['checksum', /could not be downloaded/]]) {
    const f = fixture({ onFetch: async () => { throw Object.assign(new ControlError('x', { output: { error: { details: 'cause in words' } } }), { code }); } });
    try {
      await f.native.refresh(true);
      const job = await finish(f.native, loadOp(f, { vision: 'yes' }).id);
      assert.equal(job.state, 'done', job.error);
      assert.match(f.native.state().instances[0].picturesNote, line, code);
      assert.equal(f.native.state().instances[0].status, 'ready');
    } finally { await f.native.close(); f.cleanup(); }
  }
});

test('Stop during the download ends the Load; nothing is started', async () => {
  let aborted = false;
  const f = fixture({ onFetch: (options) => new Promise((_, reject) => { options.signal.addEventListener('abort', () => { aborted = true; reject(new ControlError('stopped')); }); options.onEvent({ event: 'progress', step: 'Downloading the picture files…', done: 1, total: 9 }); }) });
  try {
    await f.native.refresh(true);
    const id = loadOp(f, { vision: 'yes' }).id;
    await until(() => f.native.state().jobs.find(j => j.id === id)?.summary.startsWith('Downloading the picture files'), 'the download');
    await f.native.cancel(id);
    await until(() => aborted, 'the abort');
    const job = f.native.state().jobs.find(j => j.id === id);
    assert.equal(job.state, 'cancelled');
    await new Promise(r => setTimeout(r, 50));
    assert.equal(f.saved.length, 0, 'no engine was started');
    assert.equal(f.native.state().instances.length, 0);
  } finally { await f.native.close(); f.cleanup(); }
});

test('nothing is downloaded unless the Load asked for pictures, and never for a model whose files are all there', async () => {
  // a restart or a tuning run passes no picture choice
  let f = fixture();
  try {
    await f.native.refresh(true);
    assert.equal((await finish(f.native, loadOp(f, {}).id)).state, 'done');
    assert.equal(f.fetches().length, 0);
    assert.equal(f.saved[0].vision, undefined);
  } finally { await f.native.close(); f.cleanup(); }
  // Read pictures off
  f = fixture();
  try { await f.native.refresh(true); assert.equal((await finish(f.native, loadOp(f, { vision: 'no' }).id)).state, 'done'); assert.equal(f.fetches().length, 0); } finally { await f.native.close(); f.cleanup(); }
  // prepared, the file is there
  f = fixture({ vision: true, mmprojFile: true });
  try { await f.native.refresh(true); assert.equal((await finish(f.native, loadOp(f, { vision: 'yes' }).id)).state, 'done'); assert.equal(f.fetches().length, 0); assert.ok(f.saved[0].vision); } finally { await f.native.close(); f.cleanup(); }
  // a DeepSeek model of an earlier Strata (Control: retired): the load stops with Control's own words, fetches nothing and writes nothing
  f = fixture({ arch: 'deepseek41' });
  try {
    const m = (await f.native.refresh(true)).installed.models[0];
    assert.deepEqual(m.retired, { line: 'DeepSeek - no longer supported.', refusal: 'DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw is a DeepSeek model. This version of Strata no longer runs DeepSeek models, so it was not started. Choose a Qwen model to run instead. If you no longer need the DeepSeek files, you can delete that folder to free the space.' });
    assert.equal(m.ready, false);
    const job = await finish(f.native, loadOp(f, { vision: 'yes' }).id);
    assert.equal(job.state, 'error');
    assert.match(String(job.error), /is a DeepSeek model\. This version of Strata no longer runs DeepSeek models, so it was not started\./);
    assert.equal(f.fetches().length, 0);
    assert.equal(f.saved.length, 0, 'no profile is saved');
  } finally { await f.native.close(); f.cleanup(); }
});

test('the picture file was deleted after the model was prepared: the model is not "missing files"; Load fetches it again, a restart goes on without pictures', async () => {
  let f = fixture({ vision: true });
  try {
    const m = (await f.native.refresh(true)).installed.models[0];
    assert.equal(m.ready, true, 'only a picture file is gone');
    assert.equal(m.picturesMissing, true);
    assert.equal(m.visionPrepared, false);
    assert.equal(m.images, 'off');
    const job = await finish(f.native, loadOp(f, { vision: 'yes' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.fetches().length, 1);
    assert.equal(f.native.state().installed.models[0].picturesMissing, false, 'the file is back');
  } finally { await f.native.close(); f.cleanup(); }
  f = fixture({ vision: true });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, loadOp(f, {}).id);   // no choice passed: never downloads
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.fetches().length, 0);
    assert.equal(f.saved[0].vision, undefined, 'the reader is not started on a file that is not there');
    assert.equal(f.native.state().instances[0].picturesNote, 'The picture files are missing, so the model is loaded without picture reading.');
  } finally { await f.native.close(); f.cleanup(); }
});

test('"Download picture-reading files" fetches the files without loading the model', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('pictures', { model: 'iq3_s', vision: 'yes' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(job.summary, 'The picture files are ready.');
    assert.equal(f.fetches().length, 1);
    assert.equal(f.saved.length, 0, 'no engine');
    assert.equal(f.installs().length, 0);
    assert.ok(f.config().vision?.mmproj);
    assert.equal(f.native.state().installed.models[0].visionPrepared, true);
    // a second press has nothing to do
    const again = await finish(f.native, f.native.operation('pictures', { model: 'iq3_s', vision: 'yes' }).id);
    assert.equal(again.state, 'done');
    assert.equal(f.fetches().length, 1);
  } finally { await f.native.close(); f.cleanup(); }
  // offline: the button says it, plainly
  const g = fixture({ onFetch: async () => { throw Object.assign(new ControlError("There is no internet connection, so the picture files can't be downloaded now."), { code: 'offline' }); } });
  try {
    await g.native.refresh(true);
    const job = await finish(g.native, g.native.operation('pictures', { model: 'iq3_s', vision: 'yes' }).id);
    assert.equal(job.state, 'error');
    assert.equal(job.error, "There is no internet connection, so the picture files can't be downloaded now.");
    assert.equal(g.config().vision, undefined);
  } finally { await g.native.close(); g.cleanup(); }
});

test('an installed model is prepared again in ITS data folder, never in the models folder, and the settings are not touched', async () => {
  for (const layout of ['own', 'normal', 'local']) {
    const f = fixture({ layout });
    try {
      await f.native.refresh(true);
      const before = f.settings();
      await f.native.plan({ family: 'qwen', model: 'IQ3_S', context: 32768, vision: 'yes', kv: 'int8' });
      const [call] = f.installs();
      assert.equal(call.arguments.data_dir, f.dataFolder, `${layout}: the folder with the pack and the draft layer`);
      assert.notEqual(call.arguments.data_dir, f.modelsDir);
      assert.equal(call.arguments.confirm, false, 'a plan only');
      if (layout !== 'normal') assert.equal(call.arguments.gguf_dir, f.modelFolder, `${layout}: its own files folder`);
      assert.equal(f.settings(), before, `${layout}: data_dir in the settings is not rewritten by looking at a plan`);
      // and the real thing, with the model's own folder again
      const job = await finish(f.native, f.native.operation('install', { family: 'qwen', model: 'IQ3_S', context: 32768, vision: 'yes', kv: 'int8' }).id);
      assert.ok(['running', 'done', 'error', 'cancelled'].includes(job.state));
      const real = f.installs().filter(p => p.arguments.confirm === true);
      assert.equal(real[0].arguments.data_dir, f.dataFolder, layout);
      assert.equal(f.settings(), before, `${layout}: the install did not move data_dir either`);
    } finally { await f.native.close(); f.cleanup(); }
  }
});

test('a model added with "Use a local GGUF" keeps its folder when the person later picks another downloads folder', async () => {
  const f = fixture({ layout: 'local' });
  try {
    await f.native.refresh(true);
    const elsewhere = join(f.root, 'elsewhere', 'Strata-data');
    mkdirSync(elsewhere, { recursive: true });
    f.native.setDownloadDirectory(elsewhere);
    const chosen = f.settings();
    assert.match(chosen, /elsewhere/, 'the person\'s choice is saved');
    await f.native.plan({ family: 'qwen', model: 'IQ3_S', context: 32768, vision: 'yes', kv: 'int8' });
    assert.equal(f.installs()[0].arguments.data_dir, f.dataFolder, 'the model\'s own pack and draft layer are not prepared again in the new folder');
    assert.equal(f.settings(), chosen, 'and the choice is not changed by the model');
    // a Load fetches the picture file next to the model\'s pack
    const job = await finish(f.native, loadOp(f, { vision: 'yes' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.fetches()[0].dataDir, f.dataFolder);
    assert.equal(existsSync(join(elsewhere, 'packs')), false);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a model that is not installed yet still goes to the person\'s chosen folder, and that choice is remembered as before', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    await f.native.plan({ family: 'qwen', model: 'Q2_0', context: 32768, vision: 'yes', kv: 'int8' });   // not in the catalog fixture: no installed model matches
    const call = f.installs()[0];
    assert.equal(call.arguments.data_dir, f.modelsDir, 'the app\'s models folder is the data folder of a new download');
    assert.match(f.settings(), /models/, 'the chosen folder is written for the installer, as before');
  } finally { await f.native.close(); f.cleanup(); }
});

// ---------------------------------------------------------------------------------------------- the Models page

const read = rel => readFileSync(new URL(rel, import.meta.url), 'utf8');

test('the Models page sends a plain Load, not the installer, when the only gap is the picture files', async () => {
  const { build } = await import('esbuild');
  const built = await build({ stdin: { contents: "export * from './src/lib/native-catalog';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
  const { fetchesPictures, PICTURE_FETCH_FAMILIES } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
  for (const family of ['qwen', 'coder', 'unsloth', 'gyro', 'swift', 'hy4-flash']) assert.equal(fetchesPictures({ sourceFamily: family }, true), true, family);
  for (const family of ['nvfp4', 'qwen36', undefined]) assert.equal(fetchesPictures({ sourceFamily: family }, true), false, `${family} keeps the installer`);
  assert.equal(fetchesPictures({ sourceFamily: 'qwen' }, false), false, 'a source checkout has no Control to fetch them');
  assert.equal(fetchesPictures({ sourceFamily: 'qwen' }, undefined), false);
  assert.equal(fetchesPictures(undefined, true), false);
  assert.deepEqual(PICTURE_FETCH_FAMILIES.slice().sort(), ['coder', 'gyro', 'hy4-flash', 'qwen', 'swift', 'unsloth']);
  const screen = read('../src/components/NativeModels.tsx');
  assert.match(screen, /const picturesByLoad = fetchesPictures\(selectedSize, native\?\.packaged\)/);
  assert.match(screen, /\(!selectedSize\?\.images \|\| vision === 'no' \|\| picturesByLoad \|\| installed\.images && installed\.images !== 'off'\)/, 'the shortcut to a plain Load no longer needs the pictures to be prepared');
  assert.match(screen, /loadedSelected\?\.picturesNote \? <p className="small muted native-vision-note" role="status">/, 'a failed picture download keeps one clear line under the checkbox');
  assert.doesNotMatch(screen, /Download picture-reading files|Picture reading needs a few more files/, 'GUI-FB2 item 1: no button and no offer sentence: Load model does it');
  assert.doesNotMatch(screen, /kind: 'pictures'/, 'the page no longer posts the picture-only step');
  assert.match(screen, /job\.kind === 'pictures'\) \{[\s\S]*Load the model again to read pictures\./, 'a model that is loaded without pictures is told to load again');
  // the host side of the same promise: the install path no longer takes its data folder from the app's models folder for an installed model
  const host = read('../server/native-control.mjs');
  assert.match(host, /const own=args\.data_dir===undefined\?existing\?\.runDataFolder:undefined;const data=own\?\?nativeDataDirectory\(/);
  assert.match(host, /if\(data&&packaged&&!args\.data_dir&&!own\)rememberDataFolder\(/, 'a model\'s own folder is never written to the settings');
});

test('every sentence the Load of the picture files can say is in the catalog in all nine languages', async () => {
  const langs = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
  const catalogs = Object.fromEntries(langs.map(l => [l, JSON.parse(read(`../src/locales/pic-${l}.json`))]));
  const english = Object.values(catalogs.en);
  const host = read('../server/native-control.mjs');
  const words = host.slice(host.indexOf('const PICTURES_WORDS={'), host.indexOf('const PLAIN_PICTURE_CODES'));
  const hostSentences = [...words.matchAll(/:'([^']+)'/g)].map(m => m[1]);
  assert.equal(hostSentences.length, 7, 'download, check, loading, offline, disk, failed, gone');
  // the sentences Control's pictures module gives the page's button (strata_control/pictures.py WORDS, with the "Try again" the module adds)
  const python = ["There is no internet connection, so the picture files can't be downloaded now.", 'There is not enough free disk space for the picture files (about 1 GB).',
    'The picture files could not be downloaded. Try again; it continues where it stopped.', 'The downloaded picture files did not pass the safety check, so they were deleted. Try again.',
    'The picture files could not be downloaded. Try again.'];
  for (const sentence of [...hostSentences, ...python, 'Downloading the picture files', 'The picture files are ready.', 'Picture files ready', 'Load the model again to read pictures.', 'Picture reading is ready.', 'Loading the model…']) {
    if (sentence === 'Loading the model…') continue;   // already in the Drafting catalog
    assert.ok(english.includes(sentence), `in the catalog: ${sentence}`);
  }
  const source = read('../../../terminal/strata_control/pictures.py');
  for (const sentence of python.slice(0, 2)) assert.ok(source.includes(sentence.replace(' Try again; it continues where it stopped.', '')), `the module says: ${sentence}`);
  const slots = t => (t.match(/\{\d+\}/g) ?? []).sort().join();
  for (const [id, text] of Object.entries(catalogs.en)) for (const l of langs) {
    const t = catalogs[l][id];
    assert.ok(typeof t === 'string' && t.trim(), `${id} in ${l}`);
    assert.equal(slots(t), slots(text), `${id} slots in ${l}`);
    if (l !== 'en' && text.length > 25) assert.notEqual(t, text, `${id} is translated in ${l}`);
  }
});
