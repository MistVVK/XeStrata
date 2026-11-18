// FIX-APP-0216 A7 (DIAG-WINHOST app-path test): after a FAILED model load the instance stays listed in state "error", and both Settings -> Updates jobs (the Strata row and the Engine row)
// refused with "Unload native models before changing engine files." until it was unloaded, so a person hit by the start-up crash who went straight to Updates could not get the fix.
// An instance in error holds no engine file open: the update jobs now drop it and go on. A model that is loading, ready or unloading keeps the refusal.
// Fixture: applin-0214-drafter.test.mjs's (fake Control, fake installer over MCP), cut down; `startFails` = how many starts of the engine fail.
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';

const REFUSAL = 'Unload native models before changing engine files.';

function fixture({ startFails = 0 } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-fixapp-a7-'));
  const install = join(root, 'install'), data = join(root, 'host-data'), home = join(root, 'home'), modelsDir = join(home, 'models');
  for (const d of [install, data, modelsDir]) mkdirSync(d, { recursive: true });
  const earlier = process.env.STRATA_INSTALL_DIR;
  process.env.STRATA_INSTALL_DIR = install;
  const dataFolder = join(root, 'Strata-data'), modelFolder = join(dataFolder, 'models', 'UD-Q4_K_XL');
  const gguf = join(modelFolder, 'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00002.gguf');
  mkdirSync(modelFolder, { recursive: true });
  writeFileSync(gguf, 'GGUF'); writeFileSync(gguf.replace('00001', '00002'), 'GGUF');
  const pack = join(dataFolder, 'packs', 'ud_q4_k_xl'), mtp = join(dataFolder, 'mtp', 'rt');
  mkdirSync(join(pack, 'tokenizer'), { recursive: true });
  writeFileSync(join(pack, 'native_experts.txt'), 'x');
  const configPath = join(install, 'strata-ud_q4_k_xl.json');
  const args = ['--pack', pack, '--native', gguf, '--ple-gguf', gguf.replace('00001', '00002'), '--mtp', mtp, '--max-context', '131072', '--kv', 'int8'];
  writeFileSync(configPath, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(pack, 'tokenizer'), args, model_name: 'qwen3.8-flash-next-ud_q4_k_xl', port: 8080 }));
  writeFileSync(process.env.STRATA_SETTINGS_FILE, JSON.stringify({ data_dir: join(root, 'host-owned', 'Strata-data'), data_dir_at: 1700000000 }));
  const state = { starts: 0, stops: 0, updates: [] };
  const control = {
    control: { dir: join(root, 'control'), python: join(root, 'control', 'python', 'python') }, version: '0.2.10',
    async installedModels() {
      return { install_dir: install, models: [{ id: 'ud_q4_k_xl', name: 'qwen3.8-flash-next-ud_q4_k_xl', source_config: configPath, context_tokens: 131072, backend: 'cuda', component_status: 'present', missing: [], issue_codes: [] }] };
    },
    async engine() { return { usable: true, backend: 'cuda', components_dir: join(root, 'c'), engine: { version: '0.1.62', platform: 'linux-x86_64', backend: 'cuda', exe: join(root, 'engine', 'strata'), root: join(root, 'engine') } }; },
    async draftingPlan(options) { return options.configPath ? { ok: true, serves: true, split: false, choice: 'mtp', drafter: 'off', needed: false, gb: 0, mtp_needed: false, line: '' } : { ok: true, serves: true, choice: 'dflash2', split: false }; },
    async draftingFetch() { return { ok: true, event: 'result', pack: null, skipped: false }; },
    async splitEngine() { return { ok: true, split: false, line: '', warn: false, reasons: [], written: null }; },
    async saveProfile() { return {}; },
    async start() { state.starts += 1; if (state.starts <= startFails) throw new Error('The engine stopped: out of memory'); return { ok: true, state: 'ready' }; },
    async stop() { state.stops += 1; return { ok: true }; },
    async update(options) { state.updates.push({ engine: options.engine, control: options.control }); return { ok: true, results: [{ component: 'engine', status: 'updated', installed_after: { version: '0.1.63' } }] }; },
  };
  const catalog = { families: [{ family: 'qwen', title: 'Qwen3.8-Flash-Next', about: 'x', images: false, experimental: false, sizes: [
    { id: 'ud_q4_k_xl', model: 'UD-Q4_K_XL', download_gb: 22, ram_needed_gb: 60, about: 'x', on_this_pc: 'fits', installed: true, images: false, experimental: false }] }],
    contexts: [32768, 131072], recommendation: { family: 'qwen', model: 'UD-Q4_K_XL', context: 131072 }, context_note: '' };
  const hardware = { ram_gb: 256, cpu: { name: 'cpu', cores: 32 }, engine_vendor: 'nvidia', gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 3060', vendor: 'nvidia', vram_gb: 12, arch: 86, usable: true }] };
  const spawnProcess = (command, argv) => {
    if (!String(argv[0]).endsWith('strata_mcp.py')) throw new Error('no process but the installer may be started in packaged mode: ' + command + ' ' + argv.join(' '));
    const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter(); child.exitCode = null;
    child.kill = () => { child.exitCode = 0; queueMicrotask(() => child.emit('exit', 0)); return true; };
    child.stdin = { write(line) {
      const message = JSON.parse(line); if (!message.id) return;
      let result;
      if (message.method === 'initialize') result = { protocolVersion: '2024-11-05' };
      else if (message.method === 'tools/list') result = { tools: ['strata_models', 'strata_status', 'strata_install', 'strata_remove'].map(name => ({ name })) };
      else if (message.params.name === 'strata_models') result = { structuredContent: catalog };
      else if (message.params.name === 'strata_status') result = { structuredContent: { hardware, recommendation: catalog.recommendation, install: null, running: [], summary: 'x' } };
      else result = { structuredContent: { summary: 'Install finished', install: { running: false, exit_code: 0 } } };
      queueMicrotask(() => child.stdout.emit('data', Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: message.id, result }) + String.fromCharCode(10))));
    } };
    return child;
  };
  const fetcher = async url => String(url).endsWith('/metrics') ? Response.json({ engine: {} }) : Response.json({ status: 'ok' });
  const native = createNativeControl({ control, directory: data, spawnProcess, fetcher, requests: () => [], modelsDirectory: () => modelsDir, hardwareProbe: () => [{ index: 0, vram_gb: 12 }], lockWaitMs: 300, lockPollMs: 20 });
  return { native, state, cleanup: () => { if (earlier === undefined) delete process.env.STRATA_INSTALL_DIR; else process.env.STRATA_INSTALL_DIR = earlier; rmSync(root, { recursive: true, force: true }); } };
}
const finish = async (native, id) => { for (let i = 0; i < 1500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };

test('after a failed load, both update jobs (Engine row, Strata row) go on instead of refusing; the failed instance is dropped', async () => {
  const f = fixture({ startFails: 20 });
  try {
    await f.native.refresh(true);
    const failed = await finish(f.native, f.native.operation('load', { model: 'ud_q4_k_xl' }).id);
    assert.equal(failed.state, 'error', 'the load failed');
    const listed = f.native.state().instances;
    assert.equal(listed.length, 1, 'the failed instance stays listed');
    assert.equal(listed[0].status, 'error');
    const engine = await finish(f.native, f.native.operation('update-engine', {}).id);
    assert.equal(engine.state, 'done', engine.error);
    assert.notEqual(engine.error, REFUSAL);
    assert.deepEqual(f.state.updates.length, 1, 'Control was asked to update');
    assert.equal(f.native.state().instances.length, 0, 'the failed instance was unloaded first');
  } finally { await f.native.close(); f.cleanup(); }
  const g = fixture({ startFails: 20 });
  try {
    await g.native.refresh(true);
    await finish(g.native, g.native.operation('load', { model: 'ud_q4_k_xl' }).id);
    assert.equal(g.native.state().instances[0].status, 'error');
    const strata = await finish(g.native, g.native.operation('update-strata', {}).id);
    assert.notEqual(strata.error, REFUSAL, 'the Strata row is not refused either');
    assert.equal(g.native.state().instances.length, 0);
  } finally { await g.native.close(); g.cleanup(); }
});

test('a model that is really running still refuses both update jobs, and stays loaded', async () => {
  const f = fixture({ startFails: 0 });
  try {
    await f.native.refresh(true);
    const loaded = await finish(f.native, f.native.operation('load', { model: 'ud_q4_k_xl' }).id);
    assert.equal(loaded.state, 'done', loaded.error);
    assert.equal(f.native.state().instances[0].status, 'ready');
    for (const kind of ['update-engine', 'update-strata']) {
      const job = await finish(f.native, f.native.operation(kind, {}).id);
      assert.equal(job.state, 'error', kind);
      assert.equal(job.error, REFUSAL, kind);
    }
    assert.equal(f.native.state().instances.length, 1, 'the running model was not touched');
    assert.equal(f.state.updates.length, 0, 'Control was not asked to update under a running model');
  } finally { await f.native.close(); f.cleanup(); }
});
