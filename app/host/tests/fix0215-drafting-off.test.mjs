/**
 * 0.2.14.x follow-ups of the app side (phaseA-tests notes-ctrl-fix-2.md):
 *  (a) Drafting Off in the app reaches setup as `--mtp off` (the installer's `mtp` word), so no draft layer is fetched, on a Mac too (the Terminal's 1821f298d);
 *  (b) the offline words (owner 11-15, the Terminal's drafting.MTP_OFFLINE_NOTE) are said where the app says its drafter notes (the job's drafterNote), and only when
 *      no other drafter stands in: when Control's plan names a drafter that runs, the sentence "The model starts without a drafter." would be wrong.
 * The fixture is applin-0214-drafter.test.mjs's (fake Control, fake installer over MCP).
 */
import './helpers/isolate-profile.mjs';   // a test run never writes the owner's real Strata profile
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl, MTP_OFFLINE_NOTE } from '../server/native-control.mjs';

function fixture({ plan }) {
  const root = mkdtempSync(join(tmpdir(), 'strata-fix0215-'));
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
  const state = { installs: 0, planCalls: [], rpc: [], starts: 0 };
  const control = {
    control: { dir: join(root, 'control'), python: join(root, 'control', 'python', 'python') }, version: '0.2.10',
    async installedModels() {
      return { install_dir: install, models: [{ id: 'ud_q4_k_xl', name: 'qwen3.8-flash-next-ud_q4_k_xl', source_config: configPath, context_tokens: 131072, backend: 'cuda', component_status: 'present', missing: [], issue_codes: [] }] };
    },
    async engine() { return { usable: true, backend: 'cuda', components_dir: join(root, 'c'), engine: { version: '0.1.60', platform: 'linux-x86_64', backend: 'cuda', exe: join(root, 'engine', 'strata'), root: join(root, 'engine') } }; },
    async draftingPlan(options) { state.planCalls.push(options); return plan(options, state); },
    async draftingFetch() { return { ok: true, event: 'result', pack: null, skipped: false }; },
    async splitEngine() { return { ok: true, split: false, line: '', warn: false, reasons: [], written: null }; },
    async saveProfile() { return {}; },
    async start() { state.starts += 1; return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
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
      if (message.method === 'tools/call') state.rpc.push(message.params);
      let result;
      if (message.method === 'initialize') result = { protocolVersion: '2024-11-05' };
      else if (message.method === 'tools/list') result = { tools: ['strata_models', 'strata_status', 'strata_install', 'strata_remove'].map(name => ({ name })) };
      else if (message.params.name === 'strata_models') result = { structuredContent: catalog };
      else if (message.params.name === 'strata_status') result = { structuredContent: { hardware, recommendation: catalog.recommendation, install: null, running: [], summary: 'x' } };
      else if (Object.keys(message.params.arguments ?? {}).length === 0) result = { structuredContent: { summary: 'Install finished', install: { running: false, exit_code: 0 } } };
      else if (message.params.arguments?.confirm) { state.installs += 1; result = { structuredContent: { summary: 'Install started', plan: {}, install: { running: true, step: '1/7' } } }; }   // offline: the layer never arrives
      else result = { structuredContent: { summary: 'Plan', plan: { title: 'Qwen3.8-Flash-Next UD-Q4_K_XL', download_gb: 5.2, disk_short: null } } };
      queueMicrotask(() => child.stdout.emit('data', Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: message.id, result }) + String.fromCharCode(10))));
    } };
    return child;
  };
  const fetcher = async url => String(url).endsWith('/metrics') ? Response.json({ engine: {} }) : Response.json({ status: 'ok' });
  const native = createNativeControl({ control, directory: data, spawnProcess, fetcher, requests: () => [], modelsDirectory: () => modelsDir, hardwareProbe: () => [{ index: 0, vram_gb: 12 }] });
  return { native, state, modelFolder, mtp, configPath,
    cleanup: () => { if (earlier === undefined) delete process.env.STRATA_INSTALL_DIR; else process.env.STRATA_INSTALL_DIR = earlier; rmSync(root, { recursive: true, force: true }); } };
}
const finish = async (native, id) => { for (let i = 0; i < 1500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };
const installs = f => f.state.rpc.filter(p => p.name === 'strata_install' && p.arguments?.confirm === true);
const atInstall = choice => (options) => (!options.configPath ? { ok: true, serves: true, split: false, choice, drafter: choice, needed: false, mtp_needed: false, gb: 0 } :
  { ok: true, serves: true, split: false, choice, drafter: choice, needed: false, mtp_needed: false, gb: 0, line: 'x' });

test('(a) Drafting Off at the install: setup is told --mtp off (no draft layer), and no drafter word', async () => {
  const f = fixture({ plan: atInstall('off') });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('install', { family: 'qwen', model: 'UD_Q4_K_XL', gguf_dir: f.modelFolder, drafting: 'off' }).id);
    assert.equal(job.state, 'done', job.error);
    const call = installs(f)[0].arguments;
    assert.equal(call.mtp, 'off');
    assert.equal(call.drafter, undefined);
  } finally { await f.native.close(); f.cleanup(); }
});

test('(a) Auto and the MTP keep the installer as before: no mtp word', async () => {
  for (const [drafting, choice] of [['auto', 'dflash2'], ['mtp', 'mtp'], ['auto', 'mtp']]) {
    const f = fixture({ plan: atInstall(choice) });
    try {
      await f.native.refresh(true);
      const job = await finish(f.native, f.native.operation('install', { family: 'qwen', model: 'UD_Q4_K_XL', gguf_dir: f.modelFolder, drafting }).id);
      assert.equal(job.state, 'done', job.error);
      assert.equal(installs(f)[0].arguments.mtp, undefined, `${drafting} -> ${choice}`);
    } finally { await f.native.close(); f.cleanup(); }
  }
});

/** The Models screen's Load with the MTP chosen, offline: the layer step comes back without the layer. `standIn`: Control's plan after it names the drafter that runs. */
const offlineLoad = standIn => (options, state) => (!options.configPath ? { ok: true, serves: true, choice: 'dflash2', split: false } :
  { ok: true, serves: true, split: false, choice: 'mtp', drafter: state.installs === 0 ? 'off' : standIn, needed: false, gb: 0, mtp_needed: true, mtp_gb: 5.2, line: 'Drafting' });

test('(b) offline with no other drafter: the owner\'s sentence, word for word, in the job\'s drafter note', async () => {
  const f = fixture({ plan: offlineLoad('off') });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'ud_q4_k_xl', values: { drafting: 'mtp' } }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(installs(f).length, 1, 'the layer step ran once');
    assert.equal(job.drafterNote, "There is no internet connection, so the draft layer can't be downloaded now. The model starts without a drafter.");
    assert.equal(MTP_OFFLINE_NOTE, job.drafterNote);
  } finally { await f.native.close(); f.cleanup(); }
});

test('(b) offline but the DFlash2 drafter on this computer stands in: the sentence is not said', async () => {
  const f = fixture({ plan: offlineLoad('dflash2') });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'ud_q4_k_xl', values: { drafting: 'mtp' } }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(installs(f).length, 1);
    assert.notEqual(job.drafterNote, MTP_OFFLINE_NOTE);
  } finally { await f.native.close(); f.cleanup(); }
});

/** Item 7 (owner 11-15): a drafter chosen in the app's Drafting setting stays chosen at later loads with no new choice, and the fallback a start takes (Control's plan
 * names the other drafter: the chosen one's file is not here) is never written over it. */
test('a chosen MTP stays the saved choice over a later plain load, even when the start falls back to DFlash2', async () => {
  const seen = [];
  const plan = (options) => {
    if (!options.configPath) return { ok: true, serves: true, choice: 'dflash2', split: false };
    seen.push(JSON.parse(readFileSync(options.configPath, 'utf8')).drafting ?? null);
    return { ok: true, serves: true, split: false, choice: 'mtp', drafter: 'dflash2', needed: false, gb: 0, mtp_needed: false, mtp_gb: 0, line: 'Drafting: DFlash2 (the MTP drafter isn\'t downloaded yet)' };
  };
  const f = fixture({ plan });
  try {
    await f.native.refresh(true);
    let job = await finish(f.native, f.native.operation('load', { model: 'ud_q4_k_xl', values: { drafting: 'mtp' } }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(JSON.parse(readFileSync(f.configPath, 'utf8')).drafting, 'mtp', 'the choice is saved as the person set it, not the fallback');
    job = await finish(f.native, f.native.operation('load', { model: 'ud_q4_k_xl' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(JSON.parse(readFileSync(f.configPath, 'utf8')).drafting, 'mtp', 'a plain load keeps it');
    assert.equal(seen.at(-1), 'mtp', 'and asks Control about the chosen MTP, not Auto');
    assert.equal(installs(f).length, 0, 'nothing was fetched for it');
  } finally { await f.native.close(); f.cleanup(); }
});

test('(b) the Terminal and the app hold the same words', () => {
  const py = readFileSync(new URL('../../../terminal/strata_control/drafting.py', import.meta.url), 'utf8');
  assert.ok(py.includes(`MTP_OFFLINE_NOTE = "${MTP_OFFLINE_NOTE}"`));
});
