/**
 * FIX-APP-0216 A3: the drafter (DFlash2, about 1.2 GB) is fetched while a model loads (the first "Tune & start chatting" included). The step used to say only
 * "Applying your resource preference and loading the model..." (the page never drew `details`). Now the job says what is downloaded, in plain words, with the size
 * as soon as Control's plan names it and the running figures after that; Control's other steps (checking, packing) keep their own words.
 * The fixture is applin-0214-drafter.test.mjs's (fake Control, fake installer over MCP), cut down to one loaded model.
 */
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl, DRAFTER_DOWNLOAD_SUMMARY } from '../server/native-control.mjs';

function fixture({ plan, steps = [] }) {
  const root = mkdtempSync(join(tmpdir(), 'strata-fixapp-drafter-'));
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
  const seen = [], state = { fetches: 0 };
  let native;
  const control = {
    control: { dir: join(root, 'control'), python: join(root, 'control', 'python', 'python') }, version: '0.2.10',
    async installedModels() {
      return { install_dir: install, models: [{ id: 'ud_q4_k_xl', name: 'qwen3.8-flash-next-ud_q4_k_xl', source_config: configPath, context_tokens: 131072, backend: 'cuda', component_status: 'present', missing: [], issue_codes: [] }] };
    },
    async engine() { return { usable: true, backend: 'cuda', components_dir: join(root, 'c'), engine: { version: '0.1.62', platform: 'linux-x86_64', backend: 'cuda', exe: join(root, 'engine', 'strata'), root: join(root, 'engine') } }; },
    async draftingPlan(options) { return plan(options, state); },
    async draftingFetch(options) {
      state.fetches += 1;
      const snap = () => { const job = native.state().jobs.find(j => j.kind === 'load'); seen.push({ summary: job.summary, details: job.details }); };
      snap();                                                       // the moment the fetch starts, before Control's first progress event
      for (const event of steps) { options.onEvent(event); snap(); }
      return { ok: true, event: 'result', pack: null, skipped: false };
    },
    async splitEngine() { return { ok: true, split: false, line: '', warn: false, reasons: [], written: null }; },
    async saveProfile() { return {}; },
    async start() { return { ok: true, state: 'ready' }; },
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
  native = createNativeControl({ control, directory: data, spawnProcess, fetcher, requests: () => [], modelsDirectory: () => modelsDir, hardwareProbe: () => [{ index: 0, vram_gb: 12 }] });
  return { native, seen, state, cleanup: () => { if (earlier === undefined) delete process.env.STRATA_INSTALL_DIR; else process.env.STRATA_INSTALL_DIR = earlier; rmSync(root, { recursive: true, force: true }); } };
}
const finish = async (native, id) => { for (let i = 0; i < 1500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };
/** Control's plan: at the load the DFlash2 drafter is needed (not on this computer), `gb` of it. */
const needs = gb => options => options.configPath
  ? { ok: true, serves: true, split: false, choice: 'dflash2', drafter: 'dflash2', needed: true, gb, mtp_needed: false, mtp_gb: 0, line: 'Drafting: DFlash2' }
  : { ok: true, serves: true, choice: 'dflash2', split: false };

test('a load that fetches the drafter says so at once, with the size, and keeps the downloading line while Control reports figures', async () => {
  const f = fixture({ plan: needs(1.2), steps: [
    { event: 'progress', step: 'Downloading the DFlash2 drafter…', done: 3e8, total: 12e8 },
    { event: 'progress', step: 'Downloading the DFlash2 drafter…', done: 12e8, total: 12e8 },
    { event: 'progress', step: 'Making the drafter ready for this engine…', done: 0, total: 0 }] });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'ud_q4_k_xl' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.fetches, 1);
    assert.deepEqual(f.seen[0], { summary: DRAFTER_DOWNLOAD_SUMMARY, details: '1.2 GB' }, 'before the first figure');
    assert.deepEqual(f.seen[1], { summary: DRAFTER_DOWNLOAD_SUMMARY, details: '0.30 of 1.20 GB' });
    assert.deepEqual(f.seen[2], { summary: DRAFTER_DOWNLOAD_SUMMARY, details: undefined }, 'a finished file shows no figure');
    assert.deepEqual(f.seen[3], { summary: 'Making the drafter ready for this engine…', details: undefined }, 'the other steps keep Control\'s words');
  } finally { await f.native.close(); f.cleanup(); }
});

test('no size in the plan: the line is still there, without a figure', async () => {
  const f = fixture({ plan: needs(undefined) });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'ud_q4_k_xl' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.deepEqual(f.seen[0], { summary: DRAFTER_DOWNLOAD_SUMMARY, details: undefined });
  } finally { await f.native.close(); f.cleanup(); }
});

test('the tuning card shows the step\'s details line, so the sizes of the drafter download are visible', async () => {
  const { readFileSync } = await import('node:fs');
  const w = readFileSync(new URL('../src/components/GuidedSetup.tsx', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  assert.match(w, /<strong>\{tr\(tuneJob\.summary\)\}<\/strong>\{tr\(tuneJob\.details \?/);
  const en = JSON.parse(readFileSync(new URL('../src/locales/fixa-en.json', import.meta.url), 'utf8'));
  assert.ok(Object.values(en).includes(DRAFTER_DOWNLOAD_SUMMARY), 'the summary is a catalog string, so it is translated');
  const css = readFileSync(new URL('../src/styles/onboarding.css', import.meta.url), 'utf8');
  assert.match(css, /\.onboard-tuning-status>small\.onboard-tuning-details\{display:block/, 'the details line is on its own line, not run into "Your choice"');
});

test('a load with the drafter already here never says it is downloading', async () => {
  const f = fixture({ plan: options => options.configPath ? { ok: true, serves: true, split: false, choice: 'dflash2', drafter: 'dflash2', needed: false, gb: 0, mtp_needed: false, line: 'Drafting: DFlash2' } : { ok: true, serves: true, choice: 'dflash2', split: false } });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'ud_q4_k_xl' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.fetches, 0);
    assert.notEqual(job.summary, DRAFTER_DOWNLOAD_SUMMARY);
  } finally { await f.native.close(); f.cleanup(); }
});
