/**
 * 0.2.10.1: the desktop app's Auto load uses the all-cards split engine wherever the Terminal does. The decision is Control's (strata_control/split_engine.py, `strata drafting plan`,
 * `strata split-engine apply`; tests/terminal/test_split_parallel.py); this file drives the host's load with a fake Control and checks what the host asks and starts:
 * no --no-split on the plan or the fetch, the MTP draft layer fetched through the existing installer step (gguf_dir, so the model is not fetched again), the split engine's
 * run config saved as the profile, a person's DFlash2 kept, a failed layer fetch ending in a working layer split with one plain note. Nothing here starts an engine.
 */
import './helpers/isolate-profile.mjs';   // a test run never writes the owner's real Strata profile
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';
import { createControl } from '../server/control-runtime.mjs';
import { resolveDrafter as hostResolve } from '../server/native-settings.mjs';

const SPLIT_PLAN = { ok: true, split: true, choice: 'mtp', drafter: 'mtp', needed: false, gb: 0, line: 'Drafting: MTP (the all-cards split engine uses the MTP draft layer, 5.2 GB)', serves: true };
const DFLASH2_PLAN = { ok: true, split: false, choice: 'dflash2', drafter: 'dflash2', needed: true, gb: 1.2, mtp_needed: false, mtp_gb: 0, line: 'Drafting: DFlash2', serves: true };

/**
 * A packaged install on two 48 GB cards. `plan(options, state)` answers `drafting plan` (state.installs says how many layer installs ran), `split(options, files)` answers `split-engine apply`
 * (it may write options.outPath), `installFails` makes the installer's MCP tool fail, `startFails` fails the first N engine starts.
 */
function fixture({ plan, split = null, installFails = false, startFails = 0, metrics = {}, config = {} } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-splitapp-test-'));
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
  writeFileSync(configPath, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(pack, 'tokenizer'), args, model_name: 'qwen3.8-flash-next-ud_q4_k_xl', port: 8080, gpu: [0, 1], ...config }));
  writeFileSync(process.env.STRATA_SETTINGS_FILE, JSON.stringify({ data_dir: join(root, 'host-owned', 'Strata-data'), data_dir_at: 1700000000 }));
  const state = { installs: 0, planCalls: [], fetchCalls: [], splitCalls: [], saved: [], starts: 0, rpc: [] };
  const control = {
    control: { dir: join(root, 'control'), python: join(root, 'control', 'python', 'python') }, version: '0.2.10',
    async installedModels() {
      return { install_dir: install, models: [{ id: 'ud_q4_k_xl', name: 'qwen3.8-flash-next-ud_q4_k_xl', source_config: configPath, context_tokens: 131072, backend: 'cuda', component_status: 'present', missing: [], issue_codes: [] }] };
    },
    async engine() { return { usable: true, backend: 'cuda', components_dir: join(root, 'c'), engine: { version: '0.1.56', platform: 'linux-x86_64', backend: 'cuda', exe: join(root, 'engine', 'strata'), root: join(root, 'engine') } }; },
    async draftingPlan(options) { state.planCalls.push(options); return typeof plan === 'function' ? plan(options, state) : plan; },
    async draftingFetch(options) { state.fetchCalls.push(options); return { ok: true, event: 'result', pack: null, skipped: false }; },
    async splitEngine(options) {
      state.splitCalls.push({ ...options, input: JSON.parse(readFileSync(options.configPath, 'utf8')) });
      if (!split) return { ok: true, split: false, line: '', warn: false, reasons: [], written: null };
      return split(options, state);
    },
    async saveProfile(name, path) { state.saved.push(JSON.parse(readFileSync(path, 'utf8'))); return {}; },
    async start() { state.starts += 1; if (state.starts <= startFails) throw new Error('The engine stopped: out of memory'); return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
  };
  const catalog = { families: [{ family: 'qwen', title: 'Qwen3.8-Flash-Next', about: 'x', images: false, experimental: false, sizes: [
    { id: 'ud_q4_k_xl', model: 'UD-Q4_K_XL', download_gb: 22, ram_needed_gb: 60, about: 'x', on_this_pc: 'fits', installed: true, images: false, experimental: false }] }],
    contexts: [32768, 131072], recommendation: { family: 'qwen', model: 'UD-Q4_K_XL', context: 131072 }, context_note: '' };
  const hardware = { ram_gb: 256, cpu: { name: 'cpu', cores: 32 }, engine_vendor: 'nvidia', gpus: [0, 1].map(index => ({ index, name: 'NVIDIA RTX A6000', vendor: 'nvidia', vram_gb: 48, arch: 86, usable: true })) };
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
      else if (message.params.arguments?.confirm) {
        if (installFails) result = { isError: true, structuredContent: { error: 'The download did not finish.' } };
        else { state.installs += 1; mkdirSync(mtp, { recursive: true }); writeFileSync(join(mtp, 'dense.bin'), 'x'); result = { structuredContent: { summary: 'Install started', plan: {}, install: { running: true, step: '1/7' } } }; }
      } else result = { structuredContent: { summary: 'Plan', plan: { title: 'Qwen3.8-Flash-Next UD-Q4_K_XL', download_gb: 5.2, disk_short: null } } };
      queueMicrotask(() => child.stdout.emit('data', Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: message.id, result }) + String.fromCharCode(10))));
    } };
    return child;
  };
  const fetcher = async url => String(url).endsWith('/metrics') ? Response.json({ engine: metrics }) : Response.json({ status: 'ok' });
  const native = createNativeControl({ control, directory: data, spawnProcess, fetcher, requests: () => [], modelsDirectory: () => modelsDir, hardwareProbe: () => [0, 1].map(index => ({ index, vram_gb: 48 })) });
  return { native, state, root, gguf, modelFolder, mtp, configPath, args,
    cleanup: () => { if (earlier === undefined) delete process.env.STRATA_INSTALL_DIR; else process.env.STRATA_INSTALL_DIR = earlier; rmSync(root, { recursive: true, force: true }); } };
}
const finish = async (native, id) => { for (let i = 0; i < 1500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };
const load = (f, args = {}) => f.native.operation('load', { model: 'ud_q4_k_xl', ...args });

/** The split engine's answer: the layer config with the split's engine and a lane count (Control's split_config, in miniature), written where the host asked. */
const splitAnswer = (options) => {
  const cfg = JSON.parse(readFileSync(options.configPath, 'utf8'));
  const lanes = Math.min(8, cfg.parallel ?? 6);
  writeFileSync(options.outPath, JSON.stringify({ ...cfg, exe: '/e/strata-split-2-sm86', args: ['--native', cfg.args[cfg.args.indexOf('--native') + 1], '--batch', String(lanes)], drafting: 'mtp', split_fallback: { exe: cfg.exe, args: cfg.args, parallel: cfg.parallel }, split_engine_note: 'Using the all-cards split engine on 2 cards.' }));
  return { ok: true, split: true, line: 'Using the all-cards split engine on 2 cards.', warn: false, reasons: [], cards: 2, written: options.outPath };
};
/** Before the layer is fetched Control says split + mtp_needed; after it, split with the layer here. */
const splitPlan = (_options, state) => state.installs === 0 ? { ...SPLIT_PLAN, mtp_needed: true, mtp_gb: 5.2 } : { ...SPLIT_PLAN, mtp_needed: false, mtp_gb: 0 };

test('no drafting call of the host carries --no-split by itself, and the flag is there for the failed layer fetch', async () => {
  const calls = [];
  const control = createControl({ control: { dir: 'C:/c', python: 'py', cli: 'strata.py' }, run: async args => { calls.push(args); return { code: 0, json: { ok: true, event: 'result', serves: true, split: false }, text: '', stderr: '' }; } });
  await control.draftingPlan({ mode: 'auto' });
  await control.draftingPlan({ configPath: 'C:/w/cfg.json', mode: 'auto', family: 'unsloth', dataDir: 'C:/d' });
  await control.draftingFetch({ configPath: 'C:/w/cfg.json', mode: 'auto' });
  assert.equal(calls.length, 3);
  for (const args of calls) assert.ok(!args.includes('--no-split'), args.join(' '));
  assert.deepEqual(calls[2], ['drafting', 'fetch', '--config', 'C:/w/cfg.json', '--mode', 'auto', '--if-needed', '--output', 'jsonl']);
  await control.draftingPlan({ configPath: 'C:/w/cfg.json', mode: 'auto', noSplit: true });
  await control.draftingFetch({ configPath: 'C:/w/cfg.json', mode: 'auto', noSplit: true });
  assert.ok(calls[3].includes('--no-split') && calls[4].includes('--no-split'));
  await control.splitEngine({ configPath: 'C:/w/layer.json', outPath: 'C:/w/split.json' });
  assert.deepEqual(calls[5], ['split-engine', 'apply', '--config', 'C:/w/layer.json', '--out', 'C:/w/split.json', '--output', 'json']);
});

test('--engines-unloading is on the plan and the fetch only when the host asks for it', async () => {
  const calls = [];
  const control = createControl({ control: { dir: 'C:/c', python: 'py', cli: 'strata.py' }, run: async args => { calls.push(args); return { code: 0, json: { ok: true, event: 'result', serves: true, split: false }, text: '', stderr: '' }; } });
  await control.draftingPlan({ configPath: 'C:/w/cfg.json', mode: 'auto' });
  await control.draftingPlan({ configPath: 'C:/w/cfg.json', mode: 'auto', unloading: true });
  await control.draftingFetch({ configPath: 'C:/w/cfg.json', mode: 'auto', unloading: true });
  assert.ok(!calls[0].includes('--engines-unloading'));
  assert.ok(calls[1].includes('--engines-unloading') && calls[2].includes('--engines-unloading'), calls.map(c => c.join(' ')).join(' | '));
});

/** Control as it answers on two cards that the app's own running engine holds: DFlash2 (and its fetch) unless the host says that engine is about to be unloaded. */
const busyWhileRunning = (get) => (options) => get().native.state().instances.length > 0 && options.unloading !== true ? DFLASH2_PLAN : { ...SPLIT_PLAN, mtp_needed: false, mtp_gb: 0 };

test('a model switch on two cards: the plan and the fetch say the running engine is about to be unloaded, so its memory is not busy and the split stays', async () => {
  let f;
  f = fixture({ plan: busyWhileRunning(() => f), split: splitAnswer });
  try {
    await f.native.refresh(true);
    const first = await finish(f.native, load(f).id);
    assert.equal(first.state, 'done', first.error);
    assert.equal(f.state.planCalls.length > 0 && f.state.planCalls.every(c => c.unloading !== true), true, 'nothing is running at the first load: nothing to unload');
    const before = f.state.planCalls.length;
    const second = await finish(f.native, load(f, { values: { drafting: 'mtp' } }).id);   // a different choice: a different engine, the first one is unloaded
    assert.equal(second.state, 'done', second.error);
    const calls = f.state.planCalls.slice(before);
    assert.ok(calls.length >= 1 && calls.every(c => c.unloading === true), JSON.stringify(calls));
    assert.equal(f.state.fetchCalls.length, 0, 'no DFlash2 drafter is fetched because of the old engine memory');
    assert.equal(f.state.splitCalls.length, 2, 'the split engine was asked at both loads');
    assert.equal(f.native.state().instances.length, 1);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a model added beside the running one keeps the memory of the running one busy: the plan is not told to ignore it', async () => {
  let f;
  f = fixture({ plan: busyWhileRunning(() => f), split: splitAnswer });
  try {
    await f.native.refresh(true);
    const first = await finish(f.native, load(f).id);
    assert.equal(first.state, 'done', first.error);
    const before = f.state.planCalls.length;
    await finish(f.native, load(f, { multiple: true, gpus: [0], values: { drafting: 'mtp' } }).id);
    const calls = f.state.planCalls.slice(before);
    assert.ok(calls.length >= 1 && calls.every(c => c.unloading !== true), JSON.stringify(calls));
  } finally { await f.native.close(); f.cleanup(); }
});

test('Auto on two cards: the MTP draft layer is fetched through the installer with the model\'s own folder, then the split engine\'s run config is what starts', async () => {
  const f = fixture({ plan: splitPlan, split: splitAnswer });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    // the plan: no --no-split, so Control's answer is the Terminal's
    assert.ok(f.state.planCalls.length >= 2);
    assert.ok(f.state.planCalls.every(c => c.noSplit !== true), 'the host never asks without the split here');
    // the layer: one installer call with drafter mtp, the model folder, and nothing that names a download of the model itself
    const real = f.state.rpc.filter(p => p.name === 'strata_install' && p.arguments?.confirm === true);
    assert.equal(real.length, 1);
    assert.equal(real[0].arguments.drafter, 'mtp');
    assert.equal(real[0].arguments.gguf_dir, dirname(f.gguf), 'the model\'s files folder, so the model is never fetched again');
    assert.equal(f.state.installs, 1);
    assert.equal(f.state.fetchCalls.length, 0, 'no DFlash2 drafter is fetched for the split');
    // the start: Control was asked once, with the layer split config the app built (4 slots, the app\'s own keys), and the profile holds the split engine\'s config
    assert.equal(f.state.splitCalls.length, 1);
    assert.equal(f.state.splitCalls[0].input.parallel, 4, 'the app\'s default of 4 slots reaches the split as 4 lanes');
    assert.equal(f.state.splitCalls[0].input.host, '127.0.0.1');
    assert.ok(f.state.splitCalls[0].input.api_key);
    assert.equal(f.state.saved.length, 1);
    assert.equal(f.state.saved[0].exe, '/e/strata-split-2-sm86');
    assert.equal(f.state.saved[0].args[f.state.saved[0].args.indexOf('--batch') + 1], '4');
    assert.equal(f.state.saved[0].drafting, 'mtp');
    assert.equal(f.state.saved[0].split_fallback.parallel, 4, 'the server\'s fallback keeps the app\'s slots');
    const instance = f.native.state().instances[0];
    assert.equal(instance.status, 'ready');
    assert.equal(instance.splitNote, 'Using the all-cards split engine on 2 cards.');
    assert.match(f.native.state().jobs.at(-1).details ?? '', /Using the all-cards split engine on 2 cards\./, 'the load says what runs');
  } finally { await f.native.close(); f.cleanup(); }
});

test('the layer is already on disk: no installer call, still the split', async () => {
  const f = fixture({ plan: { ...SPLIT_PLAN, mtp_needed: false, mtp_gb: 0 }, split: splitAnswer });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.rpc.filter(p => p.name === 'strata_install').length, 0);
    assert.equal(f.state.splitCalls.length, 1);
    assert.equal(f.state.saved[0].exe, '/e/strata-split-2-sm86');
  } finally { await f.native.close(); f.cleanup(); }
});

test('a person\'s DFlash2: the plan says layer split, DFlash2 is fetched as before, the split engine is never asked and the app keeps one request at a time', async () => {
  const f = fixture({ plan: DFLASH2_PLAN, split: splitAnswer });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f, { values: { drafting: 'dflash2' } }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.splitCalls.length, 0);
    assert.equal(f.state.rpc.filter(p => p.name === 'strata_install').length, 0, 'no draft layer download');
    assert.equal(f.state.fetchCalls.length, 1);
    assert.equal(f.state.fetchCalls[0].mode, 'dflash2');
    assert.ok(f.state.fetchCalls[0].noSplit !== true);
    assert.equal(f.state.saved[0].exe, 'component:strata-engine');
    assert.equal(f.state.saved[0].parallel, undefined, 'DFlash2 answers one request at a time');
  } finally { await f.native.close(); f.cleanup(); }
});

test('Off keeps the layer split and the split engine is not asked', async () => {
  const f = fixture({ plan: { ...DFLASH2_PLAN, choice: 'off', drafter: 'off', needed: false, gb: 0 }, split: splitAnswer });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f, { values: { drafting: 'off' } }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.splitCalls.length, 0);
    assert.ok(!f.state.saved[0].args.includes('--mtp'));
    assert.equal(f.state.saved[0].exe, 'component:strata-engine');
  } finally { await f.native.close(); f.cleanup(); }
});

test('the draft layer for the split cannot be downloaded: one plain note, the plan is DFlash2\'s again (asked with --no-split), the split is not tried and the model loads on the layer split', async () => {
  const f = fixture({
    installFails: true, split: splitAnswer,
    plan: (options) => options.noSplit === true ? DFLASH2_PLAN : { ...SPLIT_PLAN, mtp_needed: true, mtp_gb: 5.2 },
  });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f, { values: { drafting: null } }).id);   // the Models screen's Load with Auto
    assert.equal(job.state, 'done', job.error);
    assert.equal(job.drafterNote, 'The MTP drafter could not be downloaded.');
    assert.match(job.details ?? '', /The MTP drafter could not be downloaded\./);
    assert.equal(f.state.splitCalls.length, 0, 'no layer, no split');
    assert.ok(f.state.planCalls.some(c => c.noSplit === true), 'the plan is asked again as the app asked it in 0.2.10');
    assert.equal(f.state.fetchCalls.length, 1, 'the DFlash2 drafter is the fallback fetch');
    assert.equal(f.state.fetchCalls[0].noSplit, true, 'and Control is told, so it does not plan the split\'s layer again');
    assert.equal(f.state.saved.length, 1);
    assert.equal(f.state.saved[0].exe, 'component:strata-engine');
    assert.equal(f.native.state().instances[0].status, 'ready');
  } finally { await f.native.close(); f.cleanup(); }
});

test('a plain Auto restart with no draft layer on disk does not download it and does not stop: the load goes on, the split engine says no on its own', async () => {
  // no values: a restart or a tuning run carries no choice, and a saved explicit MTP is not Auto
  const f = fixture({ plan: { ...SPLIT_PLAN, mtp_needed: true, mtp_gb: 5.2 }, config: { drafting: 'mtp' } });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.rpc.filter(p => p.name === 'strata_install').length, 0, 'a restart never downloads');
    assert.equal(f.state.saved.length, 1);
  } finally { await f.native.close(); f.cleanup(); }
});

test('the split engine says no (a rule Control knows: engine package, RAM, context): the layer split config starts, nothing else changes', async () => {
  const f = fixture({ plan: { ...SPLIT_PLAN, mtp_needed: false, mtp_gb: 0 } });   // splitEngine answers split:false
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.splitCalls.length, 1);
    assert.equal(f.state.saved[0].exe, 'component:strata-engine');
    assert.equal(f.state.saved[0].parallel, 4);
    assert.equal(f.native.state().instances[0].splitNote, undefined);
  } finally { await f.native.close(); f.cleanup(); }
});

test('Control cannot be asked about the split: the layer split starts as it was', async () => {
  const f = fixture({ plan: { ...SPLIT_PLAN, mtp_needed: false, mtp_gb: 0 }, split: async () => { throw new Error('Control is busy'); } });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.saved[0].exe, 'component:strata-engine');
  } finally { await f.native.close(); f.cleanup(); }
});

test('the server started the layer split because the split engine would not come up: the load says so in one line', async () => {
  const f = fixture({ plan: { ...SPLIT_PLAN, mtp_needed: false, mtp_gb: 0 }, split: splitAnswer, metrics: { split_fallback: true } });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    const lines = (job.details ?? '').split('\n');
    assert.ok(lines.includes('The all-cards split engine could not start, so the normal layer split runs.'), job.details);
    assert.ok(!lines.some(l => /^Using the all-cards split engine/.test(l)), 'not the line that said the split would run');
  } finally { await f.native.close(); f.cleanup(); }
});

test('the second try after a failed start is never the split: one request at a time on the layer split', async () => {
  const f = fixture({ plan: { ...SPLIT_PLAN, mtp_needed: false, mtp_gb: 0 }, split: splitAnswer, startFails: 1 });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.saved.length, 2);
    assert.equal(f.state.saved[0].exe, '/e/strata-split-2-sm86');
    assert.equal(f.state.saved[1].exe, 'component:strata-engine');
    assert.equal(f.state.saved[1].parallel, undefined);
    assert.equal(f.state.splitCalls.length, 1, 'Control was asked once');
    assert.equal(f.native.state().instances[0].splitNote, undefined, 'the line that said the split would run is gone');
  } finally { await f.native.close(); f.cleanup(); }
});

test('the word "split" reaches the Models screen: Auto is the MTP where Control says split, DFlash2 where it does not, and a person\'s DFlash2 stays', () => {
  for (const [choice, split, want] of [['auto', true, 'mtp'], ['auto', false, 'dflash2'], ['dflash2', true, 'dflash2'], ['mtp', true, 'mtp'], ['off', true, 'off']]) {
    assert.equal(hostResolve(choice, 'unsloth', true, split), want, `${choice} split=${split}`);
  }
});

test('the hint of the Drafting setting and the Models screen say what Auto is on the split, with no new element', () => {
  const read = rel => readFileSync(new URL(rel, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  assert.match(read('../server/native-settings.mjs'), /on two or more identical graphics cards that the all-cards split engine takes, it is that engine with the MTP draft layer/);
  const screen = read('../src/components/NativeModels.tsx');
  assert.match(screen, /const splitCards = native\?\.capabilities\?\.dflash2Split === true && cardsGpus\.length !== 1;/, 'one card chosen in the roles is not the split');
  assert.ok(screen.indexOf('const splitCards =') > 0 && screen.indexOf('const splitCards =') < screen.indexOf('const nowDrafter ='), 'splitCards is declared before it is used');
  assert.match(screen, /splitCards && drafting === 'auto' \? "Auto uses the all-cards split engine with the MTP draft layer: about \{0\} GB to download, once\."/);
  const en = JSON.parse(read('../src/locales/df2-en.json'));
  const ids = Object.entries(en).filter(([, v]) => /all-cards split engine/.test(v)).map(([k]) => k);
  assert.equal(ids.length, 2);
  for (const lang of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he']) {
    const cat = JSON.parse(read(`../src/locales/df2-${lang}.json`));
    for (const id of ids) assert.ok(cat[id] && cat[id] !== en[id], `${id} in ${lang}`);
    assert.equal((cat[ids[0]].match(/\{0\}/g) ?? []).length, 1, `the size slot in ${lang}`);
  }
});
