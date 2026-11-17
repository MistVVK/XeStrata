/**
 * 0.2.13: the app fetches the tuned DFlash2 drafter on every path where Auto resolves to it (a plain load, a restart of the stopped models, a tuning run), like the Models screen
 * own Load does, with the 12 hour rule (--throttle); a drafter the person named or saved wins; the plan and the Settings hint carry the size of the drafter that is really fetched.
 * The decision is Control (strata drafting plan / fetch; tests/terminal/test_df2ux_0213.py); this file drives the host with a fake Control.
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
function fixture({ fetchResult = { ok: true, event: 'result', pack: 'C:/d/dflash2/x.pack', skipped: false }, plan, split = null, installFails = false, startFails = 0, metrics = {}, config = {} } = {}) {
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
    async draftingFetch(options) { state.fetchCalls.push(options); if (fetchResult instanceof Error) throw fetchResult; return fetchResult; },
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


const TUNED_PLAN = { ok: true, split: false, choice: 'dflash2', drafter: 'dflash2', needed: true, tuned_needed: true, tuned: true, gb: 0.34, drafter_gb: 0.34, mtp_needed: false, mtp_gb: 0, line: 'Drafting: DFlash2', serves: true };
const READY_PLAN = { ...TUNED_PLAN, needed: false, tuned_needed: false, gb: 0 };
const planAfterFetch = (_o, state) => (state.fetchCalls.length ? READY_PLAN : TUNED_PLAN);

test('a plain load fetches the tuned drafter with the 12 hour rule, then plans again', async () => {
  const f = fixture({ plan: planAfterFetch });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.fetchCalls.length, 1);
    assert.equal(f.state.fetchCalls[0].mode, 'auto');
    assert.equal(f.state.fetchCalls[0].throttle, true, 'nobody asked for it: the Terminal 12 hour rule');
    assert.ok(f.state.planCalls.length >= 2, 'planned again after the fetch');
  } finally { await f.native.close(); f.cleanup(); }
});

/* CLEAN-1A (owner 11-07): the plain Load follows Auto everywhere. Where Auto's drafter is the standard DFlash2 pack (Intel Arc, UD-Q4_K_XL on NVIDIA 72 GB / 12 GB and the Spark), Control says `needed` without `tuned_needed`; the app fetches it just the same, asking Control. */
const STOCK_PLAN = { ...TUNED_PLAN, tuned_needed: false, tuned: false, gb: 1.2, drafter_gb: 1.2 };
const stockAfterFetch = (_o, state) => (state.fetchCalls.length ? { ...STOCK_PLAN, needed: false, gb: 0 } : STOCK_PLAN);

test('a plain load fetches the standard DFlash2 drafter too when that is what Auto runs', async () => {
  const f = fixture({ plan: stockAfterFetch });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.fetchCalls.length, 1);
    assert.equal(f.state.fetchCalls[0].mode, 'auto');
    assert.equal(f.state.fetchCalls[0].throttle, true);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a plain load fetches nothing when Auto needs no download', async () => {
  const f = fixture({ plan: () => ({ ...STOCK_PLAN, needed: false, gb: 0 }) });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.fetchCalls.length, 0);
  } finally { await f.native.close(); f.cleanup(); }
});

test('the Models screen own Load keeps its un-throttled fetch and does not fetch twice', async () => {
  const f = fixture({ plan: planAfterFetch });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f, { values: { drafting: 'auto' } }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.fetchCalls.length, 1);
    assert.ok(f.state.fetchCalls[0].throttle !== true);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a drafter the person saved wins: MTP, off and DFlash2 are never fetched by a plain load', async () => {
  for (const drafting of ['mtp', 'off', 'dflash2']) {
    const f = fixture({ plan: TUNED_PLAN, config: { drafting } });
    try {
      await f.native.refresh(true);
      const job = await finish(f.native, load(f).id);
      assert.equal(job.state, 'done', job.error);
      assert.equal(f.state.fetchCalls.length, 0, drafting);
    } finally { await f.native.close(); f.cleanup(); }
  }
});

test('nothing needed (the tuned drafter is here, or Control says MTP): a plain load fetches nothing', async () => {
  const f = fixture({ plan: READY_PLAN });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.fetchCalls.length, 0);
  } finally { await f.native.close(); f.cleanup(); }
});

test('the feed has nothing (skipped): the load goes on exactly as before, no note', async () => {
  const f = fixture({ plan: TUNED_PLAN, fetchResult: { ok: true, event: 'result', pack: null, skipped: true } });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.state.fetchCalls.length, 1);
    assert.ok(!job.drafterNote);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a fetch that fails for a real reason never stops the load and the job says why', async () => {
  const f = fixture({ plan: TUNED_PLAN, fetchResult: new Error('The tuned DFlash2 drafter did not pass the safety check, so it was deleted.') });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, load(f).id);
    assert.equal(job.state, 'done', job.error);
    assert.match(job.drafterNote ?? '', /safety check/);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a tuning run fetches it first, and not for a saved MTP', async () => {
  for (const [drafting, calls] of [[undefined, 1], ['mtp', 0]]) {
    const f = fixture({ plan: TUNED_PLAN, config: drafting ? { drafting } : {} });
    try {
      await f.native.refresh(true);
      const id = f.native.operation('autotune', { model: 'ud_q4_k_xl' }).id;
      await finish(f.native, id);
      assert.equal(f.state.fetchCalls.length, calls, String(drafting));
      if (calls) assert.equal(f.state.fetchCalls[0].throttle, true);
    } finally { await f.native.close(); f.cleanup(); }
  }
});

test('the install plan asks Control about this model and lists the tuned size, and says nothing for MTP', async () => {
  const f = fixture({ plan: (o) => (o.mode === 'mtp' ? { ...READY_PLAN, choice: 'mtp', drafter: 'mtp' } : TUNED_PLAN) });
  try {
    await f.native.refresh(true);
    const got = await f.native.plan({ family: 'qwen', model: 'UD-Q4_K_XL', drafting: 'auto' });
    assert.equal(f.state.planCalls.find(c => c.model)?.model, 'UD-Q4_K_XL');
    assert.equal(got.drafting.gb, 0.34);
    assert.equal(got.drafting.tuned, true);
    const none = await f.native.plan({ family: 'qwen', model: 'UD-Q4_K_XL', drafting: 'mtp' });
    assert.equal(none.drafting, undefined, 'MTP chosen: no drafter is listed');
  } finally { await f.native.close(); f.cleanup(); }
});

test('the Settings hint keeps its wording and changes only the number', async () => {
  const { draftingHint, NATIVE_FIELDS } = await import('../server/native-settings.mjs');
  assert.equal(NATIVE_FIELDS.find(x => x.id === 'drafting').hint, draftingHint(1.2));
  assert.equal(draftingHint(0.34), draftingHint(1.2).replace('1.2 GB', '0.34 GB'));
  assert.match(draftingHint(0.52), /DFlash2 downloads a 0\.52 GB drafter once, the MTP its draft layer once\./);
  assert.equal(draftingHint(0), draftingHint(1.2), 'no number from Control: the generic hint');
  const f = fixture({ plan: TUNED_PLAN });
  try {
    await f.native.refresh(true);
    assert.deepEqual(await f.native.settingsHints('ud_q4_k_xl'), { drafting: draftingHint(0.34) });
    assert.equal(f.state.planCalls.at(-1).mode, 'dflash2');
  } finally { await f.native.close(); f.cleanup(); }
});

test('control passes the model, the family and --throttle to Control and not --model beside a run config', async () => {
  const calls = [];
  const control = createControl({ control: { dir: 'C:/c', python: 'py', cli: 'strata.py' }, run: async (args) => { calls.push(args); return { code: 0, json: { ok: true }, text: '', stderr: '' }; } });
  await control.draftingPlan({ mode: 'auto', family: 'unsloth', model: 'UD-Q4_K_XL', dataDir: 'C:/d' });
  assert.deepEqual(calls[0], ['drafting', 'plan', '--mode', 'auto', '--family', 'unsloth', '--model', 'UD-Q4_K_XL', '--data-dir', 'C:/d', '--output', 'json']);
  await control.draftingPlan({ configPath: 'C:/w/c.json', model: 'UD-Q4_K_XL' });
  assert.ok(!calls[1].includes('--model'));
  await control.draftingFetch({ configPath: 'C:/w/c.json', mode: 'auto', throttle: true });
  assert.deepEqual(calls[2], ['drafting', 'fetch', '--config', 'C:/w/c.json', '--mode', 'auto', '--if-needed', '--throttle', '--output', 'jsonl']);
});

test('a restart of the stopped models fetches it before the start, and so does every path through the one helper', async () => {
  const { readFileSync: read } = await import('node:fs');
  const source = read(new URL('../server/native-control.mjs', import.meta.url), 'utf8');
  const restart = source.slice(source.indexOf('async function restartStopped'), source.indexOf('function dismissRestart'));
  assert.match(restart, /await autoDrafterFor\(job,instance\.model,[^)]*\);[\s\S]*await control\.start\(instance\.profile/, 'before the engine starts again');
  const tune = source.slice(source.indexOf('async function autotune'), source.indexOf('function operation'));
  assert.match(tune, /await autoDrafterFor\(job,args\.model,/);
  assert.match(source, /if\(args\.values\?\.drafting===undefined&&drafts\?\.needed===true\)\{const got=await fetchAutoDrafter\(/, 'a plain load');
  assert.match(source, /!autoChoice\(readJson\(configFile\)\.drafting\)\)return false/, 'a saved choice wins');
});
