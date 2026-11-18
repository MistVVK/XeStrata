import './helpers/isolate-profile.mjs';   // a test run never writes the owner's real Strata profile
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { existsSync } from 'node:fs';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync, realpathSync } from 'node:fs';
import { join, basename } from 'node:path';
import { tmpdir as osTmpdir } from 'node:os';
// the canonical scratch root: on a Mac the temp folder is /var/folders/..., which resolves to /private/var/...; the paths the host reports are the resolved ones
const tmpdir = () => realpathSync(osTmpdir());
import { bundledPython, componentsDirectory, controlEnvironment, createControl, findControl, runCli } from '../server/control-runtime.mjs';
import { resolvePython } from '../server/python-resolver.mjs';
import { createNativeControl } from '../server/native-control.mjs';

const isWin = process.platform === 'win32';

function fakeControlDir() {
  const dir = mkdtempSync(join(tmpdir(), 'strata-control-dir-'));
  mkdirSync(join(dir, 'python', ...(isWin ? [] : ['bin'])), { recursive: true });
  mkdirSync(join(dir, 'strata_control'), { recursive: true });
  mkdirSync(join(dir, 'serve'), { recursive: true });
  writeFileSync(bundledPython(dir), 'x');
  writeFileSync(join(dir, 'strata.py'), '# cli');
  writeFileSync(join(dir, 'serve', 'server.py'), 'port\nparallel\n/v1/responses');
  writeFileSync(join(dir, 'BUNDLE.json'), JSON.stringify({ version: '0.1.45', python: '3.12.15' }));
  return dir;
}

test('the Control component is found by STRATA_CONTROL_DIR or beside the host, and only if it is whole', () => {
  const dir = fakeControlDir();
  // The explicit env has no STRATA_COMPONENTS_DIR, so findControl would look in the real per-user components folder and find an updated Control installed on this PC.
  const none = join(dir, 'no-components');
  try {
    const found = findControl({ env: { STRATA_CONTROL_DIR: dir }, hostDir: join(dir, 'nowhere'), components: none });
    assert.equal(found.dir, dir);
    assert.equal(found.python, bundledPython(dir));
    assert.equal(found.bundle.python, '3.12.15');
    // beside the host: <install>/host + <install>/control
    const install = mkdtempSync(join(tmpdir(), 'strata-install-'));
    try {
      mkdirSync(join(install, 'host', 'server'), { recursive: true });
      assert.equal(findControl({ env: {}, hostDir: join(install, 'host'), components: none }), null, 'nothing beside the host yet');
      mkdirSync(join(install, 'control', 'python', ...(isWin ? [] : ['bin'])), { recursive: true });
      mkdirSync(join(install, 'control', 'strata_control'));
      writeFileSync(bundledPython(join(install, 'control')), 'x');
      writeFileSync(join(install, 'control', 'strata.py'), '#');
      assert.equal(findControl({ env: {}, hostDir: join(install, 'host'), components: none }).dir, join(install, 'control'));
    } finally { rmSync(install, { recursive: true, force: true }); }
    rmSync(join(dir, 'strata.py'));
    assert.equal(findControl({ env: { STRATA_CONTROL_DIR: dir }, hostDir: join(dir, 'x'), components: none }), null, 'a half component is not used');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("Control runs on its own interpreter with nothing from the user's Python and acquiring forbidden", () => {
  const control = { dir: '/c', python: '/c/python/python3', cli: '/c/strata.py' };
  const env = controlEnvironment(control, { env: { PYTHONHOME: 'x', PYTHONPATH: 'y', VIRTUAL_ENV: 'z', STRATA_PYTHON: 'q', PATH: 'p', STRATA_API_KEY: 'secret' }, components: '/components' });
  assert.equal(env.PYTHONHOME, undefined);
  assert.equal(env.PYTHONPATH, undefined);
  assert.equal(env.VIRTUAL_ENV, undefined);
  assert.equal(env.STRATA_PYTHON, undefined);
  assert.equal(env.PYTHONNOUSERSITE, '1');
  assert.equal(env.STRATA_NO_ACQUIRE, '1');
  assert.equal(env.STRATA_COMPONENTS_DIR, '/components');
  assert.equal(env.STRATA_API_KEY, undefined, 'an inherited inference key never reaches the engine (the run config carries its own)');
  assert.equal(env.PATH, 'p');
  // the resolver prefers the bundled interpreter over py -3 / PATH / a venv
  const python = resolvePython({ repo: '/repo', control, env: {}, exists: () => true, starts: () => { throw new Error('must not probe'); }, useCache: false });
  assert.deepEqual([python.command, python.source], ['/c/python/python3', 'bundled']);
  assert.equal(resolvePython({ control, env: { STRATA_PYTHON: '/mine' }, useCache: false }).command, '/mine', 'an explicit choice still wins');
  assert.equal(componentsDirectory({ env: { STRATA_COMPONENTS_DIR: '/x' } }), '/x');
  assert.match(componentsDirectory({ env: { LOCALAPPDATA: 'C:\\L' }, platform: 'win32', home: 'C:\\H' }), /Strata[\\/]components$/);
});

test('runCli passes the arguments untouched (no shell) and reads JSON', async () => {
  const seen = [];
  const spawnProcess = (command, args, options) => {
    seen.push({ command, args, options });
    const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter(); child.kill = () => {};
    queueMicrotask(() => { child.stdout.emit('data', '{"ok": true, "n": 1}\n'); child.emit('close', 0); });
    return child;
  };
  const control = { dir: '/c', python: '/c/python/python3', cli: '/c/strata.py' };
  const result = await runCli(control, ['models', 'list', '--installed', '--output', 'json'], { spawnProcess, env: {} });
  assert.deepEqual(result.json, { ok: true, n: 1 });
  assert.deepEqual(seen[0].args, ['/c/strata.py', 'models', 'list', '--installed', '--output', 'json']);
  assert.equal(seen[0].options.shell, undefined);
  assert.equal(seen[0].options.windowsHide, true);
  assert.equal(seen[0].options.env.STRATA_NO_ACQUIRE, '1');
});

// ---------------------------------------------------------------------------------- native-control in packaged mode

function packagedFixture({ engineUsable = true, updater = null, repair = false, ids = ['iq2_xs', 'iq3_s'], controlVersion = '0.1.45', extraSizes = [], cliStartWait = 'old', installPoll = null, rpcHook = null } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-packaged-test-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  /* The Control this app found, as it is on disk. The app reads the start-wait ceiling out of these files instead of trusting
     the version, because a start wait is checked three times before the engine is reached and a released Control has those
     three ceilings disagreeing:
       'old'        no runtime ceiling at all, both gates 600 (Control before #3949)
       'split'      the runtime takes 1800, the CLI and the record still stop at 600 (every Control 0.1.50 to 0.1.52)
       'cli-only'   the CLI follows the runtime, the record still stops at 600 (`direct runtime intent has an invalid timeout`)
       'runtime'    all three hand the wait to the runtime's own ceiling */
  const controlSource = join(root, 'control', 'strata_control');
  mkdirSync(controlSource, { recursive: true });
  writeFileSync(join(controlSource, 'runtime.py'), cliStartWait === 'old' ? '' : 'MAX_START_WAIT = 1800\n');
  writeFileSync(join(controlSource, 'management.py'), ['cli-only', 'runtime'].includes(cliStartWait)
    ? '        wait = _seconds(wait, 0, runtime.MAX_START_WAIT, "start wait")\n'
    : '        wait = _seconds(wait, 0, 600, "start wait")\n');
  writeFileSync(join(controlSource, 'journal.py'), cliStartWait === 'runtime'
    ? '        ranges = {"wait_s": (0, start_wait_max) if kind == "runtime.start" else\n'
    : '        ranges = {"wait_s": (0, 600) if kind == "runtime.start" else\n');
  const earlierInstallEnv = process.env.STRATA_INSTALL_DIR;
  process.env.STRATA_INSTALL_DIR = install;      // the app's own install folder (the host always tells Control which one; it never falls back to an older CLI folder)
  const configs = ids.map(id => {
    const path = join(install, `strata-${id}.json`);
    writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: ['--max-context', '32768'], model_name: `Qwen ${id}`, port: 8080 }));
    return { id, path };
  });
  const calls = [];
  const saved = [];
  const spawned = [];
  const rpc = [];
  let startFailure = null, startFailureLeft = Infinity;
  let statusAnswer = { state: 'ready' }, metricsDown = false, healthDown = false;
  const control = {
    control: { dir: join(root, 'control'), python: join(root, 'control', 'python', 'python.exe') }, version: controlVersion,
    async installedModels(options) { calls.push(['installedModels', options?.installDir ?? null]); return { install_dir: install, models: configs.map(c => ({ id: c.id, name: `Qwen ${c.id}`, source_config: c.path, component_status: 'present', context_tokens: 32768, backend: 'cuda', ...(/^deepseek/.test(c.id) ? { component_status: 'retired', retired: 'deepseek', name: 'DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw', retired_line: 'DeepSeek - no longer supported.', refusal: 'DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw is a DeepSeek model. This version of Strata no longer runs DeepSeek models, so it was not started. Choose a Qwen model to run instead. If you no longer need the DeepSeek files, you can delete that folder to free the space.' } : {}) })) }; },
    async engine() {
      calls.push(['engine']);
      return engineUsable
        ? { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.1.45', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\engine\\strata.exe', root: 'E:\\engine' } }
        : { usable: false, notice: 'No engine for this PC was found in E:\\c: the component folder holds no engine package. The app never downloads an engine while it starts or serves.', backend: 'cuda', engine: null, components_dir: 'E:\\c' };
    },
    async importPlan(path) { calls.push(['importPlan', path]); return { file_set_status: 'plausible', candidate_selection: { family: 'qwen', model: 'IQ3_S' }, file_count: 2, total_bytes: 1e9, issues: [] }; },
    async saveProfile(name, path) { calls.push(['saveProfile', name, basename(path)]); saved.push(JSON.parse(readFileSync(path, 'utf8'))); return {}; },
    async start(name, options) { calls.push(['start', name, options?.waitSeconds]); if (startFailure && startFailureLeft-- > 0) { const e = new Error(startFailure.message); e.output = startFailure.output; throw e; } return { ok: true, state: 'ready' }; },
    async stop(name) { calls.push(['stop', name]); return { ok: true, state: 'stopped' }; },
    async killChildren() { calls.push(['killChildren']); return 0; },
    async status(name) { calls.push(['status', name]); return typeof statusAnswer === 'function' ? statusAnswer(name) : statusAnswer; },
    ...(repair ? { async repairAccess() { calls.push(['repairAccess']); return { ok: true }; } } : {}),
    ...(updater ? { async updateCheck() { calls.push(['updateCheck']); return updater.report; }, async update(options) { calls.push(['update', options.engine, options.control]); return updater.update(options); } } : {}),
  };
  const requests = [];
  const fetcher = async url => {
    calls.push(['fetch', url.replace(/:\d+/, ':PORT')]);
    if (metricsDown && /\/(metrics|mcp)/.test(url)) throw new Error('the engine does not answer');
    if (healthDown && /\/health/.test(url)) throw new Error('the engine does not answer');
    return Response.json({ status: 'ok' });
  };
  // The only process the host may start is Control's installer (its bundled tools/strata_mcp.py on Control's Python); the fake
  // answers like it: the catalog, the hardware with the GPUs, and an install plan or job.
  const catalog = { families: [{ family: 'qwen', title: 'Qwen3.8-Flash-Next', about: 'x', images: true, experimental: false, sizes: [
    { id: 'q2_0', model: 'Q2_0', download_gb: 66.4, ram_needed_gb: 48, about: 'fast', on_this_pc: 'fits', installed: false, images: true, experimental: false },
    { id: 'iq3_xxs', model: 'IQ3_XXS', download_gb: 75.8, ram_needed_gb: 60, about: 'balanced', on_this_pc: 'fits', installed: false, images: true, experimental: false }, ...extraSizes] }],
    contexts: [8192, 32768], recommendation: { family: 'qwen', model: 'IQ3_XXS', context: 32768 }, context_note: '' };
  const hardware = { ram_gb: 63, cpu: { name: 'cpu', cores: 12 }, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5070', vram_gb: 11.9, usable: true }] };
  let installJob = null;
  const spawnProcess = (command, args, options) => {
    spawned.push({ command, args, options });
    if (!String(args[0]).endsWith('strata_mcp.py')) throw new Error('no process but the installer may be started by the host in packaged mode: ' + command + ' ' + args.join(' '));
    const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter(); child.exitCode = null;
    child.kill = () => { child.exitCode = 0; queueMicrotask(() => child.emit('exit', 0)); return true; };
    child.stdin = { write(line) {
      const message = JSON.parse(line); if (!message.id) return;
      if (message.method === 'tools/call') rpc.push(message.params);
      let result;
      if (message.method === 'initialize') result = { protocolVersion: '2024-11-05' };
      else if (message.method === 'tools/list') result = { tools: ['strata_models', 'strata_status', 'strata_install', 'strata_remove'].map(name => ({ name })) };
      else if (message.params.name === 'strata_models') result = { structuredContent: catalog };
      else if (message.params.name === 'strata_remove') result = { structuredContent: { summary: 'Deleted', plan: { gb: 1.2 }, failed: [] } };
      else if (message.params.name === 'strata_status') result = { structuredContent: { hardware, recommendation: catalog.recommendation, install: installJob, running: [], summary: 'x' } };
      else if (message.params.arguments?.confirm) { installJob = { running: false, exit_code: 0 }; result = { structuredContent: { summary: 'Install started', plan: { download_gb: 66.4 }, install: { running: true, step: 'downloading' } } }; }
      else if (message.params.arguments?.cancel === undefined && message.params.arguments?.confirm === undefined) result = { structuredContent: installPoll ?? { summary: 'Install finished', install: { running: false, exit_code: 0 } } };
      else result = { structuredContent: { summary: 'Plan', plan: { title: 'Qwen3.8-Flash-Next Q2_0', download_gb: 66.4, disk_short: null } } };
      const deliver = answer => queueMicrotask(() => child.stdout.emit('data', Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: message.id, result: answer }) + String.fromCharCode(10))));
      /* a test can take over the answer to a tool call (to answer later, or differently): rpcHook(params, usualAnswer, deliver) returns true when it will deliver */
      if (rpcHook && message.method === 'tools/call' && rpcHook(message.params, result, deliver)) return;
      deliver(result);
    } };
    return child;
  };
  const native = createNativeControl({ control, directory: data, spawnProcess, fetcher, requests: () => requests });
  const others = () => spawned.filter(s => !String(s.args[0]).endsWith('strata_mcp.py'));
  return { native, calls, saved, spawned, others, rpc, setStatus: v => { statusAnswer = v; }, setMetricsDown: v => { metricsDown = v; }, setHealthDown: v => { healthDown = v; }, failStart: (v, times = Infinity) => { startFailure = v; startFailureLeft = times; }, catalog, hardware, install, root, cleanup: () => { if (earlierInstallEnv === undefined) delete process.env.STRATA_INSTALL_DIR; else process.env.STRATA_INSTALL_DIR = earlierInstallEnv; rmSync(root, { recursive: true, force: true }); } };
}
const finish = async (native, id) => { for (let i = 0; i < 400; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };

test('packaged: models come from Control, no python, git or engine process is started by the host', async () => {
  const f = packagedFixture();
  try {
    const state = await f.native.refresh(true);
    assert.equal(state.available, true, state.error);
    assert.deepEqual(state.installed.models.map(m => m.model), ['iq2_xs', 'iq3_s']);
    assert.equal(state.installed.models[0].ready, true);
    assert.equal(state.packaged, true);
    assert.match(state.summary, /engine 0\.1\.45/);
    assert.equal(state.version, '0.1.45');
    assert.equal(f.others().length, 0, 'nothing but the installer is started');
    // the catalog, the GPUs and the recommendation come from the bundled installer, run on Control's own Python
    assert.equal(state.catalog.families[0].sizes.length, 2);
    assert.equal(state.catalog.recommendation.model, 'IQ3_XXS');
    assert.equal(state.hardware.gpus[0].name, 'NVIDIA GeForce RTX 5070');
    assert.equal(state.hardware.ram_gb, 63);
    const installer = f.spawned[0];
    assert.equal(installer.command, join(f.root, 'control', 'python', 'python.exe'));
    assert.deepEqual(installer.args.slice(1), ['--root', f.install, '--src', join(f.root, 'control')]);
    assert.equal(installer.options.env.STRATA_CONTROL_INSTALL, '1');
    assert.equal(installer.options.env.STRATA_NO_ACQUIRE, '1', 'the installer process itself may not acquire; only its explicit install job does');
    assert.equal(installer.options.env.STRATA_INSTALL_DIR, f.install);
    assert.equal(installer.options.env.PYTHONNOUSERSITE, '1');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: load = saved profile + runtime start under Control; unload = runtime stop; two models get two profiles', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    const load = async args => { const job = await finish(f.native, f.native.operation('load', args).id); assert.equal(job.state, 'done', job.error); return job; };
    await load({ model: 'iq2_xs' });
    const first = f.native.state().instances[0];
    assert.equal(first.status, 'ready');
    const startCalls = f.calls.filter(c => ['saveProfile', 'start'].includes(c[0]));
    assert.deepEqual(startCalls.map(c => c.slice(0, 2)), [['saveProfile', 'app-1'], ['start', 'app-1']]);
    assert.equal(startCalls[1][2], 300);
    await load({ model: 'iq3_s', multiple: true });
    assert.deepEqual(f.calls.filter(c => c[0] === 'start').map(c => c[1]), ['app-1', 'app-2']);
    assert.equal(f.native.state().instances.length, 2);
    await f.native.unload(first.id);
    assert.deepEqual(f.calls.filter(c => c[0] === 'stop').at(-1), ['stop', 'app-1']);
    assert.equal(f.native.state().instances.length, 1);
    assert.equal(f.others().length, 0, 'the host never spawned anything itself');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: a DeepSeek model of an earlier Strata (an id with dots) is listed with its one line; its load is refused with Control\'s words and starts nothing', async () => {
  const id = 'deepseek-v4.1-flash-ist-das-2.00bpw';
  const f = packagedFixture({ ids: [id] });
  try {
    const path = join(f.install, `strata-${id}.json`);
    writeFileSync(path, JSON.stringify({ exe: join(f.install, 'engine-dsv41', 'strata-dsv41'), arch: 'deepseek41', tokenizer: join(f.install, 'tok'), args: ['--pack', join(f.install, 'pack'), '--max-context', '8192', '--eos', '1'], model_name: 'deepseek-v4.1-flash', port: 8080, engine_ready_s: 3600 }));
    const before = readFileSync(path, 'utf8');
    const state = await f.native.refresh(true);
    assert.deepEqual(state.installed.models.map(m => m.model), [id], 'listed');
    assert.equal(state.installed.models[0].ready, false);
    assert.equal(state.installed.models[0].retired.line, 'DeepSeek - no longer supported.');
    const job = await finish(f.native, f.native.operation('load', { model: id }).id);
    assert.equal(job.state, 'error');
    assert.match(String(job.error), /^DeepSeek-V4\.1-Flash ISTA-DAS 2\.00 bpw is a DeepSeek model\. This version of Strata no longer runs DeepSeek models, so it was not started\./);
    assert.deepEqual(f.calls.filter(c => ['saveProfile', 'start'].includes(c[0])), [], 'nothing is saved or started');
    assert.equal(readFileSync(path, 'utf8'), before, 'its settings file is not written');
    for (const odd of ['a..b', '../x', 'x/y', '.hidden']) assert.equal((await finish(f.native, f.native.operation('load', { model: odd }).id)).state, 'error', odd);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: no engine for this PC is the notice, and nothing is started', async () => {
  const f = packagedFixture({ engineUsable: false });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /No engine for this PC/);
    assert.match(job.error, /never downloads an engine/);
    assert.equal(f.calls.some(c => ['saveProfile', 'start'].includes(c[0])), false);
    assert.equal(f.others().length, 0);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: the user can download and prepare a model: the plan and the install job go to the bundled installer, the engine is not touched', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    const plan = await f.native.plan({ family: 'qwen', model: 'Q2_0' });
    assert.equal(plan.plan.title, 'Qwen3.8-Flash-Next Q2_0');
    const planned = f.rpc.at(-1);
    assert.equal(planned.name, 'strata_install');
    assert.equal(planned.arguments.confirm, false, 'a plan changes nothing');
    const job = await finish(f.native, f.native.operation('install', { family: 'qwen', model: 'Q2_0' }).id);
    assert.equal(job.state, 'done', job.error);
    const started = f.rpc.filter(c => c.name === 'strata_install' && c.arguments.confirm);
    assert.equal(started.length, 1);
    assert.equal(started[0].arguments.family, 'qwen');
    assert.equal(started[0].arguments.model, 'Q2_0');
    assert.equal(f.others().length, 0, 'the host starts no pip, git, python of its own or engine');
    assert.equal(f.calls.filter(c => ['saveProfile', 'start'].includes(c[0])).length, 0, 'installing does not start an engine');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: the installer works in a folder the app can write, never in the read-only bundle', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    assert.equal(existsSync(f.install), true);
    assert.notEqual(f.native.state().installDir, join(f.root, 'control'));
    assert.equal(f.spawned[0].args.at(-3), f.install);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: update, rollback and calibration are refused with the Terminal notice and start nothing', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    for (const [kind, args] of [['update-engine', {}], ['update-strata', {}], ['rollback', {}], ['calibrate', { model: 'iq3_s' }]]) {
      const job = await finish(f.native, f.native.operation(kind, args).id);
      assert.equal(job.state, 'error', kind);
      assert.match(job.error, /Terminal|calibrates NVIDIA|No saved previous engine/, `${kind}: ${job.error}`);
    }
    assert.equal(f.others().length, 0);
    assert.equal(f.calls.filter(c => ['saveProfile', 'start'].includes(c[0])).length, 0);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: the install folder can be chosen, and it is what Control is asked about', async () => {
  const f = packagedFixture();
  try {
    const other = join(f.root, 'other');
    mkdirSync(other);
    await f.native.configure({ repository: other });
    const saved = JSON.parse(readFileSync(join(f.root, 'data', 'native-host.json'), 'utf8'));
    assert.equal(saved.installDir.toLowerCase(), other.toLowerCase());
    assert.equal(f.calls.filter(c => c[0] === 'installedModels').at(-1)[1].toLowerCase(), other.toLowerCase());
  } finally { await f.native.close(); f.cleanup(); }
});

test('createControl asks the CLI for the engine, models and runtime and reads the notice of a missing engine', async () => {
  const seen = [];
  const control = createControl({
    control: { dir: '/c', python: '/p', cli: '/c/strata.py', bundle: { version: '9' } },
    run: async args => { seen.push(args.join(' ')); return args[0] === 'components' ? { code: 5, json: { ok: false, usable: false, notice: 'No engine for this PC' }, text: '', stderr: '' } : { code: 0, json: { ok: true, profiles: [] }, text: '', stderr: '' }; },
  });
  assert.equal((await control.engine()).notice, 'No engine for this PC');
  await control.installedModels({ installDir: 'D:\\Strata' });
  await control.saveProfile('app-1', 'C:\\x.json');
  await control.start('app-1');
  await control.stop('app-1');
  assert.deepEqual(seen, [
    'components engine --output json',
    'models list --installed --install-dir D:\\Strata --output json',
    'profile list --output json',
    'profile create app-1 --from-config C:\\x.json --output json',
    `runtime start --profile app-1 --wait 300 --shutdown-timeout 15 --die-with-pid ${process.pid} --output json`,
    'runtime stop --profile app-1 --wait 120 --output json',
  ]);
  assert.equal(createControl({ control: null }), null);
});

test('packaged: the catalog offers no picture reading, an install asks for none by default and for it only when Read pictures was ticked', async () => {
  const f = packagedFixture();
  try {
    const state = await f.native.refresh(true);
    assert.ok(state.catalog.families[0].sizes.every(s => s.images === false));
    assert.ok(state.catalog.families.every(fam => fam.images === false));
    await f.native.plan({ family: 'qwen', model: 'Q2_0' });
    assert.equal(f.rpc.at(-1).arguments.vision, 'no');
    await f.native.plan({ family: 'qwen', model: 'Q2_0', vision: 'yes' });
    assert.equal(f.rpc.at(-1).arguments.vision, 'yes');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: an engine that stops while loading keeps the code and the last engine line for the Details disclosure', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'server_exited', output: { log_tail: 'memory guard: 5.1 GiB free is under the 6 GiB limit' } });
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /^server_exited: memory guard: 5\.1 GiB free/);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: a start that died because Windows ran out of locked memory is tried once more with a smaller pin, said in plain words, and the saved settings stay as they were', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'server_exited', output: { log_tail: 'verify: mapped staging allocation failed [pinned-memory-exhausted] the OS gave no more page-locked memory; start again with STRATA_ARENA_PIN_GIB=26 (pin less of the expert arena)' } }, 1);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.calls.filter(c => c[0] === 'start').length, 2, 'one try and exactly one more');
    assert.match(job.result.summary, /ready, after a second try: Windows would not lock as much memory/);
    assert.match(job.details[0], /pinned-memory-exhausted/);
    assert.match(job.details[1], /STRATA_ARENA_PIN_GIB=26/);
    const instance = f.native.state().instances[0];
    assert.equal(instance.status, 'ready');
    const saved = JSON.parse(readFileSync(join(f.install, 'strata-iq2_xs.json'), 'utf8'));
    assert.equal(saved.env, undefined, 'the safer setting is for this start only; the model config is not changed');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: the second try is the last one: a second failure is reported with both, and nothing loops', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'server_exited', output: { log_tail: 'mapped staging allocation failed [pinned-memory-exhausted]' } });
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'error');
    assert.equal(f.calls.filter(c => c[0] === 'start').length, 2);
    assert.doesNotMatch(job.error, /first stop was/, 'the same stop twice is said once, not repeated');
    assert.match(job.details.join(' '), /Second try/, 'the second try stays in the details');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: retrySafer false keeps the single try', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'server_exited', output: { log_tail: 'mapped staging allocation failed [pinned-memory-exhausted]' } });
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs', retrySafer: false }).id);
    assert.equal(job.state, 'error');
    assert.equal(f.calls.filter(c => c[0] === 'start').length, 1);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: a Control whose wait ran out has not stopped the engine, so the load is kept and watched, not failed (#5177 #4293)', async () => {
  /* `runtime start --wait N` answers "startup_wait_expired" (exit 8) while the engine is still loading, and "runtime operation wait expired" when the owner
     was too busy to answer a status question in time; neither stops the engine. Control is asked: starting or ready means the app keeps the load. */
  for (const message of ['The engine did not start.', 'runtime operation wait expired; inspect runtime status']) {
    const f = packagedFixture();
    try {
      await f.native.refresh(true);
      f.setStatus({ ok: true, state: 'starting', phase: 'loading' });
      f.failStart({ message, output: { ok: false, state: 'starting', error: message.startsWith('runtime') ? undefined : 'startup_wait_expired', phase: 'loading', exit_code: 8 } }, 1);
      const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
      assert.equal(job.state, 'done', job.error);
      assert.equal(f.calls.filter(c => c[0] === 'start').length, 1, 'a wait that ran out is not a failure: no second try');
      assert.equal(f.calls.filter(c => c[0] === 'stop').length, 1, 'the one stop is startWith clearing the slot; the loading engine is never stopped by the app');
      assert.equal(f.native.state().instances[0].status, 'ready', 'ready once /health answers');
      assert.ok(f.calls.some(c => c[0] === 'status' && c[1] === 'app-1'), 'Control was asked before the load was kept');
    } finally { await f.native.close(); f.cleanup(); }
  }
});

test('packaged: an expired wait with an engine Control says is gone is a failure, and an adopted load whose engine dies ends as one', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    f.setStatus({ ok: false, state: 'failed', error: 'server_exited', engine_exit_code: 1 });
    f.failStart({ message: 'The engine did not start.', output: { ok: false, state: 'starting', error: 'startup_wait_expired', exit_code: 8 } }, 1);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs', retrySafer: false }).id);
    assert.equal(job.state, 'error', 'Control says the engine is gone: nothing to adopt');
  } finally { await f.native.close(); f.cleanup(); }
  const g = packagedFixture();
  try {
    await g.native.refresh(true);
    g.setHealthDown(true);                                           // never answers /health ...
    let asked = 0;                                                    // the first look (is the wait's engine still loading?) says starting, every later one says it ended
    g.setStatus(() => (++asked === 1 ? { ok: true, state: 'starting', phase: 'loading' } : { ok: false, state: 'failed', error: 'server_exited', engine_exit_code: 3 }));
    g.failStart({ message: 'The engine did not start.', output: { ok: false, state: 'starting', error: 'startup_wait_expired', exit_code: 8 } }, 1);
    const started = g.native.operation('load', { model: 'iq2_xs' });
    const job = await finish(g.native, started.id);
    assert.equal(job.state, 'error');
    assert.match(String(job.error ?? ''), /server_exited/);
  } finally { await g.native.close(); g.cleanup(); }
});

test('packaged: a ready instance whose engine stopped answering is marked stopped once Control says so, and stays ready while the engine is only busy (#4973)', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'done', job.error);
    f.setMetricsDown(true);
    f.setStatus({ ok: true, state: 'ready' });                       // busy, not gone
    await new Promise(r => setTimeout(r, 5200));
    assert.equal(f.native.state().instances[0].status, 'ready');
    assert.equal(f.native.state().instances[0].phase, 'unavailable');
    f.setStatus({ ok: true, state: 'stopped' });                     // the owner left with its engine
    await new Promise(r => setTimeout(r, 2600));
    const instance = f.native.state().instances[0];
    assert.equal(instance.status, 'error');
    assert.match(instance.error, /Engine stopped/);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: a run config that points into an old Control folder is repaired to this Control (a reinstall or an update moved it)', async () => {
  const f = packagedFixture();
  try {
    const here = join(f.root, 'control');
    mkdirSync(join(here, 'data'), { recursive: true });
    writeFileSync(join(here, 'data', 'expert-profile.bin'), 'profile');
    const path = join(f.install, 'strata-iq2_xs.json');
    const old = 'C:/Users/Alex/AppData/Local/Temp/strata-life-install/control';
    writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', cwd: old, args: ['--pack', 'D:/Strata/packs/x', '--expert-profile', old + '/data/expert-profile.bin', '--max-context', '32768'], model_name: 'x', port: 8080 }));
    await f.native.refresh(true);
    const repaired = JSON.parse(readFileSync(path, 'utf8'));
    assert.equal(repaired.cwd.toLowerCase(), here.toLowerCase());
    assert.equal(repaired.args[repaired.args.indexOf('--expert-profile') + 1].toLowerCase(), join(here, 'data', 'expert-profile.bin').toLowerCase());
    assert.equal(repaired.args[1], 'D:/Strata/packs/x', 'paths outside Control are left alone');
    assert.ok(existsSync(path + '.bak'), 'the old config is kept');
    // a config that is already right, or whose new target is not there, is not touched
    const before = readFileSync(path, 'utf8');
    await f.native.refresh(true);
    assert.equal(readFileSync(path, 'utf8'), before);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: deleting a downloaded model unloads it first, then the installer removes its files', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    const load = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(load.state, 'done', load.error);
    assert.equal(f.native.state().instances.length, 1);
    const job = await finish(f.native, f.native.operation('remove', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.native.state().instances.length, 0, 'the loaded model was unloaded first');
    assert.deepEqual(f.calls.filter(c => c[0] === 'stop').at(-1), ['stop', 'app-1']);
    const removed = f.rpc.filter(c => c.name === 'strata_remove');
    assert.equal(removed.length, 1);
    assert.deepEqual(removed[0].arguments, { model: 'iq2_xs', confirm: true });
    assert.equal(f.others().length, 0);
    const bad = await finish(f.native, f.native.operation('remove', { model: '../x' }).id);
    assert.equal(bad.state, 'error', 'an id that is not an installed model is refused');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: a local GGUF folder is checked by Control, then installed from where it is (no download)', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    const preview = await f.native.localPreview('D:/models/iq3s');
    assert.equal(preview.candidate_selection.model, 'IQ3_S');
    assert.deepEqual(f.calls.find(c => c[0] === 'importPlan'), ['importPlan', 'D:/models/iq3s']);
    await assert.rejects(() => f.native.localPreview('  '), /Choose a folder/);
    await f.native.plan({ family: 'qwen', model: 'IQ3_S', gguf_dir: 'D:/models/iq3s' });
    assert.equal(f.rpc.at(-1).arguments.gguf_dir, 'D:/models/iq3s');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: the install job also fetches the engine through Control (merge of the model progress and the engine step); a failed engine keeps the model and says so', async () => {
  const report = { engine: { status: 'install', offered: { version: '0.1.46' }, download_bytes: 5e8 } };
  const f = packagedFixture({ engineUsable: false, updater: { report, async update(options) {
    options.onEvent({ event: 'progress', phase: 'download', percent: 50, done: 2.5e8, total: 5e8 });
    return { ok: true, results: [{ component: 'engine', status: 'updated', installed_after: { version: '0.1.46' } }] };
  } } });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('install', { family: 'qwen', model: 'Q2_0' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(job.result.install.engine.state, 'done');
    assert.deepEqual(job.result.install.parts.map(p => p.kind), ['model', 'engine']);
    assert.deepEqual(f.calls.filter(c => c[0] === 'update'), [['update', true, false]], 'the engine only, from this install job');
    assert.equal(f.calls.filter(c => ['saveProfile', 'start'].includes(c[0])).length, 0);
  } finally { await f.native.close(); f.cleanup(); }
  const g = packagedFixture({ engineUsable: false, updater: { report, async update() { throw new Error('the download of http://x/e.tar.gz was interrupted after 100 of 200 bytes (ConnectionResetError)'); } } });
  try {
    await g.native.refresh(true);
    const job = await finish(g.native, g.native.operation('install', { family: 'qwen', model: 'Q2_0' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.match(job.summary, /Your model is ready\. The engine for this PC couldn't be downloaded/);
    assert.equal(job.result.install.engine.state, 'failed');
    assert.doesNotMatch(job.summary, /http|ConnectionReset|bytes/);
  } finally { await g.native.close(); g.cleanup(); }
});

test('packaged: a model install that fails does not cancel the engine download it started (DL99: a retry would fail the same way again)', async () => {
  const report = { engine: { status: 'install', offered: { version: '0.1.46' }, download_bytes: 5e8 } };
  const seen = { finished: false, aborted: null };
  const f = packagedFixture({ engineUsable: false,
    installPoll: { summary: 'failed', install: { running: false, exit_code: 1, error: 'this engine (99.0.0) does not run Qwen3.6-35B-A3B yet - update Strata' } },
    updater: { report, async update(options) {
      options.onEvent({ event: 'progress', phase: 'download', percent: 40, done: 2e8, total: 5e8 });
      await new Promise(r => setTimeout(r, 2500));          // longer than the install job's first poll (1.5 s), which reads the failure
      seen.aborted = options.signal?.aborted === true; seen.finished = true;
      return { ok: true, results: [{ component: 'engine', status: 'updated', installed_after: { version: '0.1.46' } }] };
    } } });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('install', { family: 'qwen', model: 'Q2_0' }).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /does not run Qwen3\.6-35B-A3B yet/);
    assert.equal(seen.finished, true, 'the engine download ran to its end before the job ended');
    assert.equal(seen.aborted, false, 'the engine download was not cancelled');
    assert.equal(job.result.install.engine.state, 'done');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: with no engine on the PC yet, the engine is installed before setup starts (setup takes the picture reader from it; Qwen3.6 came up text-only)', async () => {
  const report = { engine: { status: 'install', offered: { version: '0.1.60' }, download_bytes: 7e8 } };
  const seen = { installsWhenEngineDone: null };
  let f;
  f = packagedFixture({ engineUsable: false, updater: { report, async update(options) {
    options.onEvent({ event: 'progress', phase: 'download', percent: 50, done: 3.5e8, total: 7e8 });
    await new Promise(r => setTimeout(r, 300));
    seen.installsWhenEngineDone = f.rpc.filter(c => c.name === 'strata_install' && c.arguments.confirm).length;
    return { ok: true, results: [{ component: 'engine', status: 'updated', installed_after: { version: '0.1.60' } }] };
  } } });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('install', { family: 'qwen', model: 'Q2_0' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(seen.installsWhenEngineDone, 0, 'setup was not started before the engine was in');
    assert.equal(f.rpc.filter(c => c.name === 'strata_install' && c.arguments.confirm).length, 1);
    assert.deepEqual(f.calls.filter(c => c[0] === 'update'), [['update', true, false]], 'the engine is fetched once');
    assert.equal(job.result.install.engine.state, 'done');
  } finally { await f.native.close(); f.cleanup(); }
  // an engine that is there already: setup starts at once and the engine step runs beside it, as before
  const g = packagedFixture({ engineUsable: true, updater: { report: { engine: { status: 'current', installed: { version: '0.1.60' } } }, async update() { throw new Error('not called'); } } });
  try {
    await g.native.refresh(true);
    const job = await finish(g.native, g.native.operation('install', { family: 'qwen', model: 'Q2_0' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(g.calls.filter(c => c[0] === 'update').length, 0, 'a current engine is not fetched');
  } finally { await g.native.close(); g.cleanup(); }
});

test('packaged: Stop still cancels the engine download of an install', async () => {
  const report = { engine: { status: 'install', offered: { version: '0.1.46' }, download_bytes: 5e8 } };
  const seen = { aborted: null };
  const f = packagedFixture({ engineUsable: false,
    installPoll: { summary: 'running', install: { running: true, step: '5/7: downloading' } },
    updater: { report, async update(options) {
      await new Promise(resolve => { if (options.signal?.aborted) resolve(); else options.signal?.addEventListener('abort', resolve, { once: true }); });
      seen.aborted = true;
      throw new Error('stopped');
    } } });
  try {
    await f.native.refresh(true);
    const id = f.native.operation('install', { family: 'qwen', model: 'Q2_0' }).id;
    await new Promise(r => setTimeout(r, 300));
    await f.native.cancel(id);
    const job = await finish(f.native, id);
    assert.equal(job.state, 'cancelled');
    for (let i = 0; i < 100 && seen.aborted === null; i++) await new Promise(r => setTimeout(r, 10));
    assert.equal(seen.aborted, true);
    await new Promise(r => setTimeout(r, 2000));             // the install loop's 1.5 s wait ends and it returns before the folder goes
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: the app default of several requests at once and the memory retry share ONE second try, never two', async () => {
  const f = packagedFixture(); f.hardware.gpus[0].vram_gb = 48;
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'server_exited', output: { log_tail: 'something else went wrong' } }, 1);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.calls.filter(c => c[0] === 'start').length, 2, 'the failed first start and one second try');
    assert.match(job.result.summary, /after a second try/);
    assert.equal(f.native.state().instances[0].parallel, 1, 'the second try runs one request at a time');
  } finally { await f.native.close(); f.cleanup(); }
  const g = packagedFixture(); g.hardware.gpus[0].vram_gb = 48;
  try {
    await g.native.refresh(true);
    g.failStart({ message: 'server_exited', output: { log_tail: 'mapped staging allocation failed [pinned-memory-exhausted]' } });
    const job = await finish(g.native, g.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'error');
    assert.equal(g.calls.filter(c => c[0] === 'start').length, 2, 'memory change and parallel drop ride on the same single try');
  } finally { await g.native.close(); g.cleanup(); }
});

test('packaged: a tune measures the engine a load starts: same start args, slots included, for the same model and profile', async () => {
  for (const vram of [11.9, 48]) {
    const f = packagedFixture(); f.hardware.gpus[0].vram_gb = vram;
    try {
      await f.native.refresh(true);
      const args = { model: 'iq2_xs', resourceProfile: 'work', vision: 'no' };
      await f.native.tuneStatus(args).catch(() => {});      // the fake host cannot run Control's tuner; the config it would measure is written first
      const tuned = JSON.parse(readFileSync(join(f.root, 'data', 'native-tune', 'iq2_xs.json'), 'utf8'));
      const job = await finish(f.native, f.native.operation('load', args).id);
      assert.equal(job.state, 'done', job.error);
      const loaded = { ...f.saved.at(-1) }; delete loaded.port; delete loaded.api_key;
      assert.deepEqual(tuned.args, loaded.args, 'the engine arguments are identical');
      for (const [key, value] of Object.entries(tuned)) assert.deepEqual(loaded[key], value, `${key} is the same in the tune and the load`);   // a load adds only its own identity (aliases, model name, key, port)
      if (vram > 40) assert.ok(tuned.parallel > 1, 'a big card gets the app default of several slots in the tune too');
    } finally { await f.native.close(); f.cleanup(); }
  }
});

test('packaged: a profile an elevated run locked (PermissionError) is repaired once and the start is tried again; without repair it is a plain message, not a memory retry', async () => {
  const f = packagedFixture({ repair: true });
  try {
    await f.native.refresh(true);
    f.failStart({ message: 'cannot read profile (PermissionError)' }, 1);
    const job = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.calls.filter(c => c[0] === 'repairAccess').length, 1);
    assert.equal(f.calls.filter(c => c[0] === 'start').length, 2);
  } finally { await f.native.close(); f.cleanup(); }
  const g = packagedFixture({ repair: true });
  try {
    await g.native.refresh(true);
    g.failStart({ message: 'cannot read profile (PermissionError)' });
    const job = await finish(g.native, g.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(job.state, 'error');
    assert.equal(g.calls.filter(c => c[0] === 'start').length, 2, 'one repair try, no memory retry on top');
    assert.match(job.error, /PermissionError/);
  } finally { await g.native.close(); g.cleanup(); }
});

test('a GGUF folder Control refuses (exit 5, ok:false) still comes back as a checked plan with its reasons, so the page can say it in words', async () => {
  const plan = { file_set_status: 'invalid_or_incomplete', candidate_selection: null, issues: ['x.gguf: GGUF tensor or metadata count exceeds the inspection limit', 'unrecognized_or_unsupported_model_variant'] };
  const control = createControl({ control: { dir: '/c', python: '/p', cli: '/c/strata.py', bundle: { version: '9' } }, run: async () => ({ code: 5, json: { ok: false, plan }, text: '', stderr: '' }) });
  assert.deepEqual(await control.importPlan('/some/folder'), plan);
  const broken = createControl({ control: { dir: '/c', python: '/p', cli: '/c/strata.py', bundle: { version: '9' } }, run: async () => ({ code: 1, json: null, text: '', stderr: 'boom' }) });
  await assert.rejects(() => broken.importPlan('/some/folder'), /boom/);
});

test("preparing an adopted GGUF folder again never takes the folder's parent for the data folder (it is the person's, not Strata's)", async () => {
  const f = packagedFixture();
  try {
    // The model is already adopted: its run config points at a GGUF in someone's own ".../models/IQ3_S" folder.
    const own = join(f.root, 'own');
    const configPath = join(f.install, 'strata-iq3_s.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8'));
    writeFileSync(configPath, JSON.stringify({ ...config, args: [...config.args, '--native', join(own, 'models', 'IQ3_S', 'a-00001-of-00002.gguf')] }));
    f.catalog.families[0].sizes.push({ id: 'iq3_s', model: 'IQ3_S', download_gb: 83, ram_needed_gb: 62, about: 'x', on_this_pc: 'fits', installed: true, images: true, experimental: false });
    await f.native.refresh(true);
    await f.native.plan({ family: 'qwen', model: 'IQ3_S', gguf_dir: join(own, 'models', 'IQ3_S') });
    const sent = f.rpc.at(-1).arguments;
    assert.notEqual(sent.data_dir, own, "the person's own folder is not used for the packs");
    assert.equal(sent.gguf_dir, join(own, 'models', 'IQ3_S'));
  } finally { await f.native.close(); f.cleanup(); }
});

test('the app never adopts an older command-line install: Control is always told the app\'s own folder, and has its own settings file', async () => {
  const f = packagedFixture();
  try {
    await f.native.refresh(true);
    const asked = f.calls.filter(c => c[0] === 'installedModels').map(c => c[1]);
    assert.ok(asked.length > 0 && asked.every(dir => dir === f.install || (dir && dir.toLowerCase() === f.install.toLowerCase())), `Control was asked about ${JSON.stringify(asked)}`);
  } finally { f.cleanup(); }
  const env = controlEnvironment({ dir: '/c' }, { env: { APPDATA: '/old/appdata' }, components: join(tmpdir(), 'Strata', 'components') });
  assert.equal(env.STRATA_SETTINGS_FILE, join(tmpdir(), 'Strata', 'app-settings.json'));
});

test("onboarding 'use your own GGUF' (#3507, #3424): the models folder the person picked holds the GGUF files, so preparing them never takes it for the data folder", async () => {
  const f = packagedFixture();
  try {
    const own = join(f.root, 'Strata-data', 'models', 'unsloth-UD-Q4_K_XL');
    mkdirSync(own, { recursive: true });
    await f.native.refresh(true);
    f.native.setDownloadDirectory(own);                        // what the first-run folder chooser wrote
    await f.native.plan({ family: 'qwen', model: 'IQ3_S', gguf_dir: own });
    const sent = f.rpc.at(-1).arguments;
    assert.equal(sent.gguf_dir, own);
    assert.notEqual(sent.data_dir, own, 'the folder of GGUF files is not the data folder: the packs would be downloaded into it');
    // a folder of GGUF files named like a data folder's models/<tag> hands its Strata folder to the installer
    assert.ok(sent.data_dir === undefined || sent.data_dir === join(f.root, 'Strata-data'), `data_dir was ${sent.data_dir}`);
  } finally { await f.native.close(); f.cleanup(); }
});

/* The start wait follows the model's size, as `strata launch` does since #3949, and never asks an older Control for more than it takes.
   Reported on the app side by #3459 and #4191: a big model on a slow disk was stopped at 300 s while the engine went on reading. */
test('packaged: a big model is waited for by its size, inside the ceiling of the Control this app has (#3459 #4191)', async () => {
  const run = async (cliStartWait, downloadGb) => {
    const f = packagedFixture({ ids: ['big'], cliStartWait, extraSizes: [{ id: 'big', model: 'BIG', download_gb: downloadGb, ram_needed_gb: 60, about: 'x', on_this_pc: 'fits', installed: false, images: false, experimental: false }] });
    try {
      await f.native.refresh(true);
      const job = await finish(f.native, f.native.operation('load', { model: 'big' }).id);
      assert.equal(job.state, 'done', job.error);
      const start = f.calls.find(c => c[0] === 'start');
      const saved = f.saved[0];
      await f.native.close();
      return { wait: start[2], ready: saved.engine_ready_s };
    } finally { f.cleanup(); }
  };
  assert.equal((await run('old', 180)).wait, 600, 'a Control with no long-wait ceiling at all is asked for 600 s, as today');
  assert.equal((await run('split', 180)).wait, 600, 'a Control whose runtime takes 1800 s but whose CLI still stops at 600 is asked for 600 s: the CLI would refuse the start outright (#3949)');
  assert.equal((await run('cli-only', 180)).wait, 600, 'a Control whose CLI takes it but whose operation record still stops at 600 is asked for 600 s too: the record is written before the start runs');
  const long = await run('runtime', 180);
  assert.equal(long.wait, 1720, '600 s + 8 s per GB above 40 GB, as in strata launch, once the CLI really takes it');
  assert.equal(long.ready, 1780, 'the server readies after the start gave up, never before it');
  const small = await run('runtime', 20);
  assert.equal(small.wait, 300, 'a model under 40 GB keeps the 300 s it always had');
  assert.equal(small.ready, undefined, 'nothing is added to a small model’s run config');
  const huge = await run('runtime', 400);
  assert.equal(huge.wait, 1800, 'and never more than this Control accepts');
});

test('packaged: a run config that names a longer engine wait is honoured, and never shortened by the app', async () => {
  const f = packagedFixture({ ids: ['qwen3.6-35b-a3b-ud-q4_k_xl'], cliStartWait: 'runtime' });
  try {
    writeFileSync(join(f.install, 'strata-qwen3.6-35b-a3b-ud-q4_k_xl.json'), JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(f.install, 'tok'), args: ['--max-context', '32768'], model_name: 'Qwen qwen3.6-35b-a3b-ud-q4_k_xl', port: 8080, engine_ready_s: 3600 }));
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('load', { model: 'qwen3.6-35b-a3b-ud-q4_k_xl' }).id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.calls.find(c => c[0] === 'start')[2], 1800, 'the wait of the newest Control, which is what the ceiling allows');
    assert.equal(f.saved[0].engine_ready_s, 3600, 'setup’s own longer wait stays as it is');
  } finally { await f.native.close(); f.cleanup(); }
});

// ---------------------------------------------------------------------------------- 0.2.17: quit while "Prepare and load" runs (the model packer kept writing to the disk)

/** An installer whose job the tests can watch: `state.running` is whether setup and the packer run. `cancelIgnored` cancels leave it running (a stop that did not take). */
function watchedInstaller({ startMs = 250, cancelIgnored = 0 } = {}) {
  const state = { running: false, cancels: [], ignored: cancelIgnored };
  const rpcHook = (params, _usual, deliver) => {
    const a = params.arguments ?? {};
    if (params.name !== 'strata_install') return false;
    if (a.confirm) {                                   // the installer is busy planning, then launches the job and answers
      setTimeout(() => { state.running = true; deliver({ structuredContent: { summary: 'Install started', plan: { download_gb: 75.8 }, install: { running: true, step: 'packing' } } }); }, startMs);
      return true;
    }
    if (a.cancel) {
      state.cancels.push(state.running);
      const was = state.running;
      if (was && state.ignored > 0) state.ignored--; else state.running = false;
      deliver({ structuredContent: { summary: was ? 'the install was stopped' : 'no install is running' } });
      return true;
    }
    deliver({ structuredContent: { summary: 'x', install: { running: state.running, step: 'packing' } } });
    return true;
  };
  return { state, rpcHook };
}
const startSeen = async f => { for (let i = 0; i < 300; i++) { if (f.rpc.some(c => c.name === 'strata_install' && c.arguments.confirm)) return; await new Promise(r => setTimeout(r, 10)); } throw new Error('the install start was not sent'); };

test('quit while the install is still starting: the cancel waits for the start, then stops the job it launched (it ran on after the app had gone)', async () => {
  const w = watchedInstaller();
  const f = packagedFixture({ rpcHook: w.rpcHook });
  try {
    await f.native.refresh(true);
    const id = f.native.operation('install', { family: 'qwen', model: 'Q2_0' }).id;
    await startSeen(f);
    assert.equal(w.state.running, false, 'the installer has not launched the job yet');
    await f.native.close();
    await new Promise(r => setTimeout(r, 400));          // longer than the installer needs to launch: nothing may be left running
    assert.equal(w.state.running, false, 'the model preparation still runs after the app was closed');
    assert.ok(w.state.cancels.at(-1) === true || w.state.cancels.includes(true), 'a cancel reached the job after it was launched');
    assert.notEqual(f.native.state().jobs.find(j => j.id === id)?.state, 'running');
  } finally { f.cleanup(); }
});

test('quit while the engine or a drafter is being downloaded for the job: the download is told to stop (the job\'s engineAbort), and what Control still runs is ended', async () => {
  let seenSignal = null;
  const report = { engine: { status: 'install', offered: { version: '0.1.60' }, download_bytes: 7e8 } };
  const w = watchedInstaller({ startMs: 5 });
  const f = packagedFixture({ engineUsable: false, rpcHook: w.rpcHook, updater: { report, async update(options) {
    seenSignal = options.signal;
    await new Promise(resolve => options.signal.addEventListener('abort', resolve, { once: true }));
    throw new Error('stopped');
  } } });
  try {
    await f.native.refresh(true);
    f.native.operation('install', { family: 'qwen', model: 'Q2_0' });
    for (let i = 0; i < 300 && !seenSignal; i++) await new Promise(r => setTimeout(r, 10));
    assert.ok(seenSignal, 'the engine download started');
    assert.equal(seenSignal.aborted, false);
    await f.native.close();
    assert.equal(seenSignal.aborted, true, 'the quit stopped the download (before, only a Stop press did)');
    assert.ok(f.calls.some(c => c[0] === 'killChildren'), 'Control programs that are still running are ended last');
  } finally { f.cleanup(); }
});

test('quit: a cancel that did not take is repeated until the installer says no install runs', async () => {
  const w = watchedInstaller({ startMs: 20, cancelIgnored: 1 });
  const f = packagedFixture({ rpcHook: w.rpcHook });
  try {
    await f.native.refresh(true);
    f.native.operation('install', { family: 'qwen', model: 'Q2_0' });
    await startSeen(f);
    await new Promise(r => setTimeout(r, 100));          // the job runs
    assert.equal(w.state.running, true);
    await f.native.close();
    assert.equal(w.state.running, false, 'the second cancel stopped it');
    assert.equal(w.state.cancels.length, 2, 'one cancel that did not take, one that did');
  } finally { f.cleanup(); }
});
