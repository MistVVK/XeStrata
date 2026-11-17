import './helpers/isolate-profile.mjs';   // a test run never writes the owner's real Strata profile
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync, existsSync } from 'node:fs';
import { join, basename } from 'node:path';
import { tmpdir } from 'node:os';
import { bundledPython, controlEnvironment, createControl, findControl, runCli, updatedControlDirectory } from '../server/control-runtime.mjs';
import { checkComponentUpdates, engineFirstRun } from '../server/updates.mjs';
import { createAppUpdater } from '../server/app-updater.mjs';
import { createNativeControl } from '../server/native-control.mjs';

const isWin = process.platform === 'win32';

function controlDir(parent, name, version) {
  const dir = join(parent, name);
  mkdirSync(join(dir, 'python', ...(isWin ? [] : ['bin'])), { recursive: true });
  mkdirSync(join(dir, 'strata_control'), { recursive: true });
  writeFileSync(bundledPython(dir), 'x');
  writeFileSync(join(dir, 'strata.py'), '# cli');
  if (version) writeFileSync(join(dir, 'BUNDLE.json'), JSON.stringify({ version }));
  return dir;
}

test('only the update step may acquire: Control keeps STRATA_NO_ACQUIRE unless `acquire` is asked for', () => {
  const control = { dir: '/c', python: '/c/python/python3', cli: '/c/strata.py' };
  assert.equal(controlEnvironment(control, { env: {}, components: '/k' }).STRATA_NO_ACQUIRE, '1');
  const env = controlEnvironment(control, { env: { STRATA_NO_ACQUIRE: '1' }, components: '/k', acquire: true });
  assert.equal('STRATA_NO_ACQUIRE' in env, false);
  assert.equal(env.STRATA_COMPONENTS_DIR, '/k');
  assert.equal(env.PYTHONNOUSERSITE, '1');
});

test('an updated Control wins only when it is newer than the bundled one, and only when whole', () => {
  const root = mkdtempSync(join(tmpdir(), 'strata-update-control-'));
  try {
    const install = join(root, 'install');
    mkdirSync(join(install, 'host'), { recursive: true });
    const bundled = controlDir(install, 'control', '0.1.45');
    const components = join(root, 'components');
    const updated = controlDir(join(components, 'control'), '0.1.46', '0.1.46');
    writeFileSync(join(components, 'control', 'active.json'), JSON.stringify({ version: '0.1.46', folder: '0.1.46' }));
    const find = (extra = {}) => findControl({ env: {}, hostDir: join(install, 'host'), components, ...extra });
    assert.equal(find().dir, updated, 'the updated 0.1.46 beats the bundled 0.1.45');
    writeFileSync(join(bundled, 'BUNDLE.json'), JSON.stringify({ version: '0.1.47' }));
    assert.equal(find().dir, bundled, 'an app update that ships a newer Control is not held back by an older updated one');
    writeFileSync(join(bundled, 'BUNDLE.json'), JSON.stringify({ version: '0.1.46' }));
    assert.equal(find().dir, bundled, 'a tie keeps the bundled one');
    rmSync(join(updated, 'strata.py'));
    writeFileSync(join(bundled, 'BUNDLE.json'), JSON.stringify({ version: '0.1.45' }));
    assert.equal(find().dir, bundled, 'a half-installed updated Control is not used');
    writeFileSync(join(components, 'control', 'active.json'), JSON.stringify({ version: '0.1.46', folder: '../../etc' }));
    assert.equal(updatedControlDirectory({ env: {}, components }), null, 'a pointer cannot leave the control folder');
  } finally { rmSync(root, { recursive: true, force: true }); }
});

function fakeSpawn(lines, code = 0, seen = []) {
  return (command, args, options) => {
    seen.push({ command, args, options });
    const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter(); child.kill = () => { child.killed = true; };
    queueMicrotask(() => {
      // split a line in two chunks: the reader must wait for the newline
      const text = lines.map(l => JSON.stringify(l)).join('\n') + '\n';
      child.stdout.emit('data', text.slice(0, 20)); child.stdout.emit('data', text.slice(20));
      child.emit('close', code);
    });
    return child;
  };
}

test('runCli streams JSON lines as events and returns the last record; `acquire` reaches the environment', async () => {
  const seen = [];
  const events = [];
  const control = { dir: '/c', python: '/c/python/python3', cli: '/c/strata.py' };
  const spawnProcess = fakeSpawn([{ event: 'phase', phase: 'feed', message: 'Reading' }, { event: 'progress', percent: 50, label: 'engine 0.1.46' }, { ok: true, results: [] }], 0, seen);
  const result = await runCli(control, ['update', '--output', 'jsonl'], { spawnProcess, env: { STRATA_NO_ACQUIRE: '1' }, acquire: true, onLine: e => events.push(e) });
  assert.equal(events.length, 3);
  assert.equal(events[1].percent, 50);
  assert.deepEqual(result.json, { ok: true, results: [] });
  assert.equal('STRATA_NO_ACQUIRE' in seen[0].options.env, false);
  await runCli(control, ['models', 'list'], { spawnProcess: fakeSpawn([{ ok: true }], 0, seen), env: {} });
  assert.equal(seen[1].options.env.STRATA_NO_ACQUIRE, '1', 'everything else keeps the switch on');
});

test('createControl: the update calls ask the explicit commands; a failed step carries its message and exit code', async () => {
  const seen = [];
  const reply = { ok: true, results: [{ component: 'engine', status: 'updated' }] };
  const control = createControl({
    control: { dir: '/c', python: '/p', cli: '/c/strata.py' }, env: { STRATA_UPDATE_FEED: 'http://127.0.0.1:1/strata-feed.json', STRATA_APP_VERSION: '0.1.0' },
    run: async (args, options) => { seen.push([args.join(' '), !!options.acquire]); return { code: 0, json: args.includes('--check') ? { ok: true, engine: {} } : args[0] === 'components' ? { ok: true, to: '0.1.45' } : reply, text: '', stderr: '' }; },
  });
  await control.updateCheck();
  await control.update({ engine: true, control: false });
  await control.rollbackEngine();
  assert.deepEqual(seen, [
    ['update --check --output json --feed http://127.0.0.1:1/strata-feed.json --app-version 0.1.0', true],
    ['update --output jsonl --feed http://127.0.0.1:1/strata-feed.json --no-control --app-version 0.1.0', true],
    ['components rollback --output json', false],
  ], 'only the check and the update acquire; the rollback downloads nothing');
  const failing = createControl({
    control: { dir: '/c', python: '/p', cli: '/c/strata.py' }, env: {},
    run: async () => ({ code: 7, json: { ok: false, results: [{ component: 'engine', status: 'failed', error: 'app-1 is serving.', exit_code: 7 }] }, text: '', stderr: '' }),
  });
  await assert.rejects(failing.update({}), err => err.code === 'engine_serving' && /serving/.test(err.message) && err.exitCode === 7);
});

test('the Settings check says so, in plain words, when the own update check of the app fails, and keeps the raw text for Details', async () => {
  const feed = (control = { installed: '0.2.5', offered: '0.2.5', status: 'current' }) => ({ version: '0.2.5', async updateCheck() { return { ok: true, feed: { sequence: 3 }, engine: { installed: { version: '0.1.50' }, offered: { version: '0.1.50' }, status: 'current' }, control, app: { status: 'not-in-feed' } }; } });
  const failing = { check: async () => { throw new Error('error decoding response body for url (http://127.0.0.1:19090/latest.json)'); } };
  const nothing = await checkComponentUpdates({ control: feed(), appUpdater: failing, appVersion: '0.2.5' });
  assert.equal(nothing.strata.status, 'error', 'it no longer says "up to date" when the app could not be checked');
  assert.match(nothing.strata.message, /couldn't check for a new version of the app itself/);
  assert.doesNotMatch(nothing.strata.message, /decoding|http|latest\.json/, 'no raw text in the main line');
  assert.match(nothing.strata.details, /error decoding response body/);
  const withControl = await checkComponentUpdates({ control: feed({ installed: '0.2.5', offered: '0.2.6', status: 'available' }), appUpdater: failing, appVersion: '0.2.5' });
  assert.equal(withControl.strata.status, 'available');
  assert.match(withControl.strata.message, /new version of Strata is available.*couldn't check for a new version of the app itself/s);
  assert.match(withControl.strata.details, /decoding/);
  const fine = await checkComponentUpdates({ control: feed(), appUpdater: { check: async () => ({ available: false }) }, appVersion: '0.2.5' });
  assert.equal(fine.strata.status, 'current');
  assert.equal(fine.strata.details, undefined);
});

test('the Settings update check maps the feed to the existing result shape', async () => {
  const report = overrides => ({ ok: true, feed: { sequence: 3 }, engine: { installed: { version: '0.1.45' }, offered: { version: '0.1.46' }, status: 'available', download_bytes: 505_000_000 }, control: { installed: '0.1.45', offered: '0.1.45', status: 'current' }, app: { status: 'not-in-feed' }, ...overrides });
  const make = value => ({ version: '0.1.45', async updateCheck() { if (value instanceof Error) throw value; return value; } });
  let result = await checkComponentUpdates({ control: make(report()), appVersion: '0.1.0' });
  assert.equal(result.engine.status, 'available');
  assert.equal(result.engine.latestVersion, '0.1.46');
  assert.match(result.engine.message, /new engine is available \(505 MB\)/);
  assert.doesNotMatch(result.engine.message, /0\.1\.4|sha|exit/i);
  assert.equal(result.engine.canInstall, true);
  assert.equal(result.strata.status, 'current');
  result = await checkComponentUpdates({ control: make(report({ engine: { installed: null, offered: { version: '0.1.46' }, status: 'install', download_bytes: 505_000_000 } })) });
  assert.equal(result.engine.status, 'available');
  assert.match(result.engine.message, /needs an engine to run models\. Press Update engine/);
  result = await checkComponentUpdates({ control: make(report({ engine: { installed: { version: '0.1.46' }, offered: { version: '0.1.46' }, status: 'current' } })) });
  assert.equal(result.engine.status, 'current');
  result = await checkComponentUpdates({ control: make(report({ engine: { installed: null, offered: null, status: 'none-for-this-pc', reasons: ['0.1.46 is for linux-x86_64, this PC is windows-x86_64'] } })) });
  assert.equal(result.engine.status, 'unavailable');
  assert.doesNotMatch(result.engine.message, /windows-x86_64/);
  assert.match(result.engine.details, /windows-x86_64/, 'the technical reason is kept for the Details disclosure');
  result = await checkComponentUpdates({ control: make(report({ engine: { installed: { version: '0.1.46', backend: 'hip' }, offered: null, status: 'wrong-make', note: 'The engine installed here is for AMD cards, but this update would give your PC NVIDIA one. Nothing was changed.', reasons: ['x'] } })) });
  assert.equal(result.engine.status, 'unavailable');
  assert.equal(result.engine.canInstall, false);
  assert.match(result.engine.message, /Nothing was changed/, 'the app says why in plain words, not that no engine exists for the card');
  assert.doesNotMatch(result.engine.message, /isn't an engine for this PC/);
  result = await checkComponentUpdates({ control: make(new Error('this build trusts no update key')) });
  assert.equal(result.strata.status, 'error');
  assert.equal(result.engine.status, 'error');
  assert.doesNotMatch(result.engine.message, /update key/);
  assert.match(result.engine.details, /no update key/);
  assert.match(result.engine.message, /aren't available in this version/);
  // the app's verdict comes from the shell's updater when the bridge exists
  const appUpdater = { check: async () => ({ available: true, version: '0.2.0', notes: 'Faster starts.' }) };
  result = await checkComponentUpdates({ control: make(report()), appUpdater, appVersion: '0.1.0' });
  assert.equal(result.strata.status, 'available');
  assert.equal(result.strata.latestVersion, '0.2.0');
  assert.match(result.strata.message, /new version of Strata is available\. Faster starts/);
});

test('the shell bridge client exists only inside the desktop app, speaks to loopback with the token, and polls to the end', async () => {
  assert.equal(createAppUpdater({ env: {} }), null);
  assert.equal(createAppUpdater({ env: { STRATA_SHELL_BRIDGE: 'http://example.com:80', STRATA_SHELL_TOKEN: 'x' } }), null, 'not a loopback address');
  const calls = [];
  const states = [{ state: 'downloading', downloaded: 10, total: 100 }, { state: 'downloading', downloaded: 100, total: 100 }, { state: 'done' }];
  const fetcher = async (url, init) => {
    calls.push([init.method, url.replace(/:\d+/, ':P'), init.headers.authorization]);
    if (url.endsWith('/app-update/status')) return Response.json(states.shift());
    return Response.json(url.endsWith('/check') ? { available: true, version: '0.2.0' } : { state: 'downloading' });
  };
  const updater = createAppUpdater({ env: { STRATA_SHELL_BRIDGE: 'http://127.0.0.1:4000', STRATA_SHELL_TOKEN: 'tok' }, fetcher, sleep: async () => {} });
  assert.equal((await updater.check()).version, '0.2.0');
  const progress = [];
  await updater.install({ onProgress: p => progress.push(p.downloaded ?? null) });
  assert.deepEqual(progress, [10, 100, null]);
  assert.ok(calls.every(c => c[2] === 'Bearer tok'));
  assert.deepEqual(calls.map(c => c[0] + ' ' + c[1].split('/').pop()), ['GET check', 'POST install', 'GET status', 'GET status', 'GET status']);
  const failing = createAppUpdater({ env: { STRATA_SHELL_BRIDGE: 'http://127.0.0.1:4000', STRATA_SHELL_TOKEN: 'tok' }, sleep: async () => {},
    fetcher: async url => Response.json(url.endsWith('/status') ? { state: 'error', error: 'signature mismatch' } : {}) });
  await assert.rejects(failing.install(), /signature mismatch/);
});

// ---------------------------------------------------------------------------------- the Settings buttons in the packaged app

function fixture({ withApp = false, update, appInstall } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-update-ops-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;      // the app's own install folder: the host always names it to Control
  const config = join(install, 'strata-iq2_xs.json');
  writeFileSync(config, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: ['--max-context', '32768'], model_name: 'Qwen', port: 8080 }));
  const calls = [];
  let previous = null;
  const control = {
    control: { dir: join(root, 'control') }, version: '0.1.45',
    async installedModels() { return { install_dir: install, models: [{ id: 'iq2_xs', name: 'Qwen', source_config: config, component_status: 'present', context_tokens: 32768, backend: 'cuda' }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: 'E:\\c', previous, engine: { version: '0.1.45', platform: 'windows-x86_64', backend: 'cuda', exe: 'e', root: 'r' } }; },
    async saveProfile() { return {}; }, async start() { return { ok: true, state: 'ready' }; }, async stop() { return { ok: true }; },
    async update(options) {
      calls.push(['update', options.engine, options.control]);
      options.onEvent?.({ event: 'phase', phase: 'download', message: 'Downloading engine 0.1.46 (505 MiB)' });
      options.onEvent?.({ event: 'progress', phase: 'download', label: 'engine 0.1.46', percent: 42, done: 1, total: 2 });
      await update?.(options);
      previous = '0.1.45-windows-x86_64-cuda';
      return { ok: true, results: [{ component: 'engine', status: 'updated', installed_after: { version: '0.1.46' } }] };
    },
    async installEngineFile(path, options) { calls.push(['install-file', path]); options.onEvent?.({ event: 'phase', phase: 'verify', message: 'Checking' }); return { ok: true, results: [{ component: 'engine', status: 'updated', installed_after: { version: '0.1.46' } }] }; },
    async rollbackEngine() { calls.push(['rollback']); previous = null; return { ok: true, to: '0.1.45' }; },
  };
  const spawned = [];
  const appUpdater = withApp ? {
    check: async () => ({ available: true, version: '0.2.0' }),
    install: appInstall ?? (async ({ onProgress }) => { calls.push(['app-install']); onProgress({ downloaded: 50, total: 100 }); return { state: 'done' }; }),
  } : null;
  const native = createNativeControl({ control, appUpdater, directory: data, spawnProcess: (...args) => { if (!args[1]?.some(a => String(a).endsWith('strata_mcp.py'))) spawned.push(args); throw new Error('no process but the installer tool (Control strata_mcp.py, absent in this fixture) may be started by the host in packaged mode'); }, fetcher: async () => Response.json({}), requests: () => [] });
  return { native, calls, spawned, data, control, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 400; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };

test('packaged: "Update engine" runs the explicit component update, shows its progress, and then offers the rollback', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    assert.equal(f.native.state().capabilities.componentUpdates, true);
    assert.equal(f.native.state().capabilities.rollback, false, 'nothing to roll back before the first update');
    const started = f.native.operation('update-engine', {});
    await new Promise(r => setTimeout(r, 5));
    const job = await finish(f.native, started.id);
    assert.equal(job.state, 'done', job.error);
    assert.match(job.summary, /engine is updated and ready/);
    assert.deepEqual(f.calls[0], ['update', true, false], 'the engine only: Control and the app are not touched by "Update engine"');
    assert.equal(f.spawned.length, 0);
    await f.native.refresh(true);
    assert.equal(f.native.state().capabilities.rollback, true, 'the previous engine the update kept is offered for rollback');
    const back = await finish(f.native, f.native.operation('rollback', {}).id);
    assert.equal(back.state, 'done', back.error);
    assert.match(back.summary, /previous engine is back/);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: progress events become the existing job summary and download bar', async () => {
  const seen = [];
  const f = fixture({ update: async () => { await new Promise(r => setTimeout(r, 30)); } });
  try {
    await f.native.refresh(true);
    const started = f.native.operation('update-engine', {});
    for (let i = 0; i < 100; i++) {
      const job = f.native.state().jobs.find(j => j.id === started.id);
      if (job.result?.install?.download) { seen.push(job.result.install.download, job.summary); break; }
      await new Promise(r => setTimeout(r, 5));
    }
    await finish(f.native, started.id);
    assert.deepEqual(seen[0], { file: 'engine 0.1.46', percent: 42 });
    assert.equal(seen[1], 'Downloading: 42%');
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: no update while a model is loaded; calibration is still refused', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    const load = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(load.state, 'done', load.error);
    const job = await finish(f.native, f.native.operation('update-engine', {}).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /Unload native models/);
    assert.equal(f.calls.filter(c => c[0] === 'update').length, 0);
    const calibrate = await finish(f.native, f.native.operation('calibrate', { model: 'iq2_xs' }).id);
    assert.match(calibrate.error, /Terminal|calibrates NVIDIA/);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: "Update Strata" updates Control through the feed, then the app through the shell', async () => {
  const f = fixture({ withApp: true });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('update-strata', {}).id);
    assert.equal(job.state, 'done', job.error);
    assert.deepEqual(f.calls.map(c => c[0]), ['update', 'app-install']);
    assert.deepEqual(f.calls[0], ['update', false, true]);
    assert.match(job.summary, /app update is installed\. Strata will restart/);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: an app update that ends the app is recorded as finished by the new version, not as an interrupted operation', async () => {
  const before = process.env.STRATA_APP_VERSION;
  process.env.STRATA_APP_VERSION = '0.2.2';
  const f = fixture({ withApp: true, appInstall: () => new Promise(() => {}) });      // the installer ends the app: install() never returns
  try {
    await f.native.refresh(true);
    const started = f.native.operation('update-strata', {});
    await new Promise(r => setTimeout(r, 60));
    assert.equal(f.native.state().jobs.find(j => j.id === started.id).state, 'running');
    const saved = JSON.parse(readFileSync(join(f.data, 'native-jobs.json'), 'utf8')).find(j => j.id === started.id);
    assert.equal(saved.appUpdateFrom, '0.2.2', 'the job was saved with the version it started from before the app was ended');
    // the app is gone; the installer starts the new one (0.2.3) on the same data
    process.env.STRATA_APP_VERSION = '0.2.3';
    const again = createNativeControl({ control: f.control, directory: f.data, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({}), requests: () => [] });
    const job = again.state().jobs.find(j => j.id === started.id);
    assert.equal(job.state, 'done');
    assert.equal(job.error, null);
    assert.match(job.summary, /updated to 0\.2\.3/);
    await again.close();
    // the same record read by an app that did NOT change version stays an interruption
    process.env.STRATA_APP_VERSION = '0.2.2';
    const same = createNativeControl({ control: f.control, directory: f.data, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({}), requests: () => [] });
    const unfinished = same.state().jobs.find(j => j.id === started.id);
    assert.equal(unfinished.state, 'error');
    assert.match(unfinished.error, /update to 0\.2\.[0-9] did not finish.*releases/s, 'the app came back as the old version: the installer did not finish, and the alert says so');
    await same.close();
  } finally { if (before === undefined) delete process.env.STRATA_APP_VERSION; else process.env.STRATA_APP_VERSION = before; await f.native.close(); f.cleanup(); }
});

test('packaged: an update job an OLDER host left running is finished when the app program file is newer than the job', async () => {
  // <install>/strata-desktop(.exe) as seen from <install>/host; native-control.mjs looks for the name of the platform it runs on
  const exe = new URL(process.platform === 'win32' ? '../../strata-desktop.exe' : '../../strata-desktop', import.meta.url);
  const f = fixture({ withApp: true });
  try {
    const { fileURLToPath } = await import('node:url');
    const old = { id: 'old-1', kind: 'update-strata', state: 'running', startedAt: new Date(Date.now() - 60_000).toISOString(), summary: 'Starting…', result: null, error: null };
    writeFileSync(join(f.data, 'native-jobs.json'), JSON.stringify([old]));
    const mk = () => createNativeControl({ control: f.control, directory: f.data, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({}), requests: () => [] });
    const before = mk();
    assert.equal(before.state().jobs[0].state, 'error', 'no new program file: still an interruption');
    await before.close();
    writeFileSync(join(f.data, 'native-jobs.json'), JSON.stringify([old]));
    writeFileSync(fileURLToPath(exe), 'new program');
    try {
      const after = mk();
      assert.equal(after.state().jobs[0].state, 'done');
      assert.match(after.state().jobs[0].summary, /updated/i);
      await after.close();
    } finally { rmSync(fileURLToPath(exe), { force: true }); }
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: "Update engine" works in the first seconds, before the native snapshot has connected', async () => {
  const f = fixture();
  try {
    assert.equal(f.native.state().available, false, 'nothing has been read yet');
    const started = f.native.operation('update-engine', {});      // used to throw "Connect the native Strata installation first."
    const job = await finish(f.native, started.id);
    assert.equal(job.state, 'done', job.error);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: a chosen engine file goes to Control; only a file in the own incoming folder is deleted afterwards', async () => {
  const f = fixture();
  try {
    const incoming = join(f.data, 'incoming-engine');
    mkdirSync(incoming, { recursive: true });
    const mine = join(incoming, 'strata-engine-x.tar.gz'), theirs = join(f.data, 'keep-me.tar.gz');
    writeFileSync(mine, 'x'); writeFileSync(theirs, 'y');
    const job = await finish(f.native, f.native.operation('install-engine-file', { path: mine }).id);
    assert.equal(job.state, 'done', job.error);
    assert.deepEqual(f.calls.at(-1), ['install-file', mine]);
    assert.equal(existsSync(mine), false, 'the copy the page uploaded is removed');
    const other = await finish(f.native, f.native.operation('install-engine-file', { path: theirs }).id);
    assert.equal(other.state, 'done', other.error);
    assert.equal(existsSync(theirs), true, 'a file anywhere else is never deleted');
    const none = await finish(f.native, f.native.operation('install-engine-file', {}).id);
    assert.equal(none.state, 'error');
    assert.equal(f.calls.filter(c => c[0] === 'install-file').length, 2, 'no file, no install');
  } finally { await f.native.close(); f.cleanup(); }
});

test('first-run card data: an offer with its size, nothing-fits with reasons, offline, and an engine already there', async () => {
  const gpus = [{ name: 'NVIDIA GeForce RTX 5070', vramGb: 12 }];
  const report = engine => ({ ok: true, hardware: { nvidia_driver: '581.15', backend: 'cuda' }, engine });
  const offer = await engineFirstRun({ control: { updateCheck: async () => report({ status: 'install', installed: null, offered: { version: '0.1.48', variant: 'cuda13', backend: 'cuda' }, download_bytes: 581299508 }) }, gpus });
  assert.deepEqual([offer.status, offer.offer.bytes, offer.driver, offer.gpus[0].name], ['offer', 581299508, '581.15', 'NVIDIA GeForce RTX 5070']);
  const none = await engineFirstRun({ control: { updateCheck: async () => report({ status: 'none-for-this-pc', installed: null, offered: null, reasons: ['needs NVIDIA driver 580.65 or newer, this PC has 560.1'] }) }, gpus });
  assert.equal(none.status, 'none');
  assert.match(none.reasons[0], /driver 580/);
  const installed = await engineFirstRun({ control: { updateCheck: async () => report({ status: 'current', installed: { version: '0.1.48' }, offered: null }) }, gpus });
  assert.equal(installed.status, 'installed');
  const offline = await engineFirstRun({ control: { updateCheck: async () => { throw new Error('the update feed could not be read: could not read https://github.com/x (URLError: [WinError 10060] timed out)'); } }, gpus });
  assert.equal(offline.status, 'offline');
});

test('without the shell bridge "Update Strata" still updates Control and says nothing false about the app', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('update-strata', {}).id);
    assert.equal(job.state, 'done', job.error);
    assert.doesNotMatch(job.summary, /app update is installed/);
  } finally { await f.native.close(); f.cleanup(); }
});

test('packaged: opening the app, listing models and loading one never reach the update step', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    const load = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
    assert.equal(load.state, 'done', load.error);
    assert.deepEqual(f.calls, [], 'no update, check or rollback call: downloads happen only when the user presses Update');
    assert.equal(f.spawned.length, 0);
  } finally { await f.native.close(); f.cleanup(); }
});

test('plain words in the app: codes, hashes and exit codes stay in details, never in the main text', async () => {
  const { plainUpdateError } = await import('../server/update-messages.mjs');
  const raw = [
    'the download of http://x/e.tar.gz was interrupted after 100 of 200 bytes (ConnectionResetError)',
    'The new engine 0.1.46 did not pass its test start (it did not answer a one-token probe (server_exited))',
    'http://x/e.tar.gz does not match the SHA-256 in the signed feed; the download was deleted',
    'this build trusts no update key (update_keys.json is empty)',
    'app-1 is serving.',
    'something unexpected: exit 9',
  ];
  for (const text of raw) {
    const plain = plainUpdateError(new Error(text));
    assert.doesNotMatch(plain.message, /SHA|HTTP|http|exit|0\.1\.|server_exited|\.tar|key|bytes|probe/i, plain.message);
    assert.equal(plain.details, text);
    assert.ok(plain.next.length > 0);
  }
  assert.match(plainUpdateError(new Error(raw[0])).message, /internet connection/);
  assert.match(plainUpdateError(new Error(raw[4])).message, /Unload the model/);
});

test('a failed update job shows plain words and keeps the technical reason as details', async () => {
  const f = fixture({ update: async () => { throw new Error('the download of http://x/e.tar.gz was interrupted after 1 of 2 bytes (ConnectionResetError)'); } });
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('update-engine', {}).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /Couldn't reach the update server\. Check your internet connection, then press Update engine in Settings\./);
    assert.doesNotMatch(job.error, /http|bytes|Reset/);
    assert.match(job.details, /ConnectionResetError/);
  } finally { await f.native.close(); f.cleanup(); }
});

test('offline, the signed-feed check says there is no internet connection in plain words and keeps the reason in details', async () => {
  for (const raw of ['fetch failed', '<urlopen error [Errno 11001] getaddrinfo failed>', 'getaddrinfo ENOTFOUND updates.example']) {
    const control = { version: '0.1.45', async updateCheck() { throw new Error(raw); } };
    const result = await checkComponentUpdates({ control, appVersion: '0.2.0' });
    assert.equal(result.strata.status, 'error', raw);
    assert.equal(result.strata.message, "No internet connection. Try again when you're online.", raw);
    assert.equal(result.engine.message, "No internet connection. Try again when you're online.", raw);
    assert.equal(result.strata.details, raw);
  }
});

test('the app updater and the host never reach the network by themselves: only check() does', async () => {
  const calls = [];
  const updater = createAppUpdater({ env: { STRATA_SHELL_BRIDGE: 'http://127.0.0.1:4000', STRATA_SHELL_TOKEN: 'tok' }, fetcher: async url => { calls.push(url); return Response.json({ available: false }); } });
  assert.ok(updater);
  assert.equal(calls.length, 0, 'creating the updater makes no call');
  await updater.check();
  assert.equal(calls.length, 1);
});
