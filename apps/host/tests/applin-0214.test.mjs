import './helpers/isolate-profile.mjs';   // a test run never writes the owner's real Strata profile
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createControl, ControlError } from '../server/control-runtime.mjs';
import { plainUpdateError, plainEngineError, UPDATE_WAITING_FOR_LOCK } from '../server/update-messages.mjs';
import { createNativeControl } from '../server/native-control.mjs';
import { checkComponentUpdates, engineFailedBeforeText, engineFirstRun } from '../server/updates.mjs';

// Lane APP-LIN (0.2.14): the update and first-run fixes of the 0.2.13 GUI tests.

const LOCK = 'another update is already running for this component folder';

function fixture({ update, cfgExtra = {}, reload } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-applin-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;
  const config = join(install, 'strata-iq2_xs.json');
  mkdirSync(join(root, 'control', 'data'), { recursive: true });
  writeFileSync(join(root, 'control', 'data', 'expert-profile.bin'), 'x');
  writeFileSync(config, JSON.stringify({ exe: 'component:strata-engine', args: ['--max-context', '32768'], model_name: 'Qwen', port: 8080, ...cfgExtra }));
  const calls = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.1.60',
    async installedModels() { return { install_dir: install, models: [{ id: 'iq2_xs', name: 'Qwen', source_config: config, component_status: 'present', context_tokens: 32768, backend: 'cuda' }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: '/c', engine: { version: '0.1.60', platform: 'linux-x86_64', backend: 'cuda', exe: 'e', root: 'r' } }; },
    async saveProfile() { return {}; }, async start() { return { ok: true, state: 'ready' }; }, async stop() { return { ok: true }; },
    ...(reload ? { reload() { calls.push(['reload']); return reload(this); } } : {}),
    async update(options) {
      calls.push(['update', options.engine, options.control, options.force === true]);
      await update?.(options, calls.length);
      return { ok: true, results: [{ component: 'engine', status: 'updated', installed_after: { version: '0.1.61' } }] };
    },
  };
  const native = createNativeControl({ control, directory: data, lockWaitMs: 300, lockPollMs: 20, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({}), requests: () => [] });
  return { native, calls, root, config, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 400; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };

// ------------------------------------------------------------------ item 4: exit 7 is two things

test('exit 7 from the update lock is "an update is already running", not "a model is running"', () => {
  for (const error of [Object.assign(new Error(LOCK), { exitCode: 7 }), Object.assign(new Error(LOCK), { exitCode: 7, code: 'engine_serving' }), new ControlError(LOCK, { exitCode: 7, code: 'update_busy' })]) {
    const plain = plainUpdateError(error, 'engine', 'Continue');
    assert.match(plain.message, /An update is already running/);
    assert.doesNotMatch(plain.message, /model is running/);
  }
  // a real serving refusal keeps its words
  const serving = plainUpdateError(Object.assign(new Error('app-1 is serving.'), { exitCode: 7, code: 'engine_serving' }));
  assert.match(serving.message, /A model is running/);
  assert.match(plainEngineError(Object.assign(new Error(LOCK), { exitCode: 7 })).message, /An update is running/);
  assert.match(plainEngineError(Object.assign(new Error('x is serving'), { exitCode: 7 })).message, /A model is running/);
});

test('createControl tells the lock from a serving engine by the text, for both failure shapes', async () => {
  const lockedStep = createControl({ control: { dir: '/c', python: 'p', cli: 'c' }, run: async () => ({ code: 7, json: { ok: false, results: [{ component: 'engine', status: 'failed', error: LOCK, exit_code: 7 }] }, stderr: '' }) });
  await assert.rejects(lockedStep.update({}), e => e.code === 'update_busy' && e.exitCode === 7);
  const lockedWhole = createControl({ control: { dir: '/c', python: 'p', cli: 'c' }, run: async () => ({ code: 7, json: { ok: false, error: { message: LOCK } }, stderr: '' }) });
  await assert.rejects(lockedWhole.update({}), e => e.code === 'update_busy');
  await assert.rejects(lockedWhole.updateCheck({}), e => e.code === 'update_busy');
  const serving = createControl({ control: { dir: '/c', python: 'p', cli: 'c' }, run: async () => ({ code: 7, json: { ok: false, results: [{ component: 'engine', status: 'failed', error: 'app-1 is serving.', exit_code: 7 }] }, stderr: '' }) });
  await assert.rejects(serving.update({}), e => e.code === 'engine_serving');
});

test('the first-run Download waits for the lock the start-up check holds, then installs', async () => {
  const f = fixture({ update: async (_o, n) => { if (n < 3) throw new ControlError(LOCK, { exitCode: 7, code: 'update_busy' }); } });
  try {
    const started = f.native.operation('update-engine', { firstRun: true });
    let sawWaiting = false;
    for (let i = 0; i < 50 && !sawWaiting; i++) { sawWaiting = f.native.state().jobs.find(j => j.id === started.id)?.summary === UPDATE_WAITING_FOR_LOCK; await new Promise(r => setTimeout(r, 5)); }
    const job = await finish(f.native, started.id);
    assert.equal(job.state, 'done', job.error);
    assert.equal(f.calls.length, 3, 'tried again while the lock was held');
    assert.ok(sawWaiting, 'the job said it was waiting');
  } finally { await f.native.close(); f.cleanup(); }
});

test('a lock that stays held ends with the lock words, after the wait', async () => {
  const f = fixture({ update: async () => { throw new ControlError(LOCK, { exitCode: 7, code: 'update_busy' }); } });
  try {
    const job = await finish(f.native, f.native.operation('update-engine', { firstRun: true }).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /An update is already running/);
    assert.doesNotMatch(job.error, /model is running/);
    assert.ok(f.calls.length >= 2);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a serving refusal is not retried', async () => {
  const f = fixture({ update: async () => { throw new ControlError('app-1 is serving.', { exitCode: 7, code: 'engine_serving' }); } });
  try {
    const job = await finish(f.native, f.native.operation('update-engine', {}).id);
    assert.equal(job.state, 'error');
    assert.equal(f.calls.length, 1);
    assert.match(job.error, /A model is running/);
  } finally { await f.native.close(); f.cleanup(); }
});

// ------------------------------------------------------------------ item 3: a failed engine update is not "up to date"

test('an engine version that failed its test start here is offered again with the owner-approved words', async () => {
  const note = '0.1.59 failed its test start on 2026-11-09 (read_expert_profile: cannot open ...); run `strata update --force` to try it again';
  const control = { version: '0.1.59', async updateCheck() { return { ok: true, feed: { sequence: 123 }, engine: { installed: { version: '0.1.58' }, offered: { version: '0.1.59' }, status: 'failed-before', note, download_bytes: 723e6 }, control: { installed: '0.1.59', offered: '0.1.59', status: 'current' }, app: { status: 'not-in-feed' } }; } };
  const result = await checkComponentUpdates({ control, appVersion: '0.2.13' });
  assert.equal(result.engine.status, 'available');
  assert.equal(result.engine.canInstall, true);
  assert.equal(result.engine.retry, true);
  assert.equal(result.engine.message, "Engine 0.1.59 didn't start correctly on this computer, so Strata kept engine 0.1.58. Press Update engine to try again.");
  assert.equal(result.engine.message, engineFailedBeforeText('0.1.59', '0.1.58'));
  assert.equal(result.engine.details, note, 'the reason stays behind Details');
  assert.doesNotMatch(result.engine.message, /up to date/);
  // a version the person switched away from (skipped) still reads as current
  const skipped = await checkComponentUpdates({ control: { ...control, async updateCheck() { return { ok: true, feed: {}, engine: { installed: { version: '0.1.58' }, offered: { version: '0.1.59' }, status: 'skipped' }, control: {}, app: {} }; } }, appVersion: '0.2.13' });
  assert.equal(skipped.engine.status, 'current');
});

test('the app\'s Update engine asks Control to try a held-back version again (--force)', async () => {
  const f = fixture();
  try {
    const job = await finish(f.native, f.native.operation('update-engine', {}).id);
    assert.equal(job.state, 'done', job.error);
    assert.deepEqual(f.calls[0], ['update', true, false, true]);
  } finally { await f.native.close(); f.cleanup(); }
});

// ------------------------------------------------------------------ item 2: a stale /tmp/.mount_* path

test('the engine update repairs run-config paths of a gone AppImage mount before Control test-starts with it', async () => {
  const gone = '/tmp/.mount_StratappjGco/usr/lib/Strata/control';
  let seen = null;
  const f = fixture({ cfgExtra: { cwd: gone, args: ['--expert-profile', gone + '/data/expert-profile.bin', '--max-context', '32768'] },
    update: async () => { seen = JSON.parse(readFileSync(f.config, 'utf8')); } });
  try {
    const job = await finish(f.native, f.native.operation('update-engine', {}).id);
    assert.equal(job.state, 'done', job.error);
    assert.ok(seen, 'Control was called');
    assert.equal(seen.cwd, join(f.root, 'control'));
    assert.equal(seen.args[1], join(f.root, 'control', 'data', 'expert-profile.bin'));
  } finally { await f.native.close(); f.cleanup(); }
});

// ------------------------------------------------------------------ item 1: the app update's job across the restart

test('an app update job the old host closed as "cancelled" is read as finished by the new version; this host keeps it running when it closes', async () => {
  const before = process.env.STRATA_APP_VERSION;
  const root = mkdtempSync(join(tmpdir(), 'strata-applin-jobs-'));
  try {
    const job = { id: 'j1', kind: 'update-strata', state: 'cancelled', startedAt: new Date().toISOString(), summary: 'Downloading the app update: 100%', appUpdateFrom: '0.2.13', appUpdateTo: '0.2.14' };
    writeFileSync(join(root, 'native-jobs.json'), JSON.stringify([job, { ...job, id: 'j2', appUpdateFrom: undefined }]));
    process.env.STRATA_APP_VERSION = '0.2.14';
    const mk = () => createNativeControl({ control: { control: { dir: '/c' }, version: '0.1.60' }, directory: root, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({}), requests: () => [] });
    const next = mk();
    const jobs = next.state().jobs;
    assert.equal(jobs.find(j => j.id === 'j1').state, 'done');
    assert.match(jobs.find(j => j.id === 'j1').summary, /updated to 0\.2\.14/);
    assert.equal(jobs.find(j => j.id === 'j2').state, 'cancelled', 'a cancelled job that was not an app update stays cancelled');
    await next.close();
    // the same version came back: the update did not finish, and it says so
    writeFileSync(join(root, 'native-jobs.json'), JSON.stringify([{ ...job, appUpdateFrom: '0.2.14' }]));
    const same = mk();
    assert.equal(same.state().jobs[0].state, 'error');
    await same.close();
  } finally { if (before === undefined) delete process.env.STRATA_APP_VERSION; else process.env.STRATA_APP_VERSION = before; rmSync(root, { recursive: true, force: true }); }
});

test('closing the host during an app update leaves its job running (the next start settles it)', async () => {
  const before = process.env.STRATA_APP_VERSION;
  process.env.STRATA_APP_VERSION = '0.2.14';
  const root = mkdtempSync(join(tmpdir(), 'strata-applin-close-'));
  const control = { control: { dir: '/c' }, version: '0.1.60', async update() { return { ok: true, results: [] }; } };
  const appUpdater = { check: async () => ({ available: true, version: '0.2.15' }), install: () => new Promise(() => {}) };
  const native = createNativeControl({ control, appUpdater, directory: root, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({}), requests: () => [] });
  try {
    const started = native.operation('update-strata', {});
    for (let i = 0; i < 100 && !native.state().jobs.find(j => j.id === started.id)?.appUpdateFrom; i++) await new Promise(r => setTimeout(r, 10));
    await native.close();
    assert.equal(native.state().jobs.find(j => j.id === started.id).state, 'running');
  } finally { if (before === undefined) delete process.env.STRATA_APP_VERSION; else process.env.STRATA_APP_VERSION = before; rmSync(root, { recursive: true, force: true }); }
});

// ------------------------------------------------------------------ item 5: the first-run card offers the Control update

const feedReport = control => ({ ok: true, feed: {}, hardware: { backend: 'sycl' }, engine: { status: 'install', installed: null, offered: { version: '0.1.60', variant: '', backend: 'sycl' }, download_bytes: 300e6 }, control, app: {} });

test('the first-run card names a newer Control in the feed, and nothing when there is none', async () => {
  const withControl = await engineFirstRun({ control: { updateCheck: async () => feedReport({ installed: '0.1.60', offered: '0.1.60.1', status: 'available', download_bytes: 120e6 }) } });
  assert.equal(withControl.status, 'offer');
  assert.deepEqual(withControl.control, { version: '0.1.60.1', bytes: 120e6 });
  const current = await engineFirstRun({ control: { updateCheck: async () => feedReport({ installed: '0.1.60', offered: '0.1.60', status: 'current' }) } });
  assert.equal(current.control, null);
});

test('the first-run Download with the Control: engine first, then Control, then the app switches to it while nothing runs', async () => {
  const f = fixture({ reload: self => { self.control = { dir: '/new/control' }; self.version = '0.1.60.1'; return true; } });
  try {
    const job = await finish(f.native, f.native.operation('update-engine', { firstRun: true, withControl: true }).id);
    assert.equal(job.state, 'done', job.error);
    assert.deepEqual(f.calls.map(c => c.slice(0, 3)), [['update', true, false], ['update', false, true], ['reload']]);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a Control update that fails on first run keeps the engine and the job; no switch', async () => {
  const f = fixture({ update: async options => { if (options.control) throw new Error('the package failed its check'); }, reload: () => true });
  try {
    const job = await finish(f.native, f.native.operation('update-engine', { firstRun: true, withControl: true }).id);
    assert.equal(job.state, 'done', job.error);
    assert.ok(!f.calls.some(c => c[0] === 'reload'));
  } finally { await f.native.close(); f.cleanup(); }
});

test('without withControl (or from Settings) the engine update never touches Control', async () => {
  const f = fixture({ reload: () => true });
  try {
    await finish(f.native, f.native.operation('update-engine', { firstRun: true }).id);
    await finish(f.native, f.native.operation('update-engine', {}).id);
    assert.deepEqual(f.calls.map(c => c.slice(0, 3)), [['update', true, false], ['update', true, false]]);
  } finally { await f.native.close(); f.cleanup(); }
});

test('createControl.reload switches to a newer Control and the next call runs from it; the same folder is no switch', async () => {
  const seen = [];
  const first = { dir: '/a/control', python: '/a/py', cli: '/a/strata.py', bundle: { version: '0.1.60' } };
  const ctl = createControl({ control: first, run: async (args) => { seen.push(args[0]); return { code: 0, json: { ok: true } }; } });
  assert.equal(ctl.reload({ find: () => ({ ...first }) }), false);
  assert.equal(ctl.reload({ find: () => null }), false);
  const next = { dir: '/b/control', python: '/b/py', cli: '/b/strata.py', bundle: { version: '0.1.60.1' } };
  assert.equal(ctl.reload({ find: () => next }), true);
  assert.equal(ctl.control.dir, '/b/control');
  assert.equal(ctl.version, '0.1.60.1');
});
