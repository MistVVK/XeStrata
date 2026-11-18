// 0.2.17 lane APP, items 1 and 6 (0.2.16 retest N-A1, N-A2, N-A3):
//  - N-A1: about 1 fresh start in 8 the Welcome step said "Strata couldn't read its download list" for good. Control lets one `update --check` hold its update lock; the
//    first-run read of the app window and a second reader (a browser tab, Settings, the test driver) asked at the same moment and the loser got "another update is already
//    running", which the card shows as a list it cannot read.
//  - N-A3: Settings > About "Check for updates" in the first seconds said "another update is already running".
//  - N-A2: after a successful app update the old host's log said "update-strata failed: fetch failed" (the installer ends the shell, its address stops answering).
import test from 'node:test';
import assert from 'node:assert/strict';
import { createControl } from '../server/control-runtime.mjs';
import { engineFirstRun, checkComponentUpdates } from '../server/updates.mjs';
import * as appUpdater from '../server/app-updater.mjs';

const { createAppUpdater } = appUpdater;
const connectionLost = appUpdater.connectionLost ?? (() => undefined);   // (the check below is what says it must exist)

const control = { dir: '/c', python: 'p', cli: 'c' };
const LOCK = { code: 7, json: { ok: false, error: { message: 'another update is already running for this component folder' } }, text: '', stderr: '' };
const REPORT = { ok: true, engine: { status: 'install', offered: { version: '0.1.60', variant: 'cuda12' }, download_bytes: 875e6 }, hardware: { nvidia_driver: '580' }, control: {}, app: {} };

/** Control's `update --check` as it is: it takes the update lock for its whole run and refuses a second one at once. */
function lockedCheck({ ms = 20, busyFirst = 0 } = {}) {
  const state = { locked: false, runs: 0, refused: 0, busyLeft: busyFirst };
  const run = async args => {
    assert.equal(args[0], 'update');
    state.runs++;
    if (state.locked || state.busyLeft > 0) { if (state.busyLeft > 0) state.busyLeft--; state.refused++; return LOCK; }
    state.locked = true;
    try { await new Promise(r => setTimeout(r, ms)); return { code: 0, json: REPORT, text: '', stderr: '' }; }
    finally { state.locked = false; }
  };
  return { state, run };
}

test('N-A1: two readers asking at the same moment both get the feed (one check runs, the other takes its answer)', async () => {
  const lock = lockedCheck();
  const ctl = createControl({ control, run: lock.run });
  const [a, b] = await Promise.all([engineFirstRun({ control: ctl, gpus: [{ name: 'Tesla P100-PCIE-16GB', vramGb: 16 }] }), engineFirstRun({ control: ctl, gpus: [] })]);
  assert.equal(a.status, 'offer', JSON.stringify(a));
  assert.equal(b.status, 'offer', JSON.stringify(b));
  assert.equal(lock.state.refused, 0, 'no second check was started against the lock');
  assert.equal(lock.state.runs, 1);
});

test('N-A1: the first-run read and Settings > About asking together both answer', async () => {
  const lock = lockedCheck();
  const ctl = createControl({ control, run: lock.run });
  const [first, about] = await Promise.all([engineFirstRun({ control: ctl }), checkComponentUpdates({ control: ctl, appVersion: '0.2.16' })]);
  assert.equal(first.status, 'offer');
  assert.notEqual(about.engine.status, 'error', JSON.stringify(about));
  assert.equal(lock.state.refused, 0);
});

test('a finished check is not kept: the next press asks the feed again', async () => {
  const lock = lockedCheck({ ms: 1 });
  const ctl = createControl({ control, run: lock.run });
  await ctl.updateCheck();
  await ctl.updateCheck();
  assert.equal(lock.state.runs, 2);
});

test('N-A3: a lock held by someone else (a Terminal update, the end of a check) is waited for, then the check answers', async () => {
  const lock = lockedCheck({ ms: 1, busyFirst: 3 });
  const sleeps = [];
  const ctl = createControl({ control, run: lock.run, sleep: async ms => { sleeps.push(ms); } });
  const report = await ctl.updateCheck();
  assert.equal(report.ok, true);
  assert.equal(lock.state.runs, 4, 'three refusals, then the answer');
  assert.equal(sleeps.length, 3);
  const first = await engineFirstRun({ control: createControl({ control, run: lockedCheck({ ms: 1, busyFirst: 2 }).run, sleep: async () => {} }) });
  assert.equal(first.status, 'offer', 'the first-run card does not show "couldn’t read its download list" for a lock that frees in a moment');
});

test('a lock that stays held is still reported as the lock after the wait (an update that is really running)', async () => {
  const lock = lockedCheck({ ms: 1, busyFirst: 1000 });
  const sleeps = [];
  const ctl = createControl({ control, run: lock.run, sleep: async ms => { sleeps.push(ms); }, lockWaitMs: 5000, lockPollMs: 1000 });
  await assert.rejects(ctl.updateCheck(), error => { assert.equal(error.code, 'update_busy'); return true; });
  assert.equal(sleeps.length, 5, 'it waited five seconds, one poll each');
  assert.equal(lock.state.runs, 6);
});

test('another failure (not the lock) is not retried', async () => {
  let runs = 0;
  const ctl = createControl({ control, sleep: async () => { throw new Error('must not wait'); }, run: async () => { runs++; return { code: 5, json: null, text: '', stderr: 'could not read the feed' }; } });
  await assert.rejects(ctl.updateCheck(), /could not read the feed/);
  assert.equal(runs, 1);
});

// ---------------------------------------------------------------------------------- N-A2

const bridge = { STRATA_SHELL_BRIDGE: 'http://127.0.0.1:4000', STRATA_SHELL_TOKEN: 'tok' };
const lost = () => Object.assign(new TypeError('fetch failed'), { cause: Object.assign(new Error('connect ECONNREFUSED 127.0.0.1:4000'), { code: 'ECONNREFUSED' }) });

test('N-A2: while the installer ends the app, the shell stops answering; the host waits for its own end instead of failing the update', async () => {
  let clock = 0, polls = 0;
  const fetcher = async url => {
    if (url.endsWith('/app-update/install')) return Response.json({ state: 'downloading' });
    polls++;
    if (polls === 1) return Response.json({ state: 'installing' });
    if (polls < 6) throw lost();                         // the shell is gone
    return Response.json({ state: 'done' });             // (a shell that came back)
  };
  const updater = createAppUpdater({ env: bridge, fetcher, sleep: async ms => { clock += ms; } });
  const done = await updater.install({ pollMs: 500, restartGraceMs: 20_000, now: () => clock });
  assert.equal(done.state, 'done');
  assert.equal(polls, 6);
});

test('N-A2: a shell that stays silent past the grace is an error, as before', async () => {
  let clock = 0;
  const fetcher = async url => { if (url.endsWith('/app-update/install')) return Response.json({ state: 'downloading' }); throw lost(); };
  const updater = createAppUpdater({ env: bridge, fetcher, sleep: async ms => { clock += ms; } });
  await assert.rejects(updater.install({ pollMs: 1000, restartGraceMs: 10_000, now: () => clock }), /fetch failed/);
  assert.ok(clock >= 10_000);
});

test('N-A2: an answer from the shell that says the update failed is never taken for a lost connection', async () => {
  const fetcher = async url => Response.json(url.endsWith('/status') ? { state: 'error', error: 'signature mismatch' } : { state: 'downloading' });
  const updater = createAppUpdater({ env: bridge, fetcher, sleep: async () => {} });
  await assert.rejects(updater.install(), /signature mismatch/);
  const refused = async url => url.endsWith('/status') ? Response.json({ error: 'not allowed' }, { status: 401 }) : Response.json({});
  await assert.rejects(createAppUpdater({ env: bridge, fetcher: refused, sleep: async () => {} }).install(), /not allowed/);
});

test('N-A2: the start of the install failing to connect is still an error at once (nothing was started)', async () => {
  const updater = createAppUpdater({ env: bridge, fetcher: async () => { throw lost(); }, sleep: async () => {} });
  await assert.rejects(updater.install(), /fetch failed/);
});

test('connectionLost: refused, reset, timed out and "fetch failed" are a lost connection; an error text from the shell is not', () => {
  assert.equal(connectionLost(lost()), true);
  assert.equal(connectionLost(Object.assign(new Error('x'), { name: 'TimeoutError' })), true);
  assert.equal(connectionLost(new Error('The app update failed.')), false);
  assert.equal(connectionLost(new Error('signature mismatch')), false);
});
