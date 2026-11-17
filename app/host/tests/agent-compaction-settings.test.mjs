/**
 * The Settings -> Chat control for when the assistant summarises older messages: the saved choice wins over the
 * environment, the environment wins over the default, the host's routes read and write it, and it survives a restart.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { buildProfile, compactionThreshold, headroomFraction, localOutputBudget, setCompactionChoice, thresholdRatio, DEFAULT_HEADROOM_FRACTION, DEFAULT_THRESHOLD_RATIO } from '../server/agent/profile.mjs';

const HOST = resolve(import.meta.dirname, '..', 'server', 'index.mjs');
const freePort = () => new Promise(r => { const s = createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => r(p)); }); });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } };

test('the choice made in the app wins over the environment, and null puts the environment back', () => {
  try {
    const env = { STRATA_COMPACTION_THRESHOLD_RATIO: '0.9', STRATA_COMPACTION_HEADROOM_FRACTION: '0.05' };
    setCompactionChoice({});                                   // nothing chosen yet
    assert.equal(thresholdRatio(env), 0.9);
    assert.equal(headroomFraction(env), 0.05);
    setCompactionChoice({ thresholdRatio: 0.75, headroomFraction: 0.2 });
    assert.equal(thresholdRatio(env), 0.75, 'the person’s choice answers, not the environment');
    assert.equal(headroomFraction(env), 0.2);
    setCompactionChoice({ thresholdRatio: null, headroomFraction: 0.2 });  // Automatic on one name, a choice on the other
    assert.equal(thresholdRatio(env), 0.9, 'automatic follows the environment again');
    assert.equal(headroomFraction(env), 0.2, 'the other name keeps its choice');
    setCompactionChoice({ thresholdRatio: 5, headroomFraction: -1 });   // nonsense is not a choice
    assert.equal(thresholdRatio(env), 0.9);
    assert.equal(headroomFraction(env), 0.05);
  } finally {
    setCompactionChoice({});
  }
});

test('the threshold the control shows is the one the Harness will use', () => {
  try {
    setCompactionChoice({ thresholdRatio: 0.95, headroomFraction: 0.05 });
    const w = 524288, out = localOutputBudget(w);
    const tokens = compactionThreshold(w, out, {});
    assert.equal(tokens, Math.floor(w - out - Math.floor((w - out) * 0.05)));
    assert.ok(tokens > Math.floor(w * 0.8), 'later than the default on a 512K window');
    assert.ok(tokens + out <= w, 'a prompt at the threshold plus a full reply still fits');
  } finally {
    setCompactionChoice({});
    assert.equal(compactionThreshold(524288, localOutputBudget(524288), {}), Math.floor(524288 * DEFAULT_THRESHOLD_RATIO));
    assert.equal(headroomFraction({}), DEFAULT_HEADROOM_FRACTION);
  }
});

test('an unset setting is not a number: null, "" and false never become a legal 0', () => {
  try {
    for (const empty of [null, undefined, '', false, 'abc', 5, -1]) {
      setCompactionChoice({ thresholdRatio: empty, headroomFraction: empty });
      assert.equal(thresholdRatio({}), DEFAULT_THRESHOLD_RATIO, `thresholdRatio ${String(empty)}`);
      assert.equal(headroomFraction({}), DEFAULT_HEADROOM_FRACTION, `headroomFraction ${String(empty)} keeps the reserve`);
    }
    setCompactionChoice({ headroomFraction: 0 });
    assert.equal(headroomFraction({}), 0, 'a real 0 is a choice the person can make');
  } finally {
    setCompactionChoice({});
  }
});

test('the choice reaches the assistant’s profile, where the Harness reads it', () => {
  try {
    setCompactionChoice({ thresholdRatio: 0.93, headroomFraction: 0.05 });
    const profile = buildProfile({ local: [{ id: 'a', modelId: 'm', baseUrl: 'http://x', context: 524288 }], connections: [] });
    assert.match(profile.patch, /"provider": "strata-local",\s*"model": "m",\s*"headroomTokens": \d+,\s*"thresholdRatio": 0\.93/);
    setCompactionChoice({});
    const automatic = buildProfile({ local: [{ id: 'a', modelId: 'm', baseUrl: 'http://x', context: 524288 }], connections: [] });
    assert.match(automatic.patch, /"thresholdRatio": 0\.8/, 'and Automatic writes Strata’s own number back');
  } finally {
    setCompactionChoice({});
  }
});

async function startHost(t, home) {
  const port = await freePort();
  const ws = join(home, 'workspace'); mkdirSync(ws, { recursive: true });
  const env = { ...process.env, STRATA_HOME: home, STRATA_PORT: String(port), STRATA_WORKSPACE: ws, STRATA_REAL_ENGINE: '0', STRATA_COMPONENTS_DIR: join(home, 'components') };
  delete env.STRATA_HARNESS_DIR; delete env.STRATA_COMPACTION_THRESHOLD_RATIO; delete env.STRATA_COMPACTION_HEADROOM_FRACTION;
  const child = spawn(process.execPath, [HOST], { env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  let log = '';
  child.stdout.on('data', d => { log += d; }); child.stderr.on('data', d => { log += d; });
  const stop = async () => {
    if (child.exitCode !== null) return;
    if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true });
    else child.kill('SIGTERM');
    await new Promise(r => child.once('exit', r));
  };
  t.after(stop);
  const base = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 100; i++) {
    try { if ((await fetch(`${base}/api/agent/compaction`)).ok) break; } catch { /* not up yet */ }
    if (child.exitCode !== null) throw new Error(`the host exited: ${log.slice(-500)}`);
    await sleep(200);
  }
  const get = async () => { const res = await fetch(`${base}/api/agent/compaction`); return { status: res.status, body: await res.json() }; };
  const post = async body => {
    const res = await fetch(`${base}/api/agent/compaction`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    return { status: res.status, body: await res.json().catch(() => null) };
  };
  return { get, post, stop, log: () => log };
}

test('the host reads and writes the choice, refuses nonsense, and keeps it across a restart', async t => {
  const home = mkdtempSync(join(tmpdir(), 'strata-compact-setting-'));
  t.after(() => cleanup(home));
  const host = await startHost(t, home);

  const fresh = await host.get();
  assert.equal(fresh.status, 200);
  assert.equal(fresh.body.choice.thresholdRatio, null, 'a new install has chosen nothing');
  assert.equal(fresh.body.effective.thresholdRatio, DEFAULT_THRESHOLD_RATIO);
  assert.equal(fresh.body.source.thresholdRatio, 'default');

  const saved = await host.post({ thresholdRatio: 0.95, headroomFraction: 0.05 });
  assert.equal(saved.status, 200);
  assert.equal(saved.body.effective.thresholdRatio, 0.95);
  assert.equal(saved.body.effective.headroomFraction, 0.05);
  assert.equal(saved.body.source.thresholdRatio, 'setting');
  assert.match(readFileSync(join(home, 'data', 'state.json'), 'utf8'), /"agentChat"/, 'it is written to the host’s settings file');

  for (const bad of [1.5, 0, -0.2, 'abc']) {
    const res = await host.post({ thresholdRatio: bad });
    assert.equal(res.status, 400, `${bad} is refused`);
  }
  assert.equal((await host.get()).body.effective.thresholdRatio, 0.95, 'a refused write changed nothing');

  await host.stop();
  const again = await startHost(t, home);
  assert.equal((await again.get()).body.effective.thresholdRatio, 0.95, 'the choice answers again after a restart');

  const back = await again.post({ thresholdRatio: null, headroomFraction: null });
  assert.equal(back.body.effective.thresholdRatio, DEFAULT_THRESHOLD_RATIO, 'Automatic returns to Strata’s own number');
});
