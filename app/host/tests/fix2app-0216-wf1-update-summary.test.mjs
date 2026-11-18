import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';
import { createAppUi, BROWSER_UPDATE_MESSAGE } from '../server/app-ui.mjs';
import { APP_UPDATER_NOT_READY, APP_CHECK_FAILED } from '../server/update-messages.mjs';

/**
 * W-F1 (0.2.16 retest, WIN item 5): after the Strata row's update worked, the summary said
 * "Strata is updated. It takes effect the next time you open the app. The app updater is not ready yet. Try again in a moment."
 * The second sentence is the app's own step (the shell's signed updater) and read like a failure after a success.
 */
const UPDATED = 'Strata is updated. It takes effect the next time you open the app.';

function fixture({ appUpdater, results = [{ component: 'control', status: 'updated' }] }) {
  const root = mkdtempSync(join(tmpdir(), 'strata-wf1-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;
  const control = {
    control: { dir: join(root, 'control') }, version: '0.1.62',
    async installedModels() { return { install_dir: install, models: [] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: join(root, 'c'), previous: null, engine: { version: '0.1.62', platform: 'windows-x86_64', backend: 'cuda', exe: 'e', root: 'r' } }; },
    async saveProfile() { return {}; }, async start() { return { ok: true, state: 'ready' }; }, async stop() { return { ok: true }; },
    async update() { return { ok: true, results }; },
  };
  const native = createNativeControl({ control, appUpdater, directory: data, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({}), requests: () => [] });
  return { native, data, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 400; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };
const run = async f => { await f.native.refresh(true); return finish(f.native, f.native.operation('update-strata', {}).id); };

test('the window process has not handed over its updater: a finished Strata update says only that Strata is updated', async () => {
  const ui = createAppUi({ env: { STRATA_UI_MODE: 'window' } });       // the real app-ui: mode window, no bridge yet
  const f = fixture({ appUpdater: ui.updater });
  try {
    const job = await run(f);
    assert.equal(job.state, 'done', job.error);
    assert.equal(job.summary, UPDATED);
    assert.doesNotMatch(job.summary, /not ready|couldn't check|Try again/);
  } finally { await f.native.close(); f.cleanup(); }
});

test('the app update feed cannot be reached after Control updated: the job is still done and the summary is not a failure', async () => {
  const f = fixture({ appUpdater: { check: async () => { throw new Error('The Strata app could not check for updates (502).'); }, install: async () => { throw new Error('never'); } } });
  try {
    const job = await run(f);
    assert.equal(job.state, 'done', job.error);
    assert.equal(job.summary, UPDATED);
    assert.equal(job.error ?? null, null);
  } finally { await f.native.close(); f.cleanup(); }
});

test('nothing changed and the app side did not work: that is the whole answer, said plainly (not "everything is up to date")', async () => {
  const ui = createAppUi({ env: { STRATA_UI_MODE: 'window' } });
  let f = fixture({ appUpdater: ui.updater, results: [{ component: 'control', status: 'current' }] });
  try {
    const job = await run(f);
    assert.equal(job.state, 'done', job.error);
    assert.equal(job.summary, APP_UPDATER_NOT_READY);
  } finally { await f.native.close(); f.cleanup(); }
  f = fixture({ appUpdater: { check: async () => { throw new Error('502'); }, install: async () => ({}) }, results: [] });
  try {
    const job = await run(f);
    assert.equal(job.state, 'done', job.error);
    assert.equal(job.summary, APP_CHECK_FAILED);
  } finally { await f.native.close(); f.cleanup(); }
});

test('the reason is kept for support: the update log names what the app check did', async () => {
  const f = fixture({ appUpdater: { check: async () => { throw new Error('The Strata app could not check for updates (502).'); }, install: async () => ({}) } });
  try {
    const lines = [], original = console.log;
    console.log = (...a) => { lines.push(a.join(' ')); };
    try { await run(f); } finally { console.log = original; }
    assert.ok(lines.some(l => /^update · app check: the check did not work \(The Strata app could not check for updates \(502\)\.\)/.test(l)), lines.join('\n'));
  } finally { await f.native.close(); f.cleanup(); }
});

test('browser mode keeps its own plain sentence (it tells the person what to do)', async () => {
  const ui = createAppUi({ env: { STRATA_UI_MODE: 'browser' } });
  const f = fixture({ appUpdater: ui.updater });
  try {
    const job = await run(f);
    assert.equal(job.state, 'done', job.error);
    assert.equal(job.summary, `${UPDATED} ${BROWSER_UPDATE_MESSAGE}`);
  } finally { await f.native.close(); f.cleanup(); }
});

test('an app update that was found and then fails to install is still a real failure (the person asked for it)', async () => {
  const f = fixture({ appUpdater: { check: async () => ({ available: true, version: '0.2.17' }), install: async () => { throw new Error('The app update failed.'); } } });
  try {
    const job = await run(f);
    assert.equal(job.state, 'error');
  } finally { await f.native.close(); f.cleanup(); }
});
