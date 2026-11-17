import './helpers/isolate-profile.mjs';   // a test run never writes the owner's real Strata profile
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync as read, mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createControl } from '../server/control-runtime.mjs';
import { createNativeControl } from '../server/native-control.mjs';

const src = name => read(new URL(`../${name}`, import.meta.url), 'utf8');

/** A packaged host with one installed model and a fake Control whose `update` plays the events of `strata update --stop-model --output jsonl`. */
function fixture({ updateScript } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-update-model-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;
  const config = join(install, 'strata-iq2_xs.json');
  writeFileSync(config, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: ['--max-context', '32768'], model_name: 'Qwen', port: 8080 }));
  const calls = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.1.45',
    async installedModels() { return { install_dir: install, models: [{ id: 'iq2_xs', name: 'Qwen', source_config: config, component_status: 'present', context_tokens: 32768, backend: 'cuda' }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: 'E:\\c', previous: null, engine: { version: '0.1.45', platform: 'windows-x86_64', backend: 'cuda', exe: 'e', root: 'r' } }; },
    async saveProfile(profile) { calls.push(['saveProfile', profile]); return {}; },
    async start(profile) { calls.push(['start', profile]); return { ok: true, state: 'ready' }; },
    async stop(profile) { calls.push(['stop', profile]); return { ok: true }; },
    async update(options) {
      calls.push(['update', options.engine, options.stopModel === true]);
      const ids = native.state().instances.map(i => i.id);
      const profile = 'app-1';
      options.onEvent?.({ event: 'models', phase: 'consent', models: [{ profile, name: 'Qwen', port: 8080, kind: 'profile', state: 'ready' }], message: 'Qwen is running' });
      if (updateScript) return updateScript(options, { profile, ids, native });
      options.onEvent?.({ event: 'progress', phase: 'download', label: 'engine 0.1.46', percent: 40 });
      options.onEvent?.({ event: 'models', phase: 'stopping', models: [{ profile, name: 'Qwen', port: 8080, kind: 'profile', state: 'ready' }] });
      options.onEvent?.({ event: 'models', phase: 'stopped', models: [{ profile, name: 'Qwen', port: 8080, kind: 'profile', state: 'stopped' }] });
      options.onEvent?.({ event: 'models', phase: 'left-stopped', models: [{ profile, name: 'Qwen', port: 8080, kind: 'profile', state: 'stopped' }], start: 'strata launch' });
      return { ok: true, results: [{ component: 'engine', status: 'updated', installed_after: { version: '0.1.46' } }] };
    },
    async rollbackEngine() { return { ok: true }; },
  };
  const native = createNativeControl({ control, directory: data, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({}), requests: () => [] });
  return { native, calls, control, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 400; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };
async function loaded(f) {
  await f.native.refresh(true);
  const load = await finish(f.native, f.native.operation('load', { model: 'iq2_xs' }).id);
  assert.equal(load.state, 'done', load.error);
  assert.equal(f.native.state().instances.length, 1);
}

test('Control is called with --stop-model only when the app asks for it', async () => {
  const seen = [];
  const control = createControl({ control: { dir: '/c', python: '/c/python', cli: '/c/strata.py' }, env: {}, run: async args => { seen.push(args); return { code: 0, json: { ok: true, results: [] } }; } });
  await control.update({ engine: true, control: false });
  await control.update({ engine: true, control: false, stopModel: true });
  assert.ok(!seen[0].includes('--stop-model'));
  assert.ok(seen[1].includes('--stop-model'));
  assert.deepEqual(seen[1].slice(0, 3), ['update', '--output', 'jsonl'], 'the host reads the events (jsonl)');
  assert.ok(!seen[1].includes('--start-again'), 'starting again is the person\'s answer in the dialog, not an automatic flag');
});

test('a loaded model is stopped for the update only with the person\'s yes, and is then offered a start with the same settings', async () => {
  const f = fixture();
  try {
    await loaded(f);
    const refused = await finish(f.native, f.native.operation('update-engine', {}).id);
    assert.equal(refused.state, 'error');
    assert.match(refused.error, /Unload native models/, 'no yes, no stop');
    assert.equal(f.calls.filter(c => c[0] === 'update').length, 0);

    const job = await finish(f.native, f.native.operation('update-engine', { stopModel: true }).id);
    assert.equal(job.state, 'done', job.error);
    assert.deepEqual(f.calls.find(c => c[0] === 'update'), ['update', true, true]);
    const state = f.native.state();
    assert.equal(state.instances.length, 0, 'the stopped model has left the list');
    assert.equal(state.stoppedForUpdate.updated, true);
    assert.deepEqual(state.stoppedForUpdate.models.map(m => m.label), [state.stoppedForUpdate.models[0].label]);
    assert.equal(job.step, 'stopped');
    assert.equal(f.calls.filter(c => c[0] === 'start').length, 1, 'only the first load started it; nothing started it again by itself');

    const again = await finish(f.native, f.native.operation('restart-models', {}).id);
    assert.equal(again.state, 'done', again.error);
    assert.match(again.summary, /running again with the same settings/);
    const starts = f.calls.filter(c => c[0] === 'start');
    assert.equal(starts.length, 2);
    assert.equal(starts[1][1], starts[0][1], 'the same profile (its saved run settings) is started again');
    const after = f.native.state();
    assert.equal(after.instances.length, 1);
    assert.equal(after.instances[0].status, 'ready');
    assert.equal(after.stoppedForUpdate, null);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a failed update leaves the old engine and still offers the model back, on the old engine', async () => {
  const f = fixture({
    updateScript: async (options, { profile }) => {
      options.onEvent?.({ event: 'models', phase: 'stopping', models: [{ profile, name: 'Qwen', port: 8080, kind: 'profile', state: 'ready' }] });
      options.onEvent?.({ event: 'models', phase: 'stopped', models: [{ profile, name: 'Qwen', port: 8080, kind: 'profile', state: 'stopped' }] });
      throw Object.assign(new Error('The new engine did not pass its test start'), { exitCode: 6 });
    },
  });
  try {
    await loaded(f);
    const job = await finish(f.native, f.native.operation('update-engine', { stopModel: true }).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /didn't start correctly, so nothing was changed\. Your current version still works/, 'plain words, not the exit code');
    const state = f.native.state();
    assert.equal(state.instances.length, 0);
    assert.equal(state.stoppedForUpdate.updated, false, 'the dialog says the update did not finish and offers the old engine');
    const again = await finish(f.native, f.native.operation('restart-models', {}).id);
    assert.equal(again.state, 'done', again.error);
    assert.equal(f.native.state().instances[0].status, 'ready');
  } finally { await f.native.close(); f.cleanup(); }
});

test('a failed update that never touched the model offers nothing', async () => {
  const f = fixture({ updateScript: async () => { throw new Error('Could not reach the update server (timed out).'); } });
  try {
    await loaded(f);
    const job = await finish(f.native, f.native.operation('update-engine', { stopModel: true }).id);
    assert.equal(job.state, 'error');
    assert.equal(f.native.state().instances.length, 1, 'the model is still running');
    assert.equal(f.native.state().stoppedForUpdate, null);
  } finally { await f.native.close(); f.cleanup(); }
});

test('a model the app did not start is never stopped: the update ends at the first event, before any download', async () => {
  const f = fixture({
    updateScript: async (options, { profile }) => {
      options.onEvent?.({ event: 'models', phase: 'consent', models: [{ profile: 'launch-qwen', name: 'Qwen (port 8081)', port: 8081, kind: 'launch', state: 'ready' }] });
      if (options.signal.aborted) throw new Error('killed');   // as runCli does: an aborted signal ends the Control process
      await new Promise((resolve, reject) => options.signal.addEventListener('abort', () => reject(new Error('killed')), { once: true }));
    },
  });
  try {
    await loaded(f);
    const stopsBefore = f.calls.filter(c => c[0] === 'stop').length;
    const job = await finish(f.native, f.native.operation('update-engine', { stopModel: true }).id);
    assert.equal(job.state, 'error');
    assert.match(job.error, /not started from this app is running, so nothing was changed/);
    assert.equal(f.native.state().instances.length, 1);
    assert.equal(f.native.state().stoppedForUpdate, null);
    assert.equal(f.calls.filter(c => c[0] === 'stop').length, stopsBefore, 'nothing was stopped by the update');
  } finally { await f.native.close(); f.cleanup(); }
});

test('"Not now" forgets the offer; with nothing stopped there is nothing to start again', async () => {
  const f = fixture();
  try {
    await loaded(f);
    await finish(f.native, f.native.operation('update-engine', { stopModel: true }).id);
    assert.ok(f.native.state().stoppedForUpdate);
    f.native.dismissRestart();
    assert.equal(f.native.state().stoppedForUpdate, null);
    const none = await finish(f.native, f.native.operation('restart-models', {}).id);
    assert.equal(none.state, 'error');
    assert.match(none.error, /no stopped model/);
  } finally { await f.native.close(); f.cleanup(); }
});

test('with no model loaded the update behaves as before: no stop flag, no offer', async () => {
  const f = fixture();
  try {
    await f.native.refresh(true);
    const job = await finish(f.native, f.native.operation('update-engine', {}).id);
    assert.equal(job.state, 'done', job.error);
    assert.deepEqual(f.calls.find(c => c[0] === 'update'), ['update', true, false]);
    assert.equal(f.native.state().stoppedForUpdate, null);
    const withFlag = await finish(f.native, f.native.operation('update-engine', { stopModel: true }).id);
    assert.deepEqual(f.calls.filter(c => c[0] === 'update')[1], ['update', true, false], 'the flag is only sent when a model is loaded');
    assert.equal(withFlag.state, 'done');
  } finally { await f.native.close(); f.cleanup(); }
});

test('the dialog: warns naming the model, progress in plain words, the offer, the old-engine offer, and only Settings > About and the Models screen mount it', () => {
  const dialog = src('src/components/EngineUpdateDialog.tsx');
  assert.match(dialog, /args: \{ stopModel: true \}/);
  assert.match(dialog, /kind: 'restart-models'/);
  assert.match(dialog, /To update the engine, Strata has to stop the model that is loaded: \{0\}\./, 'the warning names the model');
  for (const words of ['Getting the new engine ready. {0} keeps working until it is ready.', 'Stopping {0}. Answers in progress finish first.', '{0} is stopped. Updating the engine…',
    'The engine is updated. Start {0} again with the same settings?', "The update didn't finish. Your old engine is unchanged and still works. Start {0} again on it?"])
    assert.ok(dialog.includes(words), words);
  assert.doesNotMatch(dialog, /fetch\(|api\.get|XMLHttpRequest/, 'no network from the dialog');
  const code = dialog.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
  assert.doesNotMatch(code, /exit code|--stop-model|jsonl/i, 'no flags or codes in the words');
  assert.match(src('src/components/SettingsUpdates.tsx'), /<EngineUpdateDialog open=\{confirm === 'update-engine' && withModel\}/);
  assert.match(src('src/components/SettingsUpdates.tsx'), /<Dialog open=\{!!confirm && !\(confirm === 'update-engine' && withModel\)\}/, 'with no model loaded the older dialog is the one shown');
  assert.equal((src('src/components/NativeModels.tsx').match(/<EngineUpdateDialog /g) ?? []).length, 1);
  assert.match(src('server/index.mjs'), /\/api\/native\/engine-restart\/dismiss/);
});

test('the notice for a newer engine stays an ordinary quiet card: no toast, no banner, no timer, no network', () => {
  const card = src('src/components/EngineUpdateCard.tsx');
  assert.doesNotMatch(card, /toast|setInterval|setTimeout|fetch\(|Notification/);
  assert.doesNotMatch(src('src/components/EngineUpdateDialog.tsx'), /setInterval|setTimeout|Notification|toast/);
});

test('offline, the update says nothing by itself: the dialog opens only from a button and shows only what the host reports', () => {
  const settings = src('src/components/SettingsUpdates.tsx');
  assert.match(settings, /onClick=\{\(\) => setConfirm\(name === 'Strata' \? 'update-strata' : 'update-engine'\)\}/, 'the button opens the confirmation; nothing opens it on a timer or on a failed check');
});

test('all nine languages carry the dialog\'s words with the same slots', () => {
  const langs = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
  const catalogs = Object.fromEntries(langs.map(l => [l, JSON.parse(src(`src/locales/upd-${l}.json`))]));
  const keys = Object.keys(catalogs.en);
  assert.equal(keys.length, 17);
  for (const l of langs) {
    assert.deepEqual(Object.keys(catalogs[l]), keys, l);
    for (const k of keys) {
      assert.ok(catalogs[l][k].trim(), `${l} ${k}`);
      assert.deepEqual([...catalogs[l][k].matchAll(/\{\d+\}/g)].map(m => m[0]).sort(), [...catalogs.en[k].matchAll(/\{\d+\}/g)].map(m => m[0]).sort(), `${l} ${k} slots`);
      if (l !== 'en') assert.notEqual(catalogs[l][k], catalogs.en[k], `${l} ${k} is translated`);
    }
  }
  // English is in the main catalog (src/lib/i18n.ts); each other language is one file (src/lib/catalog-<lang>.ts) that the app fetches when the language is used.
  const i18n = src('src/lib/i18n.ts');
  assert.match(i18n, /\.\.\.upd_en\b/);
  for (const l of langs.filter(x => x !== 'en')) assert.match(src(`src/lib/catalog-${l}.ts`), new RegExp(`from '\\.\\./locales/upd-${l}\\.json'`));
});
