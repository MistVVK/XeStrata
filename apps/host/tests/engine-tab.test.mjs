import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync as read } from 'node:fs';
import { build } from 'esbuild';
import { listEngines, changeEngine } from '../server/engine-library.mjs';
import { plainEngineError } from '../server/update-messages.mjs';
import { createControl } from '../server/control-runtime.mjs';
import { checkComponentUpdates } from '../server/updates.mjs';

const src = name => read(new URL(`../${name}`, import.meta.url), 'utf8');
const json = name => JSON.parse(src(`src/locales/${name}.json`));
const built = await build({ stdin: { contents: "export * from './src/lib/engine-list';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { versionLess, engineSize, engineMake, modelsNeedingNewer } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

const listing = { ok: true, families: [{ family: 'qwen', title: 'Qwen and other models', engines: [{ version: '0.1.55', folder: 'a', active: true }, { version: '0.1.54', folder: 'b' }] }], total_bytes: 2_000_000_000, serving: false };

test('the list: no Control (a source checkout) says so; otherwise it is Control\'s list; a failure is one plain sentence with the raw reason behind it', async () => {
  assert.deepEqual((await listEngines(null)).body, { available: false, families: [], total_bytes: 0 });
  const ok = await listEngines({ engines: async () => listing });
  assert.equal(ok.status, 200);
  assert.equal(ok.body.available, true);
  assert.equal(ok.body.families[0].engines.length, 2);
  assert.equal(ok.body.total_bytes, 2_000_000_000);
  const bad = await listEngines({ engines: async () => { throw new Error('exit 2: traceback'); } });
  assert.equal(bad.status, 500);
  assert.equal(bad.body.error, "The installed engines couldn't be read.");
  assert.match(bad.body.details, /traceback/);
});

test('switch: asks Control for the chosen engine only, and answers at once with one next step while a model is loaded', async () => {
  const calls = [];
  const control = { useEngine: async a => { calls.push(['use', a]); return { status: 'switched', to: '0.1.54' }; }, removeEngine: async a => { calls.push(['remove', a]); return { status: 'deleted' }; } };
  const busy = await changeEngine(control, 'use', { family: 'qwen', folder: 'b' }, { modelsRunning: () => true });
  assert.equal(busy.status, 409);
  assert.match(busy.body.error, /A model is running, so nothing was changed\. Unload the model in Models/);
  assert.deepEqual(calls, [], 'nothing was asked of Control');
  const ok = await changeEngine(control, 'use', { family: 'qwen', folder: 'b' }, { modelsRunning: () => false });
  assert.equal(ok.status, 200);
  assert.deepEqual(calls, [['use', { family: 'qwen', folder: 'b' }]]);
  const deleted = await changeEngine(control, 'remove', { family: 'qwen', folder: 'c' }, { modelsRunning: () => true });
  assert.equal(deleted.status, 200, 'deleting an engine nothing uses is not stopped by a loaded model; Control refuses the one in use');
  assert.deepEqual(calls.at(-1), ['remove', { family: 'qwen', folder: 'c' }]);
});

test('only a family and a plain folder name from the list are accepted: a path, a hidden folder or another family is refused before Control is asked', async () => {
  const control = { useEngine: async () => assert.fail('must not be called'), removeEngine: async () => assert.fail('must not be called') };
  for (const body of [{ family: 'x', folder: 'a' }, { family: 'deepseek', folder: 'a' }, { family: 'qwen', folder: '../a' }, { family: 'qwen', folder: 'a/b' }, { family: 'qwen', folder: 'a\\b' }, { family: 'qwen', folder: '.rejected' }, { family: 'qwen', folder: '--force' }, { family: 'qwen', folder: '-x' }, { family: 'qwen', folder: 'a b' }, { family: 'qwen' }, {}]) {
    for (const action of ['use', 'remove']) assert.equal((await changeEngine(control, action, body)).status, 400, JSON.stringify(body));
  }
  assert.equal((await changeEngine({}, 'use', { family: 'qwen', folder: 'a' })).status, 501);
});

test('the engine in use cannot be deleted: Control\'s refusal reaches the person in plain words, the raw text behind Details', async () => {
  const refusing = { removeEngine: async () => { throw Object.assign(new Error('This is the engine in use. Switch to another one first.'), { exitCode: 2 }); }, useEngine: async () => ({}) };
  const out = await changeEngine(refusing, 'remove', { family: 'qwen', folder: 'a' });
  assert.equal(out.status, 400);
  assert.equal(out.body.error, 'This is the engine in use. Switch to another one first, then delete it.');
  assert.match(out.body.details, /Switch to another one first/);
  const serving = await changeEngine({ useEngine: async () => { throw Object.assign(new Error('A model is running, so the engine was not switched.'), { exitCode: 7 }); } }, 'use', { family: 'qwen', folder: 'a' });
  assert.equal(serving.status, 409);
  assert.match(serving.body.error, /^A model is running, so nothing was changed/);
});

test('plain engine answers: nothing says exit codes, folders or hashes outside the details', () => {
  const cases = [
    [{ message: 'x', exitCode: 7 }, 'switch', /A model is running/],
    [{ message: 'a running program is using it', exitCode: 2 }, 'delete', /still using this engine/],
    [{ message: 'that engine is not installed (it may have been deleted already)', exitCode: 2 }, 'delete', /already gone/],
    [{ message: 'engine 0.1.1 can no longer be used: sig', exitCode: 5 }, 'switch', /can't run on this computer/],
    [{ message: 'something odd' }, 'switch', /wasn't switched\. Nothing was changed/],
    [{ message: 'something odd' }, 'delete', /wasn't deleted/],
  ];
  for (const [error, action, expected] of cases) {
    const plain = plainEngineError(error, action);
    assert.match(plain.message, expected);
    assert.doesNotMatch(plain.message, /exit|SHA|\.json|[A-Z]:\\|\/home\//i);
  }
});

test('createControl asks the same commands the Terminal runs, and none of them acquires', async () => {
  const seen = [];
  const control = createControl({ control: { dir: '/c', python: '/p', cli: '/c/strata.py' }, env: {}, run: async (args, options) => { seen.push([args.join(' '), !!options.acquire]); return { code: 0, json: { ok: true, families: [] } }; } });
  await control.engines();
  await control.useEngine({ family: 'qwen', folder: '0.1.54-linux-x86_64-cuda-cuda13' });
  await control.removeEngine({ family: 'qwen', folder: '0.1.50-linux-x86_64-cuda-cuda13' });
  assert.deepEqual(seen, [
    ['components engines --output json', false],
    ['components use 0.1.54-linux-x86_64-cuda-cuda13 --family qwen --output json', false],
    ['components remove 0.1.50-linux-x86_64-cuda-cuda13 --family qwen --output json', false],
  ]);
});

test('About: an engine already on this computer is offered as an update that switches to it, with nothing to download', async () => {
  const control = { version: '0.1.55', updateCheck: async () => ({ feed: {}, control: { status: 'current', installed: '0.1.55' }, app: { status: 'current' },
    engine: { status: 'available', installed: { version: '0.1.54' }, offered: { version: '0.1.55' }, download_bytes: 0, installed_locally: '0.1.55-linux-x86_64-cuda-cuda13' } }) };
  const result = await checkComponentUpdates({ control, appVersion: '0.2.9' });
  assert.equal(result.engine.status, 'available');
  assert.equal(result.engine.canInstall, true);
  assert.match(result.engine.message, /already on this computer.*nothing is downloaded/);
  assert.equal(result.engine.installedVersion, '0.1.54');
});

test('engine helpers: sizes, makes and version order', () => {
  assert.equal(engineSize(1_020_000_000, 'en'), '1.0 GB');
  assert.equal(engineSize(640_000_000, 'en'), '640 MB');
  assert.equal(engineSize(10, 'en'), '1 MB');
  assert.equal(engineMake('cuda'), 'NVIDIA');
  assert.equal(engineMake('hip'), 'AMD');
  assert.equal(engineMake('sycl'), 'Intel');
  assert.equal(versionLess('0.1.9', '0.1.10'), true);
  assert.equal(versionLess('0.1.49', '0.1.49.1'), true);
  assert.equal(versionLess('0.1.55', '0.1.55'), false);
  assert.equal(versionLess('0.1.56', '0.1.55.1'), false);
});

test('a switch to an older engine warns about the installed models that need a newer one, and does not refuse', () => {
  const native = { installed: { models: [{ model: 'gyro', model_name: 'Gyro' }, { model: 'plain', model_name: 'Plain' }] },
    catalog: { families: [{ family: 'qwen', title: 'Qwen', sizes: [{ id: 'gyro', model: 'gyro', engine_min: '0.1.45.1' }, { id: 'plain', model: 'plain', engine_min: null }] }] } };
  assert.deepEqual(modelsNeedingNewer(native, { family: 'qwen', version: '0.1.45' }), ['Gyro']);
  assert.deepEqual(modelsNeedingNewer(native, { family: 'qwen', version: '0.1.46' }), []);
  assert.deepEqual(modelsNeedingNewer(null, { family: 'qwen', version: '0.1.1' }), []);
  const page = src('src/components/SettingsEngine.tsx');
  assert.match(page, /disabled=\{pending \|\| running\}/, 'only a loaded model stops the button; the warning does not');
});

test('the Engine tab lists and switches, and has no update; About has the update and no rollback', () => {
  const engine = src('src/components/SettingsEngine.tsx');
  assert.match(src('src/components/NativeOperations.tsx'), /<SettingsEngine \/>/, 'the Engine tab shows the list');
  assert.match(engine, /Installed engines/);
  assert.doesNotMatch(engine, /Check for updates|update-engine|fetchUpdateCheck/);
  const ops = src('src/components/NativeOperations.tsx');
  assert.doesNotMatch(ops, /Update component|Check for updates|Update engine\?|fetchUpdateCheck|update-engine/);
  assert.match(ops, /View logs/, 'the logs stay on the Engine tab');
  const about = src('src/components/SettingsUpdates.tsx');
  assert.doesNotMatch(about, /Roll back|rollback|Restore previous engine|Previous engine available/i);
  assert.match(about, /Check for updates/);
  assert.match(about, /'update-engine'/);
  assert.match(about, /'update-strata'/);
});

test('the engine in use has no Delete button; every other row has Use this (when it can run here) and Delete', () => {
  const page = src('src/components/SettingsEngine.tsx');
  assert.match(page, /engine\.active \? <span className="engine-row__state">\{tr\("In use"\)\}<\/span> : <>/);
  assert.match(page, /engine\.can_use \? <Btn[\s\S]*?\{tr\("Use this"\)\}/);
  assert.match(page, /disabled=\{busy \|\| !engine\.can_delete\}/);
  assert.match(page, /run\('remove', del\)/);
});

test('the engine catalogs: same keys and placeholders in nine languages, none still English, none in another catalog, and every sentence of the host is in them', () => {
  const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
  const en = json('engine-en');
  const slots = text => [...text.matchAll(/\{\d+\}/g)].map(m => m[0]).sort();
  const others = new Set();
  for (const name of ['en', 'tune-en', 'tools-en', 'fix-en', 'gh-en', 'ui-en']) for (const v of Object.values(json(name))) others.add(v);
  for (const l of languages) {
    const catalog = json(`engine-${l}`);
    assert.deepEqual(Object.keys(catalog), Object.keys(en), l);
    for (const [id, source] of Object.entries(en)) {
      assert.ok(catalog[id].trim(), `${l}:${id}`);
      assert.deepEqual(slots(catalog[id]), slots(source), `${l}:${id}`);
      if (l !== 'en') assert.notEqual(catalog[id], source, `${l}:${id} is still English`);
    }
  }
  for (const [id, text] of Object.entries(en)) assert.ok(!others.has(text), `already in another catalog: ${id} ${text}`);
  const known = new Set(Object.values(en));
  for (const sentence of [
    'A model is running, so nothing was changed. Unload the model in Models, then try again.',
    'This is the engine in use. Switch to another one first, then delete it.',
    'A program is still using this engine, so it was not deleted. Unload your models and try again.',
    'That engine is already gone. The list has been refreshed.',
    'An update is running. Wait for it to finish, then try again.',
    "This engine can't run on this computer, so nothing was changed.",
    "The engine wasn't deleted. Nothing else was changed.",
    "The engine wasn't switched. Nothing was changed.",
    'Choose an engine from the list.',
    'Engines can be switched and deleted in the installed Strata app.',
    'A newer engine is already on this computer. Update switches to it; nothing is downloaded.',
  ]) {
    assert.ok(known.has(sentence), `not in the engine catalog: ${sentence}`);
    assert.ok([src('server/engine-library.mjs'), src('server/update-messages.mjs'), src('server/updates.mjs')].some(file => file.includes(sentence)), `no longer said by the host: ${sentence}`);
  }
});
