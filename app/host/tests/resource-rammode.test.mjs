/**
 * 0.2.17 RAMMODE (owner 11-19): setup writes the RAM budget of the new rule (the RAM less 17 GB, 47 on 64 GB) in "args" and, beside it, the number of the old rule (the RAM less
 * 24 GB, 40) as "ui_budget_work_gib". "Leave 20% for my work" starts the engine with the second number; the swap is made on the started copy only and is never saved, so the run
 * config keeps setup's number for `strata launch`, `strata serve`, Docker and "Full performance". The app holds no copy of the rule. Nothing here starts an engine; Control is a
 * fake that records the run file the host saved and asked it to start (`f.saved`), and `f.read()` is the run config on disk (what a Terminal start reads).
 */
import './helpers/isolate-profile.mjs';   // a test run never writes the owner's real Strata profile
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createNativeControl } from '../server/native-control.mjs';
import { resourceSettings, launchBudgetArgs } from '../server/native-resources.mjs';

const pc64 = { cpu: { cores: 12 }, ram_gb: 63, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5070', vram_gb: 11.9 }] };
const udQ4 = { arenaGb: 77, engineVersion: '0.1.57' };
const BASE = ['--pack', '/p', '--expert-cache', 'auto', '--spec', '4', '--max-context', '40000', '--kv', 'int8'];
const flag = (args, name) => args[args.indexOf(name) + 1];
const budgetOf = args => flag(args, '--resident-budget-gib');
const withBudget = n => [...BASE, '--resident-budget-gib', String(n)];
/** A 0.2.17 config: setup's 47 and the work number 40. */
const fresh = (extra = {}) => ({ args: withBudget(47), args_written: withBudget(47), ui_budget_work_gib: 40, ...extra });
/** A 0.2.16 config: setup's old 40, no second number. */
const old0216 = (extra = {}) => ({ args: withBudget(40), args_written: withBudget(40), ...extra });

function fixture(config = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-rammode-test-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true });
  process.env.STRATA_INSTALL_DIR = install;
  const path = join(install, 'strata-q2_0.json');
  writeFileSync(path, JSON.stringify({ exe: 'component:strata-engine', tokenizer: join(install, 'tok'), args: BASE, model_name: 'Qwen q2_0', port: 8080, ...config }));
  const saved = [];
  const control = {
    control: { dir: join(root, 'control') }, version: '0.2.0',
    async installedModels() { return { install_dir: install, models: [{ id: 'q2_0', name: 'Qwen q2_0', source_config: path, component_status: 'present', context_tokens: 40000 }] }; },
    async engine() { return { usable: true, backend: 'cuda', components_dir: 'E:\\c', engine: { version: '0.2.0', platform: 'windows-x86_64', backend: 'cuda', exe: 'E:\\e\\strata.exe', root: 'E:\\e' } }; },
    async saveProfile(name, file) { saved.push(JSON.parse(readFileSync(file, 'utf8'))); return {}; },
    async start() { return { ok: true, state: 'ready' }; },
    async stop() { return { ok: true }; },
  };
  const native = createNativeControl({ control, hardwareProbe: () => [{ index: 0, vram_gb: 12 }], directory: data, spawnProcess: () => { throw new Error('no process'); }, fetcher: async () => Response.json({ status: 'ok' }), requests: () => [] });
  return { native, saved, path, read: () => JSON.parse(readFileSync(path, 'utf8')), cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
const finish = async (native, id) => { for (let i = 0; i < 500; i++) { const job = native.state().jobs.find(j => j.id === id); if (job.state !== 'running') return job; await new Promise(r => setTimeout(r, 10)); } throw new Error('job did not finish'); };
/** One load; the run config the engine is started with. */
const load = async (f, profile, args = {}) => { const job = await finish(f.native, f.native.operation('load', { model: 'q2_0', vision: 'no', resourceProfile: profile, ...args }).id); assert.equal(job.state, 'done', job.error); return f.saved.at(-1); };
const withFixture = async (config, body) => { const f = fixture(config); try { await f.native.refresh(true); await body(f); } finally { await f.native.close(); f.cleanup(); } };

test('a work-mode load starts the engine with the old number and saves setup\'s: the run config a Terminal start reads still has 47', async () => {
  await withFixture(fresh(), async f => {
    const run = await load(f, 'work');
    assert.equal(budgetOf(run.args), '40', 'the start itself');
    const file = f.read();
    assert.equal(budgetOf(file.args), '47', 'the saved args: setup\'s own');
    assert.equal(budgetOf(file.args_written), '47');
    assert.equal(file.ui_budget_work_gib, 40);
    assert.equal(file.ui_resource_profile, 'work');
    assert.equal('setupBudget' in (file.ui_resource_baseline ?? {}), false, 'no swap was saved, so nothing to put back');
    assert.equal(file.args.filter(a => a === '--resident-budget-gib').length, 1);
    assert.equal(file.user_args, undefined, 'the swap is not taken for the person\'s');
    for (let i = 0; i < 2; i++) { assert.equal(budgetOf((await load(f, 'work')).args), '40', `work load ${i + 2}`); assert.equal(budgetOf(f.read().args), '47'); }
  });
});

test('full performance starts with setup\'s number and leaves args as they are, before and after a work load', async () => {
  await withFixture(fresh(), async f => {
    assert.equal(budgetOf((await load(f, 'performance')).args), '47');
    assert.equal(budgetOf(f.read().args), '47');
    assert.equal(budgetOf((await load(f, 'work')).args), '40');
    assert.equal(budgetOf((await load(f, 'performance')).args), '47');
    assert.equal(budgetOf(f.read().args), '47');
    assert.equal(budgetOf((await load(f, 'work')).args), '40');
  });
});

test('a load with no resource preference starts with setup\'s number too', async () => {
  await withFixture(fresh(), async f => {
    assert.equal(budgetOf((await load(f, undefined)).args), '47');
    assert.equal(budgetOf(f.read().args), '47');
  });
});

test('an existing 0.2.16 config (setup\'s 40, no second number) is left as it is in both modes; moved (47 and 40) it follows the new rule', async () => {
  await withFixture(old0216(), async f => {
    for (const profile of ['work', 'performance', 'work']) { assert.equal(budgetOf((await load(f, profile)).args), '40', profile); assert.equal(budgetOf(f.read().args), '40'); }
  });
  await withFixture(fresh(), async f => {
    assert.equal(budgetOf((await load(f, 'work')).args), '40');
    assert.equal(budgetOf((await load(f, 'performance')).args), '47');
  });
});

test('the person\'s own budget wins in both modes: user_args, a hand edit of args, a budget typed over setup\'s', async () => {
  const cases = [['user_args', fresh({ args: withBudget(36), user_args: ['--resident-budget-gib', '36'] }), '36'],
                 ['edited args', fresh({ args: withBudget(33) }), '33'],
                 ['typed 40 over 47', fresh({ args: withBudget(40) }), '40']];
  for (const [label, config, expect] of cases) {
    await withFixture(config, async f => {
      for (const profile of ['work', 'performance']) {
        assert.equal(budgetOf((await load(f, profile)).args), expect, `${label}, ${profile}`);
        assert.equal(budgetOf(f.read().args), expect, `${label}, ${profile}: the file`);
      }
    });
  }
});

test('a --ram-keep-gib, STRATA_RAM_KEEP_GIB or STRATA_RESIDENT_HEADROOM_GIB of the person\'s: no number of ours in work mode', async () => {
  for (const config of [fresh({ env: { STRATA_RAM_KEEP_GIB: '12' } }), fresh({ env: { STRATA_RESIDENT_HEADROOM_GIB: '20' } }),
                        fresh({ args: [...withBudget(47), '--ram-keep-gib', '10'], user_args: ['--ram-keep-gib', '10'] })]) {
    await withFixture(config, async f => {
      assert.equal(budgetOf((await load(f, 'work')).args), '47', JSON.stringify(config.env ?? config.user_args));
      assert.equal(budgetOf(f.read().args), '47');
    });
  }
});

test('no second number (a budget the person asked setup for, every expert in RAM): the budget is left alone in both modes', async () => {
  await withFixture({ args: withBudget(52), args_written: withBudget(52) }, async f => {
    for (const profile of ['work', 'performance']) assert.equal(budgetOf((await load(f, profile)).args), '52', profile);
  });
  await withFixture({ args: BASE, args_written: BASE, ui_budget_work_gib: 40 }, async f => {
    for (const profile of ['work', 'performance']) assert.equal((await load(f, profile)).args.includes('--resident-budget-gib'), false, `${profile}: a stale second number adds no budget`);
  });
});

test('launchBudgetArgs: the swap is made on the copy it is given, never on the saved list', () => {
  const cfg = fresh();
  const work = launchBudgetArgs(cfg, 'work');
  assert.equal(budgetOf(work), '40');
  assert.equal(budgetOf(cfg.args), '47', 'the input is not changed');
  assert.equal(launchBudgetArgs(cfg, 'performance'), cfg.args);
  assert.equal(launchBudgetArgs(cfg, undefined), cfg.args);
  assert.equal(budgetOf(launchBudgetArgs(old0216(), 'work')), '40');
  assert.equal(budgetOf(launchBudgetArgs(fresh({ ui_budget_work_gib: 8, args: withBudget(15), args_written: withBudget(15) }), 'work')), '8', '32 GB');
});

test('resourceSettings, which every write of the run config takes, never touches the budget or the baseline for it', () => {
  for (const profile of ['work', 'performance']) {
    const r = resourceSettings(fresh(), pc64, profile, udQ4);
    assert.equal(budgetOf(r.args), '47', profile);
    assert.equal('setupBudget' in r.ui_resource_baseline, false, profile);
    assert.equal(r.env, undefined, `${profile}: with a budget in the args the app adds no env of its own, as before`);
  }
  assert.equal(budgetOf(resourceSettings(old0216(), pc64, 'work', udQ4).args), '40');
});

test('the other rules are unchanged: an older engine and no RAM mode still gets the 0.2.9 rule, a new engine the keep setting', () => {
  const plain = { gpu: 0, args: [...BASE] };
  assert.equal(flag(resourceSettings(plain, pc64, 'work', { arenaGb: 50.3, engineVersion: '0.1.55' }).args, '--resident-budget-gib'), '40');
  assert.equal(resourceSettings(plain, pc64, 'work', { arenaGb: 50.3, engineVersion: '0.1.57' }).env.STRATA_RAM_KEEP_GIB, '8');
});
