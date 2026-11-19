import test from 'node:test';
import assert from 'node:assert/strict';
import { resourceSettings } from '../server/native-resources.mjs';

/* 0.2.17 RAMMODE (owner 11-19): setup writes the RAM budget of the new rule (the RAM less 17 GB, 47 on 64 GB) in "args" and, beside it, the number of the old rule (the RAM less
   24 GB, 40) as "ui_budget_work_gib". "Leave 20% for my work" swaps the second in and records the first in the baseline; "Full performance" runs with setup's own, and puts it back
   after a work load. The app holds no copy of the rule. */

const pc64 = { cpu: { cores: 12 }, ram_gb: 63, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5070', vram_gb: 11.9 }] };
const udQ4 = { arenaGb: 77, engineVersion: '0.1.57' };
const flag = (args, name) => args[args.indexOf(name) + 1];
const budgetOf = r => flag(r.args, '--resident-budget-gib');
const written = args => ({ args, args_written: [...args] });
const base = ['--native', 'x.gguf', '--max-context', '8192', '--pool-workers', '12'];
/** A 0.2.17 config: setup's 47 and the work number 40. */
const fresh = (extra = {}) => ({ gpu: 0, ...written([...base, '--resident-budget-gib', '47']), ui_budget_work_gib: 40, ...extra });
/** A 0.2.16 config: setup's old 40, no second number. */
const old0216 = (extra = {}) => ({ gpu: 0, ...written([...base, '--resident-budget-gib', '40']), ...extra });
const asSaved = (config, r, profile) => ({ ...config, args: r.args, args_written: [...r.args], env: r.env, ui_resource_profile: profile, ui_resource_baseline: r.ui_resource_baseline });

test('a config from setup 0.2.17: work gets the old number, full performance setup\'s own', () => {
  const work = resourceSettings(fresh(), pc64, 'work', udQ4);
  assert.equal(budgetOf(work), '40');
  assert.equal(work.ui_resource_baseline.setupBudget, 47, 'what it replaced is kept to put back');
  assert.equal(work.env, undefined, 'a RAM mode is in the args: the app adds no env of its own (as before)');
  const perf = resourceSettings(fresh(), pc64, 'performance', udQ4);
  assert.equal(budgetOf(perf), '47');
  assert.equal(perf.ui_resource_baseline.setupBudget, undefined);
});

test('every RAM size setup writes both numbers for (32, 48, 64 GB): work takes the second, performance the first', () => {
  for (const [ram, newer, older] of [[32, 15, 8], [48, 31, 24], [64, 47, 40]]) {
    const pc = { ...pc64, ram_gb: ram - 0.3 };
    const config = { gpu: 0, ...written([...base, '--resident-budget-gib', String(newer)]), ui_budget_work_gib: older };
    assert.equal(budgetOf(resourceSettings(config, pc, 'work', udQ4)), String(older), `${ram} GB work`);
    assert.equal(budgetOf(resourceSettings(config, pc, 'performance', udQ4)), String(newer), `${ram} GB performance`);
  }
});

test('work then performance puts setup\'s number back, work again swaps again, and a second work load changes nothing', () => {
  const cfg = fresh();
  const work = resourceSettings(cfg, pc64, 'work', udQ4);
  const savedWork = asSaved(cfg, work, 'work');
  const again = resourceSettings(savedWork, pc64, 'work', udQ4);
  assert.equal(budgetOf(again), '40');
  assert.equal(again.ui_resource_baseline.setupBudget, 47, 'the number it replaced is not lost on a second work load');
  const back = resourceSettings(savedWork, pc64, 'performance', udQ4);
  assert.equal(budgetOf(back), '47');
  assert.equal(back.ui_resource_baseline.setupBudget, undefined);
  const backAndForth = resourceSettings(asSaved(cfg, back, 'performance'), pc64, 'work', udQ4);
  assert.equal(budgetOf(backAndForth), '40');
  assert.equal(backAndForth.args.filter(a => a === '--resident-budget-gib').length, 1);
});

test('an existing 0.2.16 config (setup\'s 40, no second number) is left as it is in both modes (the update moves it, setup.migrate_ram_budget)', () => {
  for (const profile of ['work', 'performance']) {
    const r = resourceSettings(old0216(), pc64, profile, udQ4);
    assert.equal(budgetOf(r), '40', profile);
    assert.equal(r.ui_resource_baseline.setupBudget, undefined, profile);
    assert.equal(r.env, undefined, profile);
  }
  // after the move: 47 in args, 40 beside it: work keeps 40, performance has 47
  const moved = { ...old0216(), ...written([...base, '--resident-budget-gib', '47']), ui_budget_work_gib: 40 };
  assert.equal(budgetOf(resourceSettings(moved, pc64, 'work', udQ4)), '40');
  assert.equal(budgetOf(resourceSettings(moved, pc64, 'performance', udQ4)), '47');
});

test('the person\'s own budget wins in both modes: user_args, a hand edit of args, an edit after the app wrote its number', () => {
  const inUser = fresh({ user_args: ['--resident-budget-gib', '36'], args: [...base, '--resident-budget-gib', '36'] });
  const edited = { gpu: 0, args: [...base, '--resident-budget-gib', '33'], args_written: [...base, '--resident-budget-gib', '47'], ui_budget_work_gib: 40 };
  const sameAsWork = { gpu: 0, args: [...base, '--resident-budget-gib', '40'], args_written: [...base, '--resident-budget-gib', '47'], ui_budget_work_gib: 40 };   // typed by hand
  for (const [label, cfg, expect] of [['user_args', inUser, '36'], ['edited args', edited, '33'], ['typed 40 over setup\'s 47', sameAsWork, '40']]) {
    for (const profile of ['work', 'performance']) {
      const r = resourceSettings(cfg, pc64, profile, udQ4);
      assert.equal(budgetOf(r), expect, `${label}, ${profile}`);
      assert.equal(r.ui_resource_baseline.setupBudget, undefined, `${label}, ${profile}: nothing to put back`);
    }
  }
  // the app swapped 40 in, then the person set 30: performance does not put 47 over it
  const work = resourceSettings(fresh(), pc64, 'work', udQ4);
  const later = { ...asSaved(fresh(), work, 'work'), args: [...work.args.slice(0, work.args.indexOf('--resident-budget-gib') + 1), '30'] };
  assert.equal(budgetOf(resourceSettings(later, pc64, 'performance', udQ4)), '30');
  assert.equal(budgetOf(resourceSettings(later, pc64, 'work', udQ4)), '30');
});

test('a --ram-keep-gib or a RAM mode of the person\'s, and STRATA_RAM_KEEP_GIB / STRATA_RESIDENT_HEADROOM_GIB of theirs in the env: no number swapped in, nothing added', () => {
  for (const env of [{ STRATA_RAM_KEEP_GIB: '12' }, { STRATA_RESIDENT_HEADROOM_GIB: '20' }]) {
    const work = resourceSettings(fresh({ env }), pc64, 'work', udQ4);
    assert.equal(budgetOf(work), '47', JSON.stringify(env) + ': their RAM setting says how much to leave; setup\'s number stays');
    assert.deepEqual(work.env ?? env, env);
    assert.equal(budgetOf(resourceSettings(fresh({ env }), pc64, 'performance', udQ4)), '47');
  }
  const keepFlag = fresh({ user_args: ['--ram-keep-gib', '10'], args: [...base, '--resident-budget-gib', '47', '--ram-keep-gib', '10'] });
  assert.equal(budgetOf(resourceSettings(keepFlag, pc64, 'work', udQ4)), '47', 'a --ram-keep-gib of the person\'s says how much to leave: setup\'s number stays');
  assert.equal(flag(resourceSettings(keepFlag, pc64, 'work', udQ4).args, '--ram-keep-gib'), '10');
  // the env setting the app itself wrote (the 0.2.9 headroom) is not the person's
  const appEnv = fresh({ env: { STRATA_RESIDENT_HEADROOM_GIB: '13' }, ui_resource_baseline: { workers: 12, reserve: 0, ramHeadroom: true } });
  assert.equal(budgetOf(resourceSettings(appEnv, pc64, 'work', udQ4)), '40');
});

test('no second number (a budget the person asked setup for, a unified-memory PC, a layer split, every expert in RAM): the budget is left alone in both modes', () => {
  const none = { gpu: 0, ...written([...base, '--resident-budget-gib', '52']) };
  for (const profile of ['work', 'performance']) {
    assert.equal(budgetOf(resourceSettings(none, pc64, profile, udQ4)), '52', profile);
    const noBudget = resourceSettings({ gpu: 0, ...written(base), ui_budget_work_gib: 40 }, pc64, profile, udQ4);
    assert.equal(noBudget.args.includes('--resident-budget-gib'), false, `${profile}: a stale second number adds no budget`);
  }
  const uma = { cpu: { cores: 8 }, ram_gb: 63, gpus: [{ index: 0, name: 'AMD Radeon 8060S', vram_gb: 112, uma: true }] };
  const halo = { gpu: 0, ...written([...base, '--resident-budget-gib', '40']) };      // unified memory: 24 in both rules, setup writes no second number
  for (const profile of ['work', 'performance']) assert.equal(budgetOf(resourceSettings(halo, uma, profile, udQ4)), '40', `unified memory, ${profile}`);
});

test('the other rules are unchanged: an older engine and no RAM mode still gets the 0.2.9 rule, a new engine the keep setting', () => {
  const plain = { gpu: 0, args: [...base] };
  const oldEngine = resourceSettings(plain, pc64, 'work', { arenaGb: 50.3, engineVersion: '0.1.55' });
  assert.equal(flag(oldEngine.args, '--resident-budget-gib'), '40');
  assert.equal(oldEngine.ui_resource_baseline.setupBudget, undefined);
  assert.equal(resourceSettings(plain, pc64, 'work', { arenaGb: 50.3, engineVersion: '0.1.57' }).env.STRATA_RAM_KEEP_GIB, '8');
});
