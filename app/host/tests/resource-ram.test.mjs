import test from 'node:test';
import assert from 'node:assert/strict';
import { resourceSettings, saferStart, ramKeepGib, engineKnowsRamKeep, engineVersionOfExe, RAM_KEEP_MIN_ENGINE } from '../server/native-resources.mjs';

const pc64 = { cpu: { cores: 12 }, ram_gb: 63, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5070', vram_gb: 11.9 }] };
const iq3s = { arenaGb: 50.3 };                                   // engine version not known
const iq3sNew = { arenaGb: 50.3, engineVersion: '0.1.56' };       // an engine that knows the setting
const iq3sOld = { arenaGb: 50.3, engineVersion: '0.1.55' };       // the 0.2.9 release engine
const config = { gpu: 0, args: ['--native', 'x.gguf', '--max-context', '32768', '--pool-workers', '12'] };
const flag = (args, name) => args[args.indexOf(name) + 1];
const old279 = (ram, arena = 50.3) => Math.max(8, Math.min(Math.floor(ram * .8) - 10, Math.floor(arena / 1.073741824)));   // the 0.2.9 rule

test('which engines know the setting: RAM_KEEP_MIN_ENGINE and newer; unknown or unreadable versions are treated as older', () => {
  assert.equal(RAM_KEEP_MIN_ENGINE, '0.1.56');
  for (const v of ['0.1.56', '0.1.56.1', '0.1.57', '0.2.0', 'v0.1.56', '1.0']) assert.equal(engineKnowsRamKeep(v), true, v);
  for (const v of ['0.1.52', '0.1.54', '0.1.55', '0.1.55.1', '0.1.55.2', '0.1.5', '0.1', null, undefined, '', 'local', 'dev']) assert.equal(engineKnowsRamKeep(v), false, String(v));
  assert.equal(engineVersionOfExe('C:\\Users\\x\\AppData\\Local\\Strata\\components\\engine\\0.1.54-windows-x86_64-cuda-cuda13\\strata.exe'), '0.1.54');
  assert.equal(engineVersionOfExe('/home/u/.strata/components/engine/0.1.56-linux-x86_64-cuda/strata'), '0.1.56');
  assert.equal(engineVersionOfExe('component:strata-engine'), null);
});

test('an engine that knows the setting: "leave 20% for my work" asks it to keep W GiB of RAM free (env STRATA_RAM_KEEP_GIB), not a budget of 80% of the RAM', () => {
  const work = resourceSettings(config, pc64, 'work', iq3sNew);
  assert.equal(work.env.STRATA_RAM_KEEP_GIB, '8', '63 GB: a fifth is 12.6, capped at 8');
  assert.equal(work.args.includes('--ram-keep-gib'), false, 'in the env, not as a flag');
  assert.equal(work.args.includes('--resident-budget-gib'), false, 'the engine decides from the RAM free at its start, not from the total: no budget from the app');
  assert.equal(work.env.STRATA_RESIDENT_HEADROOM_GIB, undefined, 'and no STRATA_RESIDENT_HEADROOM_GIB of 20%');
  assert.deepEqual([work.ui_resource_baseline.ramKeep, work.ui_resource_baseline.ramBudget, work.ui_resource_baseline.ramHeadroom], [true, false, false]);
});

test('an older engine (0.1.52, 0.1.54, 0.1.55 in the Engine tab) gets the 0.2.9 rule unchanged and never an argument it does not know', () => {
  for (const v of ['0.1.52', '0.1.54', '0.1.55', '0.1.55.1']) {
    const w = resourceSettings(config, pc64, 'work', { arenaGb: 50.3, engineVersion: v });
    assert.equal(flag(w.args, '--resident-budget-gib'), String(old279(63)), `${v}: the 80% budget`);
    assert.equal(w.env.STRATA_RESIDENT_HEADROOM_GIB, '13', `${v}: 20% of 63 GB, rounded up`);
    assert.equal(w.env.STRATA_RAM_KEEP_GIB, undefined, `${v}: the setting it does not know is not sent`);
    assert.equal(w.args.includes('--ram-keep-gib'), false);
    assert.deepEqual([w.ui_resource_baseline.ramKeep, w.ui_resource_baseline.ramBudget, w.ui_resource_baseline.ramHeadroom], [false, true, true]);
  }
  // no engine version at all (not readable): the rule every engine understands
  const unknown = resourceSettings(config, pc64, 'work', iq3s);
  assert.equal(flag(unknown.args, '--resident-budget-gib'), String(old279(63)));
  assert.equal(unknown.env.STRATA_RAM_KEEP_GIB, undefined);
  // an engine named by the run config's exe wins over the active engine's version
  const pinned = resourceSettings({ ...config, exe: 'C:\\x\\components\\engine\\0.1.54-windows-x86_64-cuda-cuda13\\strata.exe' }, pc64, 'work', iq3sNew);
  assert.equal(pinned.env.STRATA_RAM_KEEP_GIB, undefined);
  assert.equal(flag(pinned.args, '--resident-budget-gib'), String(old279(63)));
  const newer = resourceSettings({ ...config, exe: '/x/components/engine/0.1.57-linux-x86_64-cuda/strata' }, pc64, 'work', iq3sOld);
  assert.equal(newer.env.STRATA_RAM_KEEP_GIB, '8');
  assert.equal(newer.args.includes('--resident-budget-gib'), false);
});

test('24, 32, 64 and 128 GB: a newer engine gets W in the env, an older one gets the 0.2.9 budget and 20% headroom; "Leave 20%" off gives neither', () => {
  const cases = [[24, '5', 9, '5'], [32, '7', 15, '7'], [64, '8', 41, '13'], [128, '8', null, null], [16, '5', 8, '4'], [26, '6', 10, '6'], [36, '8', 18, '8'], [63, '8', old279(63), '13']];
  for (const [ram, w, budget, headroom] of cases) {
    const pc = { ...pc64, ram_gb: ram };
    const on = resourceSettings(config, pc, 'work', iq3sNew);
    assert.equal(on.env.STRATA_RAM_KEEP_GIB, w, `${ram} GB, new engine, Leave 20% on`);
    assert.equal(on.args.includes('--ram-keep-gib'), false, `${ram} GB: not a flag`);
    assert.equal(on.args.includes('--resident-budget-gib'), false, `${ram} GB: no budget from the app`);
    assert.equal(on.env.STRATA_RESIDENT_HEADROOM_GIB, undefined, `${ram} GB: no env headroom`);
    const older = resourceSettings(config, pc, 'work', iq3sOld);
    assert.equal(older.env?.STRATA_RAM_KEEP_GIB, undefined, `${ram} GB, old engine: no setting it does not know`);
    assert.equal(older.args.includes('--ram-keep-gib'), false);
    if (budget === null) {
      assert.equal(older.args.includes('--resident-budget-gib'), false, `${ram} GB, old engine: the experts fit 80% of the RAM with 10 to spare: left alone`);
      assert.equal(older.env, undefined);
    } else {
      assert.equal(flag(older.args, '--resident-budget-gib'), String(budget), `${ram} GB, old engine: the 0.2.9 budget`);
      assert.equal(older.env.STRATA_RESIDENT_HEADROOM_GIB, headroom, `${ram} GB, old engine: 20% of the RAM rounded up`);
    }
    for (const v of [iq3sNew, iq3sOld]) {
      const off = resourceSettings(config, pc, 'performance', v);
      assert.equal(off.args.includes('--ram-keep-gib'), false, `${ram} GB, Leave 20% off`);
      assert.equal(off.args.includes('--resident-budget-gib'), false, `${ram} GB, off: no budget either`);
      assert.equal(off.env, undefined, `${ram} GB, off: no env`);
    }
  }
  assert.equal(ramKeepGib(24), 5); assert.equal(ramKeepGib(32), 7); assert.equal(ramKeepGib(64), 8); assert.equal(ramKeepGib(128), 8);
  assert.equal(ramKeepGib(0), 5, 'unknown RAM: the floor'); assert.equal(ramKeepGib(undefined), 5);
});

test('full performance takes the RAM settings of either engine out again and keeps what the installation had', () => {
  for (const model of [iq3sNew, iq3sOld]) {
    const work = resourceSettings(config, pc64, 'work', model);
    const back = resourceSettings({ ...config, args: work.args, env: work.env, ui_resource_profile: 'work', ui_resource_baseline: work.ui_resource_baseline }, pc64, 'performance', model);
    for (const name of ['STRATA_RAM_KEEP_GIB', 'STRATA_RESIDENT_HEADROOM_GIB']) assert.equal(back.env?.[name], undefined, `${model.engineVersion}: ${name}`);
    assert.equal(back.args.includes('--ram-keep-gib'), false);
    assert.equal(back.args.includes('--resident-budget-gib'), false);
    assert.deepEqual([back.ui_resource_baseline.ramKeep, back.ui_resource_baseline.ramBudget, back.ui_resource_baseline.ramHeadroom], [false, false, false]);
    assert.equal(flag(back.args, '--pool-workers'), '12');
  }
  assert.equal(resourceSettings(config, pc64, 'performance', iq3sNew).env, undefined, 'performance from the start: nothing is passed, the whole system is used');
});

test('switching the engine in the Engine tab between loads swaps the rule: new -> old and old -> new', () => {
  const asConfig = r => ({ ...config, args: r.args, env: r.env, ui_resource_profile: 'work', ui_resource_baseline: r.ui_resource_baseline });
  const onNew = resourceSettings(config, pc64, 'work', iq3sNew);
  const toOld = resourceSettings(asConfig(onNew), pc64, 'work', iq3sOld);
  assert.equal(toOld.env.STRATA_RAM_KEEP_GIB, undefined, 'the setting is not left behind for the engine that ignores it');
  assert.equal(flag(toOld.args, '--resident-budget-gib'), String(old279(63)));
  assert.equal(toOld.env.STRATA_RESIDENT_HEADROOM_GIB, '13');
  const backNew = resourceSettings(asConfig(toOld), pc64, 'work', iq3sNew);
  assert.equal(backNew.args.includes('--resident-budget-gib'), false, 'the old budget is gone again');
  assert.equal(backNew.env.STRATA_RESIDENT_HEADROOM_GIB, undefined);
  assert.equal(backNew.env.STRATA_RAM_KEEP_GIB, '8');
});

test('a model whose size is unknown, or one with a RAM mode of the person, is left alone (either engine)', () => {
  for (const model of [iq3sNew, iq3sOld]) {
    assert.equal(resourceSettings(config, { ...pc64, ram_gb: 0 }, 'work', model).env, undefined, 'no RAM figure, no change');
    const ownEnv = resourceSettings({ ...config, env: { STRATA_RAM_KEEP_GIB: '3' } }, pc64, 'work', model);
    if (model === iq3sNew) assert.equal(ownEnv.env, undefined, 'a STRATA_RAM_KEEP_GIB of the person is left in the run config as it is');
    else {   // an engine that ignores that variable: it keeps the person's value in the file and the 0.2.9 rule gives the room
      assert.equal(ownEnv.env.STRATA_RAM_KEEP_GIB, '3');
      assert.equal(flag(ownEnv.args, '--resident-budget-gib'), String(old279(63)));
    }
    for (const mine of [['--mmap-experts'], ['--resident-experts'], ['--resident-budget-gib', '30'], ['--ram-keep-gib', '3']]) {
      const own = { ...config, args: [...config.args, ...mine] };
      const work = resourceSettings(own, pc64, 'work', model);
      assert.equal(work.env, undefined, `${mine[0]}: the app adds no RAM setting of its own`);
      assert.ok(work.args.includes(mine[0]), `${mine[0]} stays as it is`);
      if (mine[0] === '--resident-budget-gib') assert.equal(flag(work.args, '--resident-budget-gib'), '30');
      assert.equal(work.args.filter(a => a === '--ram-keep-gib').length, mine[0] === '--ram-keep-gib' ? 1 : 0, `${mine[0]}: the app adds no flag`);
    }
  }
  assert.equal(resourceSettings(config, pc64, 'work').env, undefined, 'no model size, no change');
});

test('a config written by the earlier version (an 80% budget and a 20% env headroom) is replaced by the rule of the engine in use, and full performance takes the old settings out', () => {
  const old = { ...config, args: [...config.args, '--resident-budget-gib', '38'], env: { STRATA_RESIDENT_HEADROOM_GIB: '13' },
    ui_resource_profile: 'work', ui_resource_baseline: { workers: 12, reserve: 0, ramBudget: true, ramHeadroom: true } };
  const work = resourceSettings(old, pc64, 'work', iq3sNew);
  assert.equal(work.args.includes('--resident-budget-gib'), false, 'the old budget is gone for an engine that knows the setting');
  assert.equal(work.env.STRATA_RAM_KEEP_GIB, '8');
  assert.equal(work.env.STRATA_RESIDENT_HEADROOM_GIB, undefined, 'and the old 20% env with it');
  const same = resourceSettings(old, pc64, 'work', iq3sOld);
  assert.equal(flag(same.args, '--resident-budget-gib'), String(old279(63)), 'an older engine keeps the 0.2.9 rule');
  assert.equal(same.env.STRATA_RESIDENT_HEADROOM_GIB, '13');
  const perf = resourceSettings(old, pc64, 'performance', iq3sNew);
  assert.equal(perf.args.includes('--resident-budget-gib'), false);
  assert.equal(perf.args.includes('--ram-keep-gib'), false);
  assert.equal(perf.env.STRATA_RESIDENT_HEADROOM_GIB, undefined);
  assert.equal(perf.env.STRATA_RAM_KEEP_GIB, undefined);
});

test('neither profile reserves less VRAM than the engine\'s safe floor (the driver pages GPU memory below it and decoding collapses)', () => {
  const card12 = { cpu: { cores: 12 }, ram_gb: 63, gpus: [{ index: 0, vram_gb: 11.9 }] };
  const card24 = { cpu: { cores: 12 }, ram_gb: 63, gpus: [{ index: 0, vram_gb: 24 }] };
  for (const profile of ['work', 'performance']) {
    const lowConfig = { gpu: 0, args: ['--max-context', '32768', '--vram-reserve-mib', '700'] };
    assert.ok(resourceSettings(lowConfig, card12, profile).vram_reserve_mib >= 1536, `${profile} on 12 GB`);
    assert.ok(resourceSettings({ gpu: 0, args: [] }, card12, profile).vram_reserve_mib >= 1536, `${profile}, nothing configured`);
    assert.ok(resourceSettings(lowConfig, card24, profile).vram_reserve_mib >= 1024, `${profile} on 24 GB`);
  }
  assert.equal(resourceSettings({ gpu: 0, args: [] }, card12, 'performance').vram_reserve_mib, 1536);
  assert.equal(resourceSettings({ gpu: 0, args: [] }, card24, 'performance').vram_reserve_mib, 1024);
  assert.equal(resourceSettings({ gpu: 0, args: [], vram_reserve_mib: 2437 }, card12, 'performance').vram_reserve_mib, 2437, 'a larger reserve of the installation is kept');
});

test('a STRATA_RESIDENT_HEADROOM_GIB of the person is left in the env (the engine reads it first); the app sets no env headroom of its own for an engine that knows the setting', () => {
  const own = resourceSettings({ ...config, env: { STRATA_RESIDENT_HEADROOM_GIB: '6', OTHER: 'x' } }, pc64, 'work', iq3sNew);
  assert.equal(own.env.STRATA_RESIDENT_HEADROOM_GIB, '6', 'a value the user set is not overwritten (the engine reads it before W)');
  assert.equal(own.env.OTHER, 'x');
  assert.equal(own.env.STRATA_RAM_KEEP_GIB, '8');
  for (const ram of [24, 32, 64, 128]) assert.equal(resourceSettings(config, { ...pc64, ram_gb: ram }, 'work', iq3sNew).env.STRATA_RESIDENT_HEADROOM_GIB, undefined, `${ram} GB: no env headroom`);
  // an older engine: the 0.2.9 behaviour, a headroom of the person's is not overwritten either
  const oldOwn = resourceSettings({ ...config, env: { STRATA_RESIDENT_HEADROOM_GIB: '6' } }, pc64, 'work', iq3sOld);
  assert.equal(oldOwn.env, undefined);
  assert.equal(flag(oldOwn.args, '--resident-budget-gib'), String(old279(63)));
});

test('a start that died gets ONE safer second try: pin less when the locked memory ran out, else a RAM budget with the rest on the SSD', () => {
  const pinned = saferStart(config, pc64, iq3s, 'server_exited: mapped staging allocation failed [pinned-memory-exhausted] start again with STRATA_ARENA_PIN_GIB=26 (pin less)');
  assert.equal(pinned.config.env.STRATA_ARENA_PIN_GIB, '26');
  assert.match(pinned.words, /would not lock as much memory/);
  assert.equal(flag(saferStart(pinned.config, pc64, iq3s, 'server_exited: mapped staging allocation failed [pinned-memory-exhausted]').config.args, '--resident-budget-gib'), String(Math.floor(63 * .5) - 4), 'already pinned less: the next safer setting is the RAM budget');
  const noHint = saferStart(config, pc64, iq3s, 'mapped staging allocation failed');
  assert.equal(noHint.config.env.STRATA_ARENA_PIN_GIB, String(Math.floor(63 / 2) - 8));
  const low = saferStart(config, pc64, iq3s, 'server_exited: memory guard: 1.9 GiB free');
  assert.equal(flag(low.config.args, '--resident-budget-gib'), String(Math.floor(63 * .5) - 4));
  assert.match(low.words, /low-memory mode/);
  const already = saferStart({ ...config, args: [...config.args, '--resident-budget-gib', '40'] }, pc64, iq3s, 'server_exited');
  assert.equal(flag(already.config.args, '--resident-budget-gib'), '28', '70% of the budget it had');
  assert.equal(saferStart({ ...config, args: [...config.args, '--resident-budget-gib', '10'] }, pc64, iq3s, 'server_exited'), null, 'below 8 GB there is nothing left to give');
  assert.equal(saferStart({ ...config, args: [...config.args, '--resident-experts'] }, pc64, iq3s, 'server_exited'), null, 'already the SSD mode');
  assert.equal(saferStart(config, pc64, { arenaGb: 20 }, 'server_exited'), null, 'a model that fits RAM gains nothing from the SSD mode');
  assert.equal(saferStart(config, pc64, undefined, 'server_exited'), null, 'unknown model size: no guess');
  assert.equal(saferStart(config, pc64, iq3s, 'Some model files are missing'), null, 'not a memory failure: no second try');
  assert.equal(config.args.includes('--resident-budget-gib'), false, 'the input config is not changed');
});

test('an engine that stopped because the GPU expert cache got no room is started again with less VRAM kept free, then a shorter context (#3192)', () => {
  const tenGb = { cpu: { cores: 16 }, ram_gb: 64, gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 3080', vram_gb: 10 }] };
  const cfg = { gpu: 0, args: ['--native', 'x.gguf', '--max-context', '131072', '--vision', '--vram-reserve-mib', '2048'] };
  const log = "server_exited (exit code 1): strata generate: no VRAM is left for the expert cache: it needs at least 288 slots (639 MiB), about 727 MiB more than this card has free. To make room: a smaller --max-context (the session, mostly its KV cache, takes 1931 MiB at 131072 tokens), --kv q4_0 | strata serve: verify: needs the profile-filled VRAM expert tier";
  const safer = saferStart(cfg, tenGb, iq3s, log);
  assert.equal(flag(safer.config.args, '--vram-reserve-mib'), '1065', '2048 - (727 + 256 margin)');
  assert.equal(flag(safer.config.args, '--max-context'), '131072', 'the reserve alone was enough: the context stays');
  assert.match(safer.words, /no room left/);
  const tight = saferStart({ ...cfg, args: ['--max-context', '131072', '--vram-reserve-mib', '1100'] }, tenGb, iq3s, log);
  assert.equal(flag(tight.config.args, '--vram-reserve-mib'), '1024', 'never below 1024 MiB (#2065)');
  assert.ok(Number(flag(tight.config.args, '--max-context')) < 131072, 'the rest comes from the context');
  assert.equal(saferStart({ ...cfg, args: ['--max-context', '16384', '--vram-reserve-mib', '1024'] }, tenGb, iq3s, log), null, 'nothing left to give');
  assert.equal(flag(saferStart(cfg, tenGb, undefined, 'server_exited: strata serve: verify: needs the profile-filled VRAM expert tier').config.args, '--vram-reserve-mib'), '1092', 'without the numbers: a fixed step, no model size needed');
  assert.equal(cfg.args[cfg.args.indexOf('--vram-reserve-mib') + 1], '2048', 'the input config is not changed');
});

test('0.2.3 tuning: the Work profile takes 80% of the PHYSICAL cores rounded up, never more than the cores less the host thread', () => {
  const w = cores => Number(flag(resourceSettings({ gpu: 0, args: [] }, { cpu: { cores }, ram_gb: 64, gpus: [{ index: 0, vram_gb: 12 }] }, 'work').args, '--pool-workers'));
  assert.equal(w(6), 5, 'Ryzen 5 7600: the engine\'s own 5 (floor gave 4, 8-10% slower; 12 logical threads gave 9 in 0.2.1)');
  assert.equal(w(4), 3);
  assert.equal(w(8), 7);
  assert.equal(w(10), 8);
  assert.equal(w(12), 10);
  assert.equal(w(16), 13);
  assert.equal(w(2), 1);
  const own = { gpu: 0, args: ['--pool-workers', '3'] };
  assert.equal(flag(resourceSettings(own, { cpu: { cores: 6 }, ram_gb: 64, gpus: [] }, 'work').args, '--pool-workers'), '3', 'a count the person set stays when it is lower');
});

test('0.2.3 tuning: Work keeps a fifth of the card but no more than 1536 MiB on a card of 16 GB or less; a CPU picture reader needs no 2048 floor', () => {
  const card = gb => ({ cpu: { cores: 6 }, ram_gb: 64, gpus: [{ index: 0, vram_gb: gb, drives_display: true }] });
  assert.equal(resourceSettings({ gpu: 0, args: [] }, card(11.9), 'work').vram_reserve_mib, 1536, 'RTX 5070: 2437 gave 38 tok/s, 1536 gave 46 (Gyro-S 64K), with 1.1 GiB still free');
  assert.equal(resourceSettings({ gpu: 0, args: [] }, card(16), 'work').vram_reserve_mib, 1536);
  assert.equal(resourceSettings({ gpu: 0, args: [] }, card(24), 'work').vram_reserve_mib, 4915, 'above 16 GB the fifth stays');
  assert.equal(resourceSettings({ gpu: 0, args: [], vision: { gpu: false } }, card(12), 'performance').vram_reserve_mib, 1536, 'the CPU reader takes no VRAM');
  assert.equal(resourceSettings({ gpu: 0, args: [], vision: { gpu: false } }, card(12), 'work').vram_reserve_mib, 1536);
  assert.equal(resourceSettings({ gpu: 0, args: [], vision: { gpu: true } }, card(12), 'performance').vram_reserve_mib, 2048, 'the GPU reader keeps its 2048');
  assert.equal(resourceSettings({ gpu: 0, args: [], vision: { gpu: true } }, card(12), 'work').vram_reserve_mib, 2048);
  assert.equal(resourceSettings({ gpu: 0, args: ['--vram-reserve-mib', '3000'], user_args: ['--vram-reserve-mib', '3000'] }, card(12), 'work').vram_reserve_mib, 3000, 'the person\'s own number is kept');
});

test('an engine that refused the prompt chunk at the start is not retried with a memory setting: the same chunk would fail again', () => {
  const line = 'strata generate: a 32768-token prompt chunk does not fit beside the 52.1 GiB expert cache on this GPU (a stream could not be created): use --prefill auto:16384, or a smaller --expert-cache';
  const withBudget = { ...config, args: [...config.args, '--resident-budget-gib', '40'] };
  assert.equal(saferStart(withBudget, pc64, iq3s, `server_exited (exit code 2): ${line}`), null);
  assert.equal(saferStart(config, pc64, iq3s, `Engine stopped (exit code 2): ${line}`), null);
  assert.ok(saferStart(withBudget, pc64, iq3s, 'server_exited: memory guard: 1.9 GiB free'), 'a real memory stop still gets its safer try');
});
