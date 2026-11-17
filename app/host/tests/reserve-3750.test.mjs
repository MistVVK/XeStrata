/**
 * #3750: "UI resource profile hard-codes vram_reserve_mib 3256: 256K context + Q4 KV leaves 0 MiB for the expert cache -> EngineDied at verify".
 * The 3256 was never a constant: it was a fifth of the reporter's 16 GB card (0.2.2 kept 20% for "my work"). Since 0.2.3 a card of 16 GB or
 * less keeps at most 1536 MiB, and a start that dies for lack of cache room is started once more with less kept free (saferStart, #3192).
 * These tests pin both on the reporter's configuration (RX 9070 XT 16 GB, 256K context, Q4 KV, the file the old app wrote) and on a 12 GiB card,
 * and pin that a larger card keeps its fifth (default output unchanged).
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultReserves, reserveChoice, resourceSettings, saferStart } from '../server/native-resources.mjs';

const flag = (args, name) => args[args.indexOf(name) + 1];
const card = (gb, extra = {}) => ({ cpu: { cores: 8 }, ram_gb: 64, gpus: [{ index: 0, name: 'card', vram_gb: gb, ...extra }] });
const rx9070xt = card(15.9, { name: 'AMD Radeon RX 9070 XT', vendor: 'amd' });
// the reporter's file after the 0.2.2 app wrote it
const reporterFile = () => ({
  gpu: 0, vram_reserve_mib: 3256, ui_reserve_applied: 3256, ui_resource_profile: 'work', ui_resource_baseline: { workers: 0, reserve: 0, ramBudget: false, ramHeadroom: false },
  args: ['--native', 'iq3_s.gguf', '--max-context', '262144', '--kv', 'q4_0', '--mtp', 'mtp.gguf', '--vram-reserve-mib', '3256', '--expert-cache', 'auto'],
});

test('the reporter\'s card and file: Work and Performance keep 1536 MiB, not 3256, and the old 3256 is not taken for a choice of the person', () => {
  const choice = reserveChoice(reporterFile(), rx9070xt);
  assert.equal(choice.asked, null, 'the 3256 the app itself wrote is not the person\'s');
  assert.deepEqual(choice.defaults, { work: 1536, performance: 1536 });
  for (const profile of ['work', 'performance']) {
    const r = resourceSettings(reporterFile(), rx9070xt, profile, { arenaGb: 83 });
    assert.equal(r.vram_reserve_mib, 1536, profile);
    assert.equal(flag(r.args, '--vram-reserve-mib'), '1536', profile);
    assert.equal(r.args.filter(a => a === '--vram-reserve-mib').length, 1, 'one flag only');
    assert.equal(flag(r.args, '--max-context'), '262144', 'the context is the person\'s and is not touched');
    assert.equal(flag(r.args, '--kv'), 'q4_0');
  }
});

test('a 12 GiB card at 256K context and Q4 KV: the reserve is 1536 MiB, which leaves the expert cache room', () => {
  const twelve = card(12, { name: 'NVIDIA GeForce RTX 3060' });
  const cfg = { gpu: 0, args: ['--native', 'iq3_s.gguf', '--max-context', '262144', '--kv', 'q4_0', '--mtp', 'mtp.gguf'] };
  for (const profile of ['work', 'performance']) {
    const r = resourceSettings(cfg, twelve, profile, { arenaGb: 83 });
    assert.equal(r.vram_reserve_mib, 1536, profile);
    // a fifth of 12 GiB (2457) would have been 921 MiB more of the 12 GiB gone before the cache: the cap is why it is not taken
    assert.ok(Math.floor(12 * 1024 * 0.2) > r.vram_reserve_mib);
  }
});

test('every card of 16 GB or less keeps at most 1536 MiB; a larger card keeps its fifth, exactly as before', () => {
  for (const gb of [6, 8, 10, 11.9, 12, 15.9, 16]) {
    const d = defaultReserves({ gpu: 0, args: [] }, card(gb));
    assert.ok(d.work <= 1536 && d.performance <= 1536, `${gb} GB: ${JSON.stringify(d)}`);
  }
  assert.deepEqual(defaultReserves({ gpu: 0, args: [] }, card(24)), { work: 4915, performance: 1024 });
  assert.deepEqual(defaultReserves({ gpu: 0, args: [] }, card(32)), { work: 6553, performance: 1024 });
  assert.deepEqual(defaultReserves({ gpu: 0, args: [] }, card(48)), { work: 9830, performance: 1024 });
});

test('a reserve the person set is theirs, 3256 included: it is not replaced by the default', () => {
  const mine = { ...reporterFile(), ui_reserve_applied: undefined, ui_resource_baseline: undefined, vram_reserve_mib: 3256 };
  const r = resourceSettings(mine, rx9070xt, 'work', { arenaGb: 83 }, { vram_reserve_mib: 3256 });
  assert.equal(r.vram_reserve_mib, 3256);
  assert.equal(r.ui_reserve_applied, undefined, 'and the app does not claim it');
});

test('a start that still dies for lack of cache room (the reporter\'s own log) is started again once with less kept free', () => {
  const log = 'server_exited (exit code 1): strata generate: expert cache auto: 6.62 GiB free, 3256 MiB reserved (+218 MiB for the draft head) -> 1301 slots | '
    + 'strata generate: only 0 MiB free once the slots are written (reserve 3256 MiB); shrinking the expert cache | '
    + 'strata serve: verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache); with --expert-cache auto, no VRAM was left for it';
  const cfg = reporterFile();
  const safer = saferStart(cfg, rx9070xt, { arenaGb: 83 }, log);
  assert.ok(safer, 'a second try is offered');
  const next = Number(flag(safer.config.args, '--vram-reserve-mib'));
  assert.ok(next < 3256 && next >= 1024, `reserve ${next}`);
  assert.equal(flag(cfg.args, '--vram-reserve-mib'), '3256', 'the input is not changed and nothing is saved');
  // at the app's own 1536 the same failure still gets its second try, down to the 1024 floor and then a shorter context
  const tight = saferStart({ ...cfg, args: cfg.args.map(a => (a === '3256' ? '1536' : a)) }, rx9070xt, { arenaGb: 83 }, log + ' | strata generate: no VRAM is left for the expert cache: it needs at least 288 slots (639 MiB), about 900 MiB more than this card has free. To make room: a smaller --max-context (the session, mostly its KV cache, takes 2200 MiB at 262144 tokens)');
  assert.equal(flag(tight.config.args, '--vram-reserve-mib'), '1024');
  assert.ok(Number(flag(tight.config.args, '--max-context')) < 262144, 'the rest comes from the context');
});
