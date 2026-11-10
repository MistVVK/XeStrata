/**
 * QAGUI-STRIX 11-09 (APP-WIN item 3): on a chip that shares the PC's memory the app's Load keeps the reserve setup wrote (Windows Strix Halo: 700, pictures on
 * or off), so it starts the engine as `strata launch` does; with none of setup's in the run config it takes the one Control says setup would write for this
 * chip (hardware report `setup_reserve_mib`, Windows gfx1151, one card); without that it keeps the floor of before (1024, 2048 with pictures), so an old
 * Windows config never falls to the engine's own 2560 MiB floor (G16). A card with its own memory is unchanged.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { resourceSettings } from '../server/native-resources.mjs';

const reserveOf = r => { const at = r.args.indexOf('--vram-reserve-mib'); return at >= 0 ? Number(r.args[at + 1]) : null; };
const strixWin = { cpu: { cores: 16 }, ram_gb: 63.6, gpus: [{ index: 0, name: 'AMD Radeon(TM) 8060S Graphics', vram_gb: 95.8, uma: true, dedicated_gb: 64, setup_reserve_mib: 700 }] };
const strixLinux = { cpu: { cores: 16 }, ram_gb: 63.6, gpus: [{ index: 0, name: 'AMD Radeon 8060S', vram_gb: 95.8, uma: true, dedicated_gb: 64 }] };
const setupArgs = ['--max-context', '131072', '--kv', 'int8', '--resident-budget-gib', '38', '--vram-reserve-mib', '700'];
const noReserveArgs = ['--max-context', '131072', '--kv', 'int8', '--resident-budget-gib', '38'];
const pictures = { gpu: true, exe: 'strata-vision-hip.exe' };

test('Windows Strix Halo, setup wrote 700: both profiles, pictures on or off, keep 700 (it was 1024 / 2048)', () => {
  for (const profile of ['work', 'performance']) for (const vision of [undefined, pictures]) {
    const r = resourceSettings({ gpu: 0, args: setupArgs, ...(vision ? { vision } : {}) }, strixWin, profile);
    assert.equal(reserveOf(r), 700, `${profile} pictures ${!!vision}`);
    assert.equal(r.args.filter(a => a === '--vram-reserve-mib').length, 1);
  }
});

test('a 0.2.13 Strix config the app already rewrote to 1024 / 2048 comes back to setup\'s 700 (its baseline recorded it)', () => {
  const first = resourceSettings({ gpu: 0, args: setupArgs, vision: pictures }, strixLinux, 'work');
  const rewritten = { gpu: 0, args: [...noReserveArgs, '--pool-workers', '13', '--vram-reserve-mib', '2048'], vision: pictures, vram_reserve_mib: 2048, ui_reserve_applied: 2048, ui_resource_baseline: { ...first.ui_resource_baseline } };
  assert.equal(first.ui_resource_baseline.reserve, 700);
  assert.equal(reserveOf(resourceSettings(rewritten, strixWin, 'work')), 700);
  assert.equal(reserveOf(resourceSettings(rewritten, strixWin, 'performance')), 700);
});

test('Windows Strix Halo, an old config with no reserve: the one setup would write (Control\'s setup_reserve_mib), never no reserve (G16)', () => {
  for (const profile of ['work', 'performance']) for (const vision of [undefined, pictures])
    assert.equal(reserveOf(resourceSettings({ gpu: 0, args: noReserveArgs, ...(vision ? { vision } : {}) }, strixWin, profile)), 700, `${profile} pictures ${!!vision}`);
});

test('no setup reserve and no setup_reserve_mib (Linux Strix, an older Control): the floor of before, 1024 and 2048 with pictures', () => {
  for (const profile of ['work', 'performance']) {
    assert.equal(reserveOf(resourceSettings({ gpu: 0, args: noReserveArgs }, strixLinux, profile)), 1024, profile);
    assert.equal(reserveOf(resourceSettings({ gpu: 0, args: noReserveArgs, vision: pictures }, strixLinux, profile)), 2048, profile);
  }
  // two cards named: setup's one-card rule does not apply
  const two = { ...strixWin, gpus: [...strixWin.gpus, { ...strixWin.gpus[0], index: 1 }] };
  assert.equal(reserveOf(resourceSettings({ gpu: [0, 1], args: noReserveArgs }, two, 'performance')), 1024);
});

test('a person\'s own reserve still wins, 0 included; a card with its own memory is unchanged', () => {
  assert.equal(reserveOf(resourceSettings({ gpu: 0, args: setupArgs, user_args: ['--vram-reserve-mib', '0'] }, strixWin, 'work')), 0);
  assert.equal(reserveOf(resourceSettings({ gpu: 0, args: setupArgs }, strixWin, 'work', {}, { vram_reserve_mib: 3000 })), 3000);
  const rtx = { cpu: { cores: 16 }, ram_gb: 64, gpus: [{ index: 0, name: 'RTX 5070', vram_gb: 11.9 }] };
  assert.equal(reserveOf(resourceSettings({ gpu: 0, args: noReserveArgs, vision: pictures }, rtx, 'performance')), 2048);
  assert.equal(reserveOf(resourceSettings({ gpu: 0, args: noReserveArgs }, rtx, 'work')), 1536);
});
