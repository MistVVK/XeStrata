// #6012: "Energy since start" stayed at 0 on a PC whose second GPU (an AMD iGPU on Windows) has no power counter, because the
// meter was fed the machine total, and that total is unknown when any one GPU gives no power reading.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { addEnergy, energyWh, reportedPowerW } from '../server/energy-meter.mjs';

test('the energy meter counts the GPUs that report power and skips one that has no power counter', () => {
  const rtx = { id: 'GPU-6f6f', powerW: 40.5, powerLimitW: 450, source: 'sensor' };
  const igpu = { id: 'PCI\\VEN_1002&DEV_164E', powerW: null, powerLimitW: null, source: 'sensor' };
  assert.equal(reportedPowerW([rtx, igpu]), 40.5);
  assert.equal(reportedPowerW([rtx, { ...rtx, id: 'GPU-2', powerW: 59.5 }]), 100);
  assert.equal(reportedPowerW([igpu]), null);
  assert.equal(reportedPowerW([]), null);
  assert.equal(reportedPowerW(undefined), null);
  assert.equal(reportedPowerW([{ id: 'x', powerW: Number.NaN }, { id: 'y', powerW: -1 }]), null);

  const before = energyWh();
  addEnergy(reportedPowerW([rtx, igpu]), 20_000);
  addEnergy(reportedPowerW([{ ...rtx, powerW: 360 }, igpu]), 21_000);   // 360 W for 1 s = 0.1 Wh, the iGPU adds nothing
  assert.ok(Math.abs(energyWh() - before - 0.1) < 1e-9);
});

test('the monitor feeds the energy meter from the per-GPU readings, not the all-or-nothing machine total', () => {
  const src = readFileSync(new URL('../server/monitor.mjs', import.meta.url), 'utf8');
  assert.match(src, /addEnergy\(reportedPowerW\(hardwareSnapshot\(\)\.gpus\),/);
  assert.doesNotMatch(src, /addEnergy\(hardwareSnapshot\(\)\.powerW/);
});
