import test from 'node:test';
import assert from 'node:assert/strict';
import { addEnergy, reportedPowerW } from '../server/energy-meter.mjs';

test('an unmetered iGPU does not prevent energy accumulation from a metered GPU', () => {
  const readings = [
    { id: 'GPU-discrete', powerW: 72, powerLimitW: 575 },
    { id: 'GPU-integrated', powerW: null, powerLimitW: null },
  ];
  assert.equal(reportedPowerW(readings), 72);
  assert.equal(reportedPowerW(readings, 'powerLimitW'), 575);
  const start = addEnergy(reportedPowerW(readings), 1_000);
  assert.equal(addEnergy(reportedPowerW(readings), 6_000) - start, 0.1);
});

test('power remains unavailable when no GPU reports it, and multiple readings are summed', () => {
  assert.equal(reportedPowerW([]), null);
  assert.equal(reportedPowerW([{ powerW: null }, { powerW: NaN }]), null);
  assert.equal(reportedPowerW([{ powerW: 20 }, { powerW: 30 }]), 50);
});

// GUI-1116: the Monitor's total power and limit come from the same sum as the energy meter (a mixed PC no longer shows a dash).
test('the Monitor sums the cards that report power, as the meter does', async () => {
  const { readFileSync } = await import('node:fs');
  const src = readFileSync(new URL('../server/monitor.mjs', import.meta.url), 'utf8');
  assert.match(src, /powerW:reportedPowerW\(readings\),powerLimitW:reportedPowerW\(readings,'powerLimitW'\),/);
});
