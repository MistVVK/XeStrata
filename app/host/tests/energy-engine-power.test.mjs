// #3727: "Energy since start" stayed at 0.0 Wh on a Windows AMD card while the Power tile (the engine's reading) showed 235 to 285 W: the host has no
// power reading of its own for that card, and the meter only knew the host's. The engine's own reading, noted by the poll of its metrics, now counts.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { addEnergy, energyWh, enginePowerW, ENGINE_POWER_MAX_AGE_MS, meteredPowerW, metricsPowerW, noteEnginePower } from '../server/energy-meter.mjs';

test('the watts of an engine answer: the per-card readings added up, else the top-level one, else nothing', () => {
  assert.equal(metricsPowerW({ hardware: { gpus: [{ power: 100 }, { power: 40.5 }, { power: null }] } }), 140.5);
  assert.equal(metricsPowerW({ hardware: { gpu_power: 250 } }), 250);
  assert.equal(metricsPowerW({ hardware: { gpus: [{ power: null }], gpu_power: 90 } }), 90);
  assert.equal(metricsPowerW({ hardware: { gpus: [{ util: 5 }] } }), null);
  assert.equal(metricsPowerW({ hardware: { gpu_power: -3 } }), null);
  assert.equal(metricsPowerW(null), null);
});

test('an engine reading fills the meter where the host has none, and is not added to the host\'s own for the same card', () => {
  noteEnginePower('a', 250, 10_000);
  assert.equal(enginePowerW(10_000), 250);
  assert.equal(meteredPowerW(null, 10_000), 250, 'the host has no reading: the engine\'s');
  assert.equal(meteredPowerW(240, 10_000), 250, 'both read the same card: the larger, not the sum');
  assert.equal(meteredPowerW(300, 10_000), 300);
  noteEnginePower('b', 30, 10_000);
  assert.equal(enginePowerW(10_000), 280, 'two engines on two cards add up');
  noteEnginePower('b', null, 10_000);
  assert.equal(enginePowerW(10_000), 250);
  assert.equal(meteredPowerW(null, 10_000 + ENGINE_POWER_MAX_AGE_MS + 1), null, 'an engine that stopped answering counts for nothing');
  assert.equal(meteredPowerW(70, 10_000 + ENGINE_POWER_MAX_AGE_MS + 1), 70);
});

test('0 Wh no more: 250 W from the engine for ten seconds is about 0.69 Wh', () => {
  noteEnginePower('c', 250, 100_000);
  const before = energyWh();
  addEnergy(meteredPowerW(null, 100_000), 100_000);
  for (let s = 1; s <= 10; s++) { noteEnginePower('c', 250, 100_000 + s * 1000); addEnergy(meteredPowerW(null, 100_000 + s * 1000), 100_000 + s * 1000); }
  assert.ok(Math.abs(energyWh() - before - 250 * 10 / 3600) < 1e-9, String(energyWh() - before));
  noteEnginePower('c', null);
});

test('the host feeds the meter with the engines\' watts: the monitor tick and the native poll are wired', () => {
  const monitor = readFileSync(new URL('../server/monitor.mjs', import.meta.url), 'utf8');
  assert.match(monitor, /addEnergy\(meteredPowerW\(reportedPowerW\(hardwareSnapshot\(\)\.gpus\)\),/);
  const native = readFileSync(new URL('../server/native-control.mjs', import.meta.url), 'utf8');
  assert.match(native, /noteEnginePower\(instance\.id,metricsPowerW\(instance\.metrics\)\)/);
  assert.match(native, /noteEnginePower\(instance\.id,null\)/);
});
