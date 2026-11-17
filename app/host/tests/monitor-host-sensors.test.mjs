import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';

const built = await build({ stdin: { contents: "export * from './src/lib/native-monitor';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const { withHostSensors } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

const none = { id: 'GPU-1', utilizationPct: null, temperatureC: null, memoryUsedGb: null, powerW: null, powerLimitW: null, pcieRxMbs: null };
const live = { hardware: { gpus: [none], gpuLoad: NaN, vramUsedGb: NaN, vramTotalGb: 11.9, gpuTempC: NaN, powerW: NaN, powerLimitW: NaN, pcieRxMbs: NaN, cpuLoad: 20 } };
const host = { hardware: { gpus: [{ id: 'GPU-1', utilizationPct: 7, temperatureC: 41, memoryUsedGb: 0.9, powerW: 22, powerLimitW: 250, pcieRxMbs: 3 }], gpuLoad: 7, vramUsedGb: 0.9, vramTotalGb: 11.9, gpuTempC: 41, powerW: 22, powerLimitW: 250, pcieRxMbs: 3 } };

test('with no model loaded the GPU temperature, load, VRAM, power and PCIe come from the host sensors', () => {
  const out = withHostSensors(live, host);
  assert.equal(out.hardware.gpus[0].temperatureC, 41);
  assert.equal(out.hardware.gpus[0].memoryUsedGb, 0.9);
  assert.equal(out.hardware.gpuLoad, 7);
  assert.equal(out.hardware.powerW, 22);
  assert.equal(out.hardware.pcieRxMbs, 3);
  assert.equal(out.hardware.cpuLoad, 20, 'what the engine reported is kept');
});

test('what a running engine reports wins over the host sensors', () => {
  const running = { hardware: { ...live.hardware, gpus: [{ ...none, temperatureC: 55, utilizationPct: 90 }], gpuTempC: 55, gpuLoad: 90 } };
  const out = withHostSensors(running, host);
  assert.equal(out.hardware.gpus[0].temperatureC, 55);
  assert.equal(out.hardware.gpuLoad, 90);
});

test('no host monitor, no change', () => {
  assert.equal(withHostSensors(live, null), live);
});

// #5946 #4660 #3727: an AMD card on Windows. The engine reported one reading (its memory in use, read through HIP) and the card counted as
// "not empty", so the host's own load and temperature (Windows GPU counters, D3DKMT) were dropped and the tile read "not available".
test('the host fills each reading the engine has not got, field by field, and never overwrites one it has', () => {
  const partial = { hardware: { ...live.hardware, gpus: [{ ...none, memoryUsedGb: 5.2 }] } };
  const g = withHostSensors(partial, host).hardware.gpus[0];
  assert.equal(g.memoryUsedGb, 5.2, 'the engine\'s own reading stays');
  assert.equal(g.utilizationPct, 7, 'load from the host');
  assert.equal(g.temperatureC, 41, 'temperature from the host');
  assert.equal(g.powerW, 22);
  const zeroPower = withHostSensors({ hardware: { ...live.hardware, gpus: [{ ...none, powerW: 0 }] } }, host).hardware.gpus[0];
  assert.equal(zeroPower.powerW, 0, 'a reading of 0 is a reading');
  assert.equal(zeroPower.temperatureC, 41);
  const other = withHostSensors({ hardware: { ...live.hardware, gpus: [{ ...none, id: 'GPU-2', memoryUsedGb: 1 }] } }, host).hardware.gpus[0];
  assert.equal(other.utilizationPct, null, 'another card is not filled from this one');
});
