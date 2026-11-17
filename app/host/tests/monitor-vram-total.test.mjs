// Visual walkthrough (GUI-VIS-1117, finding 8): the Monitor read "VRAM 88.1 / 64 GB" on a Strix Halo (96 GB carve-out, the Windows probe said 64): a reading above its own total.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const { code } = await transform(src('lib/monitor-hardware.ts'), { loader: 'ts', format: 'esm' });
const { gpuTotalShown } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

const strix = [{ id: 'amd:0', vendor: 'amd', index: 0, name: 'Radeon 8060S', vramGb: 64, unified: false }];
const reading = (extra = {}) => [{ id: 'amd:0', memoryUsedGb: 88.1, ...extra }];

test('the engine\'s own total wins when it is larger than the host probe', () => {
  assert.equal(gpuTotalShown(strix, reading({ memoryTotalGb: 96 }), 88.1), 96);
});

test('with no engine total and a reading above the probe there is no total at all (never "88.1 / 64")', () => {
  assert.ok(Number.isNaN(gpuTotalShown(strix, reading(), 88.1)));
  assert.ok(Number.isNaN(gpuTotalShown(strix, reading({ memoryTotalGb: null }), 88.1)));
});

test('a normal reading keeps the host probe total, and a smaller engine total does not lower it', () => {
  assert.equal(gpuTotalShown(strix, reading(), 40), 64);
  assert.equal(gpuTotalShown(strix, reading({ memoryTotalGb: 60 }), 40), 64);
});

test('two dedicated cards add up; a unified device counts once', () => {
  const two = [{ id: 'nvidia:0', vendor: 'nvidia', index: 0, vramGb: 48, unified: false }, { id: 'nvidia:1', vendor: 'nvidia', index: 1, vramGb: 48, unified: false }];
  assert.equal(gpuTotalShown(two, [{ id: 'nvidia:0', memoryTotalGb: 48 }, { id: 'nvidia:1', memoryTotalGb: 48 }], 30), 96);
  const mac = [{ id: 'apple:0', vendor: 'apple', index: 0, vramGb: 48, unified: true }];
  assert.equal(gpuTotalShown(mac, [{ id: 'apple:0', memoryTotalGb: 48 }], 10), 48);
});

test('nothing used yet still shows the total', () => {
  assert.equal(gpuTotalShown(strix, [{ id: 'amd:0', memoryUsedGb: null }], null), 64);
});

test('the Monitor and the engine reading are wired to it', () => {
  assert.ok(src('screens/Monitor.tsx').includes('gpuTotalShown(report(\'memoryUsedGb\').partial ? report(\'memoryUsedGb\').devices : selectedGpus, readings, vram)'));
  assert.ok(src('lib/native-monitor.ts').includes('memoryTotalGb:bytes(r?.mem_total)'));
  assert.ok(src('lib/native-monitor.ts').includes('mem_total:h.gpu_mem_total'));
});
