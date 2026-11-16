// #6015: with the processor's graphics first in the list (AMD Radeon(TM) Graphics, 2 GB) and an RX 9070 (16 GB) second, the first-run
// screen said "Strata found your graphics card: AMD Radeon(TM) Graphics, 2 GB". It names the card with the most memory now.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const source = readFileSync(fileURLToPath(new URL('../src/components/EngineFirstRun.tsx', import.meta.url)), 'utf8').replace(/\r\n/g, '\n');

test('the first-run screen takes its card from firstRunCard', () => {
  assert.match(source, /import \{ firstRunCard \} from '\.\.\/lib\/first-run-card';/);
  const line = source.slice(source.indexOf('function gpuLine('), source.indexOf('\n}\n', source.indexOf('function gpuLine(')));
  assert.match(line, /const first = firstRunCard\(info\.gpus\);/);
});

const built = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/first-run-card.ts', import.meta.url))], bundle: true, write: false, platform: 'node', format: 'esm' });
const { firstRunCard, builtIntoProcessor } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

test('setup\'s order: a discrete card before the processor\'s graphics, then the most memory, then the lower number', () => {
  const igpu = { name: 'AMD Radeon(TM) Graphics', vramGb: 2 }, rx = { name: 'AMD Radeon RX 9070', vramGb: 16 };
  assert.equal(firstRunCard([igpu, rx]), rx, 'the reporter\'s PC (#6015)');
  assert.equal(firstRunCard([rx, igpu]), rx);
  const halo = { name: 'AMD Radeon(TM) 8060S Graphics', vramGb: 64 }, w = { name: 'AMD Radeon PRO W7600', vramGb: 8 };
  assert.equal(firstRunCard([halo, w]), w, 'an APU with more (shared) memory than the discrete card still comes after it');
  assert.equal(firstRunCard([halo]), halo, 'an APU alone is the card');
  assert.equal(firstRunCard([{ name: 'Intel(R) UHD Graphics 770', vramGb: 1 }, { name: 'Intel(R) Arc(TM) A750 Graphics', vramGb: 8 }]).name, 'Intel(R) Arc(TM) A750 Graphics');
  assert.equal(firstRunCard([{ name: 'NVIDIA GeForce RTX 3060', vramGb: 12 }, { name: 'NVIDIA GeForce RTX 4090', vramGb: 24 }]).name, 'NVIDIA GeForce RTX 4090');
  const a = { name: 'NVIDIA GeForce RTX 3060', vramGb: 12 }, b = { name: 'NVIDIA GeForce RTX 3060', vramGb: 12.1 };
  assert.equal(firstRunCard([a, b]), a, 'equal (rounded) memory: the lower number');
  const unknown = { name: 'no figure', vramGb: null };
  assert.equal(firstRunCard([unknown, { name: 'other', vramGb: null }]), unknown, 'no memory figures: the first card');
  assert.equal(firstRunCard([]), undefined);
});

test('which names are graphics built into the processor', () => {
  for (const n of ['AMD Radeon(TM) Graphics', 'AMD Radeon 780M Graphics', 'AMD Radeon(TM) 890M Graphics', 'AMD Radeon(TM) 8060S Graphics', 'Intel(R) UHD Graphics', 'Intel(R) Iris(R) Xe Graphics', 'Intel(R) Arc(TM) Graphics'])
    assert.equal(builtIntoProcessor(n), true, n);
  for (const n of ['AMD Radeon RX 9070', 'AMD Radeon RX 7900 XTX', 'AMD Radeon PRO W7900', 'AMD Radeon AI PRO R9700', 'Intel(R) Arc(TM) A750 Graphics', 'Intel(R) Arc(TM) B580 Graphics', 'NVIDIA GeForce RTX 5070'])
    assert.equal(builtIntoProcessor(n), false, n);
});
