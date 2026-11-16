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

test('the card with the most memory is named, not the first one with any memory', async () => {
  const built = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/first-run-card.ts', import.meta.url))], bundle: true, write: false, platform: 'node', format: 'esm' });
  const { firstRunCard } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
  const igpu = { name: 'AMD Radeon(TM) Graphics', vramGb: 2 }, rx = { name: 'AMD Radeon RX 9070', vramGb: 16 };
  assert.equal(firstRunCard([igpu, rx]), rx, 'the reporter\'s PC (#6015)');
  assert.equal(firstRunCard([rx, igpu]), rx);
  assert.equal(firstRunCard([{ name: 'tiny', vramGb: 0.5 }, rx]), rx);
  const a = { name: 'A', vramGb: 12 }, b = { name: 'B', vramGb: 12 };
  assert.equal(firstRunCard([a, b]), a, 'equal cards keep their order');
  const unknown = { name: 'no figure', vramGb: null };
  assert.equal(firstRunCard([unknown, { name: 'other', vramGb: null }]), unknown, 'no memory figures: the first card');
  assert.equal(firstRunCard([]), undefined);
});
