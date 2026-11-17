import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { sensorPace, ACTIVE_MS, IDLE_MS, WATCH_MS } from '../server/sensor-pace.mjs';
import { readGpuTelemetry } from '../server/device-probes.mjs';

test('an idle app with the Monitor out of sight asks the card rarely (#4066)', () => {
  const idle = sensorPace({ live: false, watchedUntil: 0, now: 1_000 });
  assert.equal(idle.intervalMs, IDLE_MS);
  assert.equal(idle.transfer, false);
  assert.ok(IDLE_MS >= 10 * ACTIVE_MS);
});

test('a running model or an open Monitor keeps the 3 s pace', () => {
  assert.equal(sensorPace({ live: true, now: 1_000 }).intervalMs, ACTIVE_MS);
  assert.equal(sensorPace({ live: true, now: 1_000 }).transfer, true);
  const watched = sensorPace({ live: false, watchedUntil: 1_000 + WATCH_MS, now: 1_000 });
  assert.equal(watched.intervalMs, ACTIVE_MS);
  assert.equal(watched.transfer, false, 'the PCIe sample only means something while a model runs');
  assert.equal(sensorPace({ live: false, watchedUntil: 1_000, now: 1_000 + 1 }).intervalMs, IDLE_MS);
});

const gpus = [{ id: 'GPU-1', vendor: 'nvidia', index: 0, sensorIndex: 0 }];
const calls = [];
const run = async (cmd, args) => {
  calls.push(args[0]);
  return args[0] === 'dmon' ? '# gpu rxpci txpci\n# Idx MB/s MB/s\n 0 12 3' : 'GPU-1, 41, 5, 1024, 30.5, 200, 4, 4, 16';
};

test('without the PCIe sample one nvidia-smi process is started, with it two', async () => {
  calls.length = 0;
  const idle = await readGpuTelemetry(gpus, { transfer: false, run });
  assert.deepEqual(calls.map(c => c === 'dmon' ? 'dmon' : 'query'), ['query']);
  assert.equal(idle[0].temperatureC, 41);
  calls.length = 0;
  await readGpuTelemetry(gpus, { run });
  assert.deepEqual(calls.map(c => c === 'dmon' ? 'dmon' : 'query'), ['query', 'dmon']);
});

test('the monitor uses the pace, the Monitor screen asks while it is open, and /api/monitor marks it watched', async () => {
  const monitor = await readFile(new URL('../server/monitor.mjs', import.meta.url), 'utf8');
  assert.match(monitor, /sensorPace\(/);
  assert.doesNotMatch(monitor, /sensorsAt>=3000/);
  const index = await readFile(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.match(index, /monitor\.watch\(\); if \(url\.searchParams\.get\('drives'\) === '1'\) monitor\.watchDrives\(\); return json\(res, 200, monitor\.snapshot\(\)\)/);
  assert.match(index, /monitor\.setLiveProbe\(/);
  const screen = await readFile(new URL('../src/screens/Monitor.tsx', import.meta.url), 'utf8');
  assert.match(screen, /api\.get\('\/api\/monitor\?drives=1'\)/);
});

test('the drive-temperature reader starts only while the Monitor is watched and ends after it is gone (#4661)', async () => {
  const monitor = await readFile(new URL('../server/monitor.mjs', import.meta.url), 'utf8');
  assert.match(monitor, /STRATA_REAL_ENGINE==='1'&&Date\.now\(\)<drivesUntil&&Date\.now\(\)-hostStartedAt>DRIVES_AFTER_MS\?readTemperatures\(\)/);
  assert.match(monitor, /stopTemperatures\(\)/);
  const { readTemperatures, stopTemperatures } = await import('../server/temperature-telemetry.mjs');
  assert.equal(typeof stopTemperatures, 'function');
  // nothing is started by importing or by stopping: on a PC that never opens the Monitor no PowerShell and no disk query exists
  stopTemperatures();
  assert.deepEqual(await readTemperatures({ platform: 'darwin' }), { cpuTempC: null, ssdTempC: null });
});

test('#4661: the drive reader asks NVMe drives only, leaves a marker across a crash, and gives up a stuck first pass', async () => {
  const { readFile } = await import('node:fs/promises');
  const { EventEmitter } = await import('node:events');
  const { createWindowsSsdReader } = await import('../server/temperature-telemetry.mjs');
  const source = await readFile(new URL('../server/temperature-telemetry.mjs', import.meta.url), 'utf8');
  assert.doesNotMatch(source, /foreach\(\$n in 0\.\.15\)/, 'no blind walk over PhysicalDrive0..15');
  assert.match(source, /BusType -eq 'NVMe'/);
  const fake = () => Object.assign(new EventEmitter(), { stdout: Object.assign(new EventEmitter(), { unref() {} }), unref() {}, kill() { this.killed = true; } });
  const marker = () => { const m = { has: false, exists: () => m.has, write: () => { m.has = true; }, remove: () => { m.has = false; } }; return m; };
  // a healthy first pass: the marker is there while the drives are asked and gone after the first answer
  let clock = 1000, child = fake(), m = marker();
  let reader = createWindowsSsdReader({ spawnFn: () => child, now: () => clock, sentinel: m });
  assert.equal(reader.read(), null);
  assert.equal(m.has, true, 'the marker is left before the first question');
  child.stdout.emit('data', 'ssd 44\r\n');
  assert.equal(m.has, false, 'the first answer clears it');
  assert.equal(reader.read(), 44);
  // the host (or the PC) died in the middle of a question: the marker is still there at the next start, and the drives are never asked again
  let spawned = 0;
  m = marker(); m.has = true;
  reader = createWindowsSsdReader({ spawnFn: () => { spawned++; return fake(); }, now: () => clock, sentinel: m });
  assert.equal(reader.read(), null);
  assert.equal(spawned, 0, 'a marker from a crashed question means nothing is started');
  assert.equal(reader.off, true);
  // a first pass that never answers is ended for good and the marker stays
  child = fake(); m = marker(); clock = 1000;
  reader = createWindowsSsdReader({ spawnFn: () => child, now: () => clock, sentinel: m, firstPassMs: 45000 });
  reader.read();
  clock += 46000;
  assert.equal(reader.read(), null);
  assert.equal(child.killed, true);
  assert.equal(reader.off, true);
  assert.equal(m.has, true);
  // a marker that cannot be written means no question either
  spawned = 0;
  reader = createWindowsSsdReader({ spawnFn: () => { spawned++; return fake(); }, now: () => clock, sentinel: { exists: () => false, write: () => { throw new Error('read-only'); }, remove() {} } });
  assert.equal(reader.read(), null);
  assert.equal(spawned, 0);
});

test('#4661: only the Monitor screen asks for the drives; the start-up fetch and the Models page do not', async () => {
  const { readFile, readdir } = await import('node:fs/promises');
  const { join } = await import('node:path');
  const hits = [];
  const walk = async dir => { for (const entry of await readdir(dir, { withFileTypes: true })) { const p = join(dir, entry.name); if (entry.isDirectory()) await walk(p); else if (/\.(ts|tsx)$/.test(entry.name) && /drives=1/.test(await readFile(p, 'utf8'))) hits.push(entry.name); } };
  await walk(new URL('../src', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
  assert.deepEqual(hits, ['Monitor.tsx']);
  const monitor = await import('../server/monitor.mjs');
  assert.equal(typeof monitor.watchDrives, 'function');
  assert.ok(monitor.DRIVES_AFTER_MS >= 60000);
});
