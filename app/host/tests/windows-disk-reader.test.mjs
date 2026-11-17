import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { parseWindowsDiskLine, createWindowsDiskReader, diskRate } from '../server/system-telemetry.mjs';

function fakeSpawn(log) {
  return (cmd, args) => {
    const proc = new EventEmitter();
    proc.stdout = new EventEmitter();
    proc.stdout.unref = () => {};
    proc.unref = () => {};
    proc.kill = () => { proc.killed = true; };
    log.push({ cmd, args, proc });
    return proc;
  };
}

test('one PowerShell is started and reused for every sample (#3939)', () => {
  const log = [];
  const reader = createWindowsDiskReader({ spawnFn: fakeSpawn(log), now: () => 1000 });
  assert.equal(reader.read(), null, 'nothing before the first line');
  reader.read(); reader.read();
  assert.equal(log.length, 1);
  log[0].proc.stdout.emit('data', '1000 2');
  log[0].proc.stdout.emit('data', '000\n');
  assert.deepEqual(reader.read(), { read: 1000, write: 2000, at: 1000 });
  assert.equal(log.length, 1);
});

test('after repeated early exits it stops starting processes', () => {
  const log = [];
  const reader = createWindowsDiskReader({ spawnFn: fakeSpawn(log), maxStarts: 3 });
  for (let i = 0; i < 6; i++) { reader.read(); log.at(-1).proc.emit('exit'); }
  assert.equal(log.length, 3);
  assert.equal(reader.read(), null);
});

test('garbage lines are ignored and a repeated reading gives no new rate', () => {
  assert.equal(parseWindowsDiskLine('Name : x'), null);
  assert.deepEqual(parseWindowsDiskLine('12 34\r'), { read: 12, write: 34 });
  const prev = { time: 0, counters: { read: 0, write: 0, at: 1000 } };
  assert.deepEqual(diskRate(prev, { read: 1048576, write: 0, at: 2000 }, 5000), { readMbs: 1, writeMbs: 0 });
});
