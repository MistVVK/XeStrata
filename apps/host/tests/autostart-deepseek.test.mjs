/** A remembered DeepSeek model (Strata 0.2.13 and earlier) at the host's start: not started, the approved refusal logged, and the app's
 *  settings file (data/state.json) byte-identical afterwards: the product changes nothing of the person's. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { autostartModel, refusalLine } from '../server/autostart.mjs';

const REAL = { STRATA_REAL_ENGINE: '1' };

test('a remembered DeepSeek model is not started, the refusal is logged, and state.json stays byte-identical', async () => {
  const home = mkdtempSync(join(tmpdir(), 'strata-autostart-'));
  try {
    mkdirSync(join(home, 'data'), { recursive: true });
    const file = join(home, 'data', 'state.json');
    const saved = JSON.stringify({ loaded: 'deepseek-v4.1-flash-ist-das-2.00bpw', current: 'deepseek-v4.1-flash-ist-das-2.00bpw', beginner: { x: 1 } }, null, 2);
    writeFileSync(file, saved);
    const before = readFileSync(file);
    process.env.STRATA_HOME = home;
    const store = await import(`../server/store.mjs?autostart=${Date.now()}`);      // the real store, on this folder
    const lines = [];
    const id = autostartModel(store.get(), REAL, (l) => lines.push(l));
    assert.equal(id, null);
    assert.deepEqual(lines, [refusalLine('deepseek-v4.1-flash-ist-das-2.00bpw')]);
    assert.match(lines[0], /is a DeepSeek model\. This version of Strata no longer runs DeepSeek models, so it was not started\.$/);
    assert.equal(store.get().loaded, 'deepseek-v4.1-flash-ist-das-2.00bpw', 'the remembered value is kept');
    assert.deepEqual(readFileSync(file), before, 'state.json is byte-identical');
  } finally {
    delete process.env.STRATA_HOME;
    rmSync(home, { recursive: true, force: true });
  }
});

test('a remembered Qwen model is still started; nothing without a real engine', () => {
  assert.equal(autostartModel({ loaded: 'qwen3.8-flash-next-iq3_s' }, REAL, () => assert.fail('no log')), 'qwen3.8-flash-next-iq3_s');
  assert.equal(autostartModel({ current: 'qwen3.8-flash-next-iq3_s' }, REAL), 'qwen3.8-flash-next-iq3_s');
  assert.equal(autostartModel({ loaded: 'qwen3.8-flash-next-iq3_s' }, {}), null);
  assert.equal(autostartModel({ loaded: 'deepseek-v4.1-flash' }, {}, () => assert.fail('no log without a real engine')), null);
});
