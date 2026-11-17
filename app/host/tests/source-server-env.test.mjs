import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { sourceServerEnvironment } from '../server/control-runtime.mjs';

test('the source-checkout server never gets an empty STRATA_API_KEY (#3851)', async () => {
  const env = sourceServerEnvironment({ PATH: 'x', STRATA_API_KEY: 'from-the-shell' });
  assert.equal('STRATA_API_KEY' in env, false);
  assert.equal(env.PATH, 'x');
  assert.equal(env.PYTHONUTF8, '1');
  assert.equal('STRATA_API_KEY' in sourceServerEnvironment({}), false);
  const control = await readFile(new URL('../server/native-control.mjs', import.meta.url), 'utf8');
  assert.doesNotMatch(control, /STRATA_API_KEY:\s*''/);
  assert.match(control, /env:sourceServerEnvironment\(\)/);
});
