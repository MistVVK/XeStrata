import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { spawnSync } from 'node:child_process';

test('the host test runner isolates the home and Strata data inherited by its test children', t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-suite-home-'));
  const parent = realpathSync(tmpdir());
  t.after(() => {
    assert.equal(dirname(realpathSync(dir)), parent, 'remove only the owned scratch directory');
    rmSync(dir, { recursive: true, force: true, maxRetries: 3 });
  });
  const owner = join(dir, 'owner');
  const personalSkill = join(owner, '.agents', 'skills', 'personal', 'SKILL.md');
  const personalState = join(owner, '.strata', 'data', 'state.json');
  mkdirSync(dirname(personalSkill), { recursive: true });
  mkdirSync(dirname(personalState), { recursive: true });
  writeFileSync(personalSkill, 'Personal skill fixture; never part of the test profile.');
  writeFileSync(personalState, '{"personalFixture":true}');
  const scripts = join(dir, 'suite', 'scripts'), tests = join(dir, 'suite', 'tests');
  mkdirSync(scripts, { recursive: true }); mkdirSync(tests);
  const runner = join(scripts, 'test.mjs');
  copyFileSync(new URL('../scripts/test.mjs', import.meta.url), runner);
  const probe = `
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
const owner = ${JSON.stringify(owner)};
assert.notEqual(homedir(), owner, 'Harness must not discover skills in the owner home');
assert.equal(process.env.HOME, homedir());
assert.equal(process.env.USERPROFILE, homedir());
assert.equal(process.env.STRATA_HOME, homedir(), 'the host data root must use the same scratch home');
assert.equal(existsSync(join(homedir(), '.agents', 'skills', 'personal', 'SKILL.md')), false);
assert.equal(existsSync(join(process.env.STRATA_HOME, 'data', 'state.json')), false);
mkdirSync(process.env.STRATA_HOME, { recursive: true });
writeFileSync(join(process.env.STRATA_HOME, 'child-write'), 'scratch only');
console.log('HOME_PROBE_EXECUTED');
`;
  writeFileSync(join(tests, 'probe.test.mjs'), probe);
  const temp = join(dir, 'tmp'); mkdirSync(temp);
  const env = { ...process.env, HOME: owner, USERPROFILE: owner, STRATA_HOME: join(owner, '.strata'),
    TEMP: temp, TMP: temp, TMPDIR: temp };
  // This is an independent test runner, not another worker of the surrounding node:test run.
  delete env.NODE_TEST_CONTEXT;
  const result = spawnSync(process.execPath, [runner], {
    env,
    encoding: 'utf8', windowsHide: true, timeout: 20_000,
  });
  assert.equal(result.error, undefined, result.error?.message);
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.match(result.stdout, /HOME_PROBE_EXECUTED/, 'the child probe actually ran');
  assert.equal(existsSync(join(owner, 'child-write')), false);
  assert.equal(existsSync(join(owner, '.strata', 'child-write')), false);
  assert.equal(readFileSync(personalSkill, 'utf8'), 'Personal skill fixture; never part of the test profile.');
  assert.equal(readFileSync(personalState, 'utf8'), '{"personalFixture":true}');
});
