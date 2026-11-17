// The host test runner (scripts/test.mjs) hands every tests/*.test.mjs to `node --test` on one command line. With absolute paths that line grew with the length of
// the checkout's own path: 337 files in a worktree with a long name were 34,272 characters against Windows' 32,767, and the spawn failed with ENAMETOOLONG.
import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { spawnSync } from 'node:child_process';

const runnerSource = new URL('../scripts/test.mjs', import.meta.url);

test('the runner passes the test files relative to the host folder, in sorted order', () => {
  const source = readFileSync(runnerSource, 'utf8').replace(/\r\n/g, '\n');
  assert.ok(source.includes(".sort().map(name=>`tests/${name}`);"), 'relative names');
  assert.doesNotMatch(source, /resolve\(root,'tests',name\)/, 'no absolute names');
  assert.match(source, /\{cwd:root,stdio:'inherit',env\}/, 'node runs in the host folder, where the relative names resolve');
});

test('a suite in a folder with a long path runs: 400 files whose absolute names would not fit on one command line', t => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-suite-long-'));
  const parent = realpathSync(tmpdir());
  t.after(() => {
    assert.equal(dirname(realpathSync(dir)), parent, 'remove only the owned scratch directory');
    rmSync(dir, { recursive: true, force: true, maxRetries: 3 });
  });
  // a long folder name, so 400 absolute paths come to far more than Windows' 32,767 characters (they are about 100 characters each)
  const suite = join(dir, 'a-checkout-with-a-rather-long-name-'.repeat(2) + 'x'), scripts = join(suite, 'scripts'), tests = join(suite, 'tests');
  mkdirSync(scripts, { recursive: true }); mkdirSync(tests);
  copyFileSync(runnerSource, join(scripts, 'test.mjs'));
  const N = 400;
  for (let i = 0; i < N; i++) writeFileSync(join(tests, `t${String(i).padStart(3, '0')}.test.mjs`), "import test from 'node:test'; test('ok', () => {});\n");
  const absolute = [...Array(N).keys()].map(i => join(tests, `t${String(i).padStart(3, '0')}.test.mjs`).length + 1).reduce((a, b) => a + b, 0);
  assert.ok(absolute > 32_767, `the old form is ${absolute} characters`);
  const temp = join(dir, 'tmp'); mkdirSync(temp);
  const env = { ...process.env, TEMP: temp, TMP: temp, TMPDIR: temp };
  delete env.NODE_TEST_CONTEXT;                                     // an independent runner, not another worker of this node:test run
  const result = spawnSync(process.execPath, [join(scripts, 'test.mjs')], { env, encoding: 'utf8', windowsHide: true, timeout: 120_000 });
  assert.equal(result.error, undefined, result.error?.message);
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.match(result.stdout, new RegExp(`(?:^|\\n)\\W*pass ${N}\\s*(?:\\n|$)`), 'all 400 files ran');
});
