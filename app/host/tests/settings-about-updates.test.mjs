/**
 * Settings > About, the update lines:
 * - QAGUI-STRIX 11-09 finding 2: the first-run card's failed engine update ("A model is running ... then press Continue") stayed under About after
 *   Continue installed the engine and after restarts (jobs persist in native-jobs.json). A failure a later finished update of the same component
 *   made right is no longer shown there.
 * - Lane 3 finding 6 (11-09): after an in-app update the row kept "Update available" until the next manual check. An update that finished after
 *   the check shown makes About check again, once per update.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const out = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/native-job-diagnostics.ts', import.meta.url))], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent' });
const { currentUpdateFailure, updateFinishedAfter } = await import(`data:text/javascript;base64,${Buffer.from(out.outputFiles[0].text).toString('base64')}`);

const job = (id, kind, state, startedAt, extra = {}) => ({ id, kind, state, startedAt, summary: state, error: state === 'error' ? 'A model is running, so nothing was changed. Unload the model in Models, then press Continue.' : null, result: null, ...extra });

test('the Strix case: a first-run engine failure, then Continue installed the engine: About shows no failure, also after a restart (same records)', () => {
  const jobs = [job('a', 'update-engine', 'error', '2026-11-09T14:10:00Z'), job('b', 'update-engine', 'done', '2026-11-09T14:10:30Z')];
  assert.equal(currentUpdateFailure(jobs), undefined);
  assert.equal(currentUpdateFailure(JSON.parse(JSON.stringify(jobs))), undefined);
});

test('a failure with no later finished update of the same component is still shown, the last one', () => {
  const a = job('a', 'update-engine', 'error', '2026-11-09T14:10:00Z');
  assert.equal(currentUpdateFailure([a])?.id, 'a');
  assert.equal(currentUpdateFailure([a, job('b', 'update-strata', 'done', '2026-11-09T14:20:00Z')])?.id, 'a', 'another component does not settle it');
  assert.equal(currentUpdateFailure([a, job('b', 'update-engine', 'running', '2026-11-09T14:20:00Z')])?.id, 'a', 'a running retry does not');
  assert.equal(currentUpdateFailure([a, job('b', 'update-engine', 'cancelled', '2026-11-09T14:20:00Z')])?.id, 'a', 'a cancelled retry does not');
  assert.equal(currentUpdateFailure([job('b', 'update-engine', 'done', '2026-11-09T14:00:00Z'), a])?.id, 'a', 'a success before it does not');
  assert.equal(currentUpdateFailure([a, job('c', 'update-strata', 'error', '2026-11-09T14:30:00Z')])?.id, 'c', 'the last failure');
  assert.equal(currentUpdateFailure([a, job('d', 'load', 'error', '2026-11-09T14:30:00Z')])?.id, 'a', 'not an update: not here');
});

test('an interrupted update with no later finished one is still shown in About, with its own words; once a later one finished it is not', async () => {
  const { INTERRUPTED, jobErrorText } = await import(`data:text/javascript;base64,${Buffer.from(out.outputFiles[0].text).toString('base64')}`);
  const cut = { ...job('i', 'update-strata', 'error', '2026-11-09T14:10:00Z'), error: INTERRUPTED };
  const shown = currentUpdateFailure([cut]);
  assert.equal(shown?.id, 'i');
  assert.equal(jobErrorText(shown, [cut]), INTERRUPTED);
  assert.equal(currentUpdateFailure([cut, job('r', 'update-strata', 'running', '2026-11-09T14:20:00Z')])?.id, 'i', 'a retry still running: shown');
  assert.equal(currentUpdateFailure([cut, job('d', 'update-strata', 'done', '2026-11-09T14:20:00Z')]), undefined, 'settled by a later finished update: not in About (Diagnostics keeps it)');
});

test('an update that finished after the check shown asks for one new check; none before it, none without a check', () => {
  const check = '2026-11-09T14:00:00Z';
  assert.equal(updateFinishedAfter([job('u', 'update-strata', 'done', '2026-11-09T14:05:00Z')], check), 'u');
  assert.equal(updateFinishedAfter([job('u', 'update-engine', 'done', '2026-11-09T13:55:00Z', { finishedAt: '2026-11-09T14:02:00Z' })], check), 'u', 'finished after the check');
  assert.equal(updateFinishedAfter([job('u', 'update-engine', 'done', '2026-11-09T13:50:00Z')], check), undefined, 'the check is newer');
  assert.equal(updateFinishedAfter([job('u', 'update-engine', 'error', '2026-11-09T14:05:00Z')], check), undefined, 'a failed update');
  assert.equal(updateFinishedAfter([job('u', 'load', 'done', '2026-11-09T14:05:00Z')], check), undefined, 'not an update');
  assert.equal(updateFinishedAfter([job('u', 'update-engine', 'done', '2026-11-09T14:05:00Z')], undefined), undefined, 'no check shown: nothing (no network on opening Settings)');
});

test('Settings > About uses both: the failure line and the check again, once per finished update', () => {
  const src = readFileSync(fileURLToPath(new URL('../src/components/SettingsUpdates.tsx', import.meta.url)), 'utf8');
  assert.match(src, /currentUpdateFailure\(native\?\.jobs \?\? \[\]\)/);
  assert.doesNotMatch(src, /j\.state === 'error' && \['update-strata', 'update-engine'\]\.includes\(j\.kind\)\)\.slice\(-1\)/);
  assert.match(src, /updateFinishedAfter\(native\?\.jobs \?\? \[\], result\?\.checkedAt\)/);
  assert.match(src, /rechecked\.current !== finished\) \{ rechecked\.current = finished; void check\(\); \}/);
});
