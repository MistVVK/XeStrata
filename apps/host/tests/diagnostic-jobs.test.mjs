// #5489: Settings -> Engine -> Diagnostics kept "UD-IQ4_XS-v31 needs engine 0.1.57 or newer; this one is 0.1.56" with engine 0.1.58 in use,
// also after a restart (the job history is kept, #5802). A failure is shown only while nothing later has settled it.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const built = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/diagnostic-jobs.ts', import.meta.url))], bundle: true, write: false, platform: 'node', format: 'esm' });
const { openFailures } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

let t = 0;
const job = (kind, state, extra = {}) => ({ id: `j${++t}`, kind, state, startedAt: new Date(Date.UTC(2026, 10, 16, 10, 0, t)).toISOString(), summary: '', error: state === 'error' ? `${kind} failed` : null, result: null, ...extra });
const ids = list => openFailures(list).map(j => j.id);

test('the reported case: a load that needed a newer engine is gone once the engine was updated', () => {
  const failed = job('load', 'error', { model: 'ud-iq4_xs-v31', error: 'UD-IQ4_XS-v31 needs engine 0.1.57 or newer; this one is 0.1.56' });
  assert.deepEqual(ids([failed]), [failed.id], 'shown while nothing settled it');
  assert.deepEqual(ids([failed, job('update-engine', 'done')]), [], 'a later engine update settles it');
  assert.deepEqual(ids([failed, job('update-engine', 'error')]), [failed.id, `j${t}`], 'a failed update settles nothing');
});

test('a later success for the same model settles its failures; other models stay', () => {
  const a = job('install', 'error', { model: 'a' }), b = job('load', 'error', { model: 'b' });
  assert.deepEqual(ids([a, b, job('load', 'done', { model: 'a' })]), [b.id]);
});

test('jobs without a model are settled by a later success of the same kind', () => {
  const tune = job('calibrate', 'error'), bench = job('benchmark', 'error');
  assert.deepEqual(ids([tune, bench, job('calibrate', 'done')]), [bench.id]);
});

test('every engine change settles all earlier failures; a failure after it stays', () => {
  for (const kind of ['update-engine', 'rollback', 'install-engine-file', 'update-strata']) {
    const before = job('load', 'error', { model: 'x' }), change = job(kind, 'done'), after = job('load', 'error', { model: 'y' });
    assert.deepEqual(ids([before, change, after]), [after.id], kind);
  }
});

test('at most three, newest last; running and cancelled jobs settle nothing; order is by start time', () => {
  const fails = [1, 2, 3, 4].map(i => job('load', 'error', { model: `m${i}` }));
  assert.deepEqual(ids(fails), fails.slice(-3).map(j => j.id));
  assert.deepEqual(ids([fails[0], job('load', 'cancelled', { model: 'm1' }), job('load', 'running', { model: 'm1' })]), [fails[0].id]);
  const update = job('update-engine', 'done'), later = job('load', 'error', { model: 'z' });
  assert.deepEqual(ids([later, update]), [later.id], 'sorted by startedAt, not by array order');
  assert.deepEqual(openFailures(undefined), []);
});

test('the Diagnostics card lists openFailures, not every failed job', () => {
  const src = readFileSync(fileURLToPath(new URL('../src/components/NativeOperations.tsx', import.meta.url)), 'utf8');
  assert.match(src, /openFailures\(native\?\.jobs\)\.map\(j => <ErrorNotice key=\{j\.id\}/);
  assert.doesNotMatch(src, /jobs\.filter\(j => j\.state === 'error'\)\.slice\(-3\)/);
});
