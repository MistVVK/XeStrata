import test from 'node:test';
import assert from 'node:assert/strict';
import { nativeModelEntries } from '../server/native-models.mjs';
const ready = (patch = {}) => ({ id: 'inst-a', apiModelId: 'qwen-q2_0', status: 'ready', gpus: [0], label: 'Qwen 3.6', aliases: ['qwen-q2_0'], context: 65536, metrics: null, ...patch });

test('the configured context is reported as n_ctx and the other keys are unchanged', () => {
  assert.deepEqual(nativeModelEntries([ready()]), [{ id: 'qwen-q2_0', object: 'model', owned_by: 'strata', meta: { instanceId: 'inst-a', gpus: [0], label: 'Qwen 3.6', aliases: ['qwen-q2_0'], n_ctx: 65536 } }]);
});

test('the engine-reported window wins over the config', () => {
  const [entry] = nativeModelEntries([ready({ context: 65536, metrics: { engine: { max_context: 32768 } } })]);
  assert.equal(entry.meta.n_ctx, 32768);
});

test('an unknown engine value falls back to the config', () => {
  for (const max_context of [0, null, undefined, NaN, -1, '32768']) {
    const [entry] = nativeModelEntries([ready({ context: 65536, metrics: { engine: { max_context } } })]);
    assert.equal(entry.meta.n_ctx, 65536, `max_context=${String(max_context)}`);
  }
});

test('unknown context is left out, not 0', () => {
  for (const context of [0, undefined, null, NaN, Infinity, -4096, 1.5, '65536']) {
    const [entry] = nativeModelEntries([ready({ context, metrics: null })]);
    assert.equal('n_ctx' in entry.meta, false, `context=${String(context)}`);
    assert.deepEqual(entry.meta, { instanceId: 'inst-a', gpus: [0], label: 'Qwen 3.6', aliases: ['qwen-q2_0'] });
  }
  const [bothUnknown] = nativeModelEntries([ready({ context: 0, metrics: { engine: { max_context: 0 } } })]);
  assert.equal('n_ctx' in bothUnknown.meta, false);
});

test('only ready instances are listed, in input order', () => {
  const result = nativeModelEntries([
    ready({ id: 'a', apiModelId: 'm-a' }),
    ready({ id: 'b', apiModelId: 'm-b', status: 'loading' }),
    ready({ id: 'c', apiModelId: 'm-c', status: 'error' }),
    ready({ id: 'd', apiModelId: 'm-d', status: 'unloading' }),
    ready({ id: 'e', apiModelId: 'm-e', context: 131072 }),
  ]);
  assert.deepEqual(result.map(e => e.id), ['m-a', 'm-e']);
  assert.equal(result[1].meta.n_ctx, 131072);
});

test('empty input', () => {
  assert.deepEqual(nativeModelEntries([]), []);
  assert.deepEqual(nativeModelEntries(undefined), []);
  assert.deepEqual(nativeModelEntries(null), []);
});

test('the input is not changed', () => {
  const input = [ready({ metrics: { engine: { max_context: 32768 } } })];
  const copy = structuredClone(input);
  nativeModelEntries(input);
  assert.deepEqual(input, copy);
});

test('n_ctx is the only key added to meta', () => {
  const [entry] = nativeModelEntries([ready()]);
  assert.deepEqual(Object.keys(entry).sort(), ['id', 'meta', 'object', 'owned_by']);
  assert.deepEqual(Object.keys(entry.meta).sort(), ['aliases', 'gpus', 'instanceId', 'label', 'n_ctx']);
});
