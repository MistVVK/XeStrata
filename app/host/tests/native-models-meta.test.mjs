/* #6180: the native /v1/models route must mirror the engine's metadata —
 * meta.n_ctx, status, architecture, and alias entries — so OpenAI-compatible
 * clients can auto-detect the context window instead of guessing it. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { nativeModelEntries } from '../server/native-models.mjs';

// One ready native instance exactly as native-control's state() reports it:
// metrics from the engine's /metrics (engine.max_context), vision on,
// two aliases of which one is the model tag itself.
function instance(over = {}) {
  return {
    id: 'e02a266a-1111-2222-3333-444455556666',
    apiModelId: 'swift-iq3_s',
    model: 'swift-1.5-iq3_s',
    label: 'Swift 1.5 IQ3_S',
    context: 262144,
    gpus: [1],
    vision: true,
    aliases: ['Strata', 'swift-iq3_s'],
    status: 'ready',
    loadedAt: '2026-11-17T22:27:55Z',
    metrics: { engine: { max_context: 262144 }, live: { state: 'idle' } },
    ...over,
  };
}

test('native /v1/models rows carry the engine-side truth (#6180)', () => {
  const rows = nativeModelEntries([instance()]);
  const canonical = rows.find(r => r.id === 'swift-iq3_s');
  assert.ok(canonical, 'the canonical apiModelId row exists');
  assert.equal(canonical.alias_of, undefined, 'the canonical row is not an alias of anything');
  assert.deepEqual(canonical.status, { value: 'loaded' });
  assert.deepEqual(canonical.architecture, { input_modalities: ['text', 'image'], output_modalities: ['text'] });
  assert.equal(canonical.meta.n_ctx, 262144, 'n_ctx is the engine window, not a guess');
  assert.equal(canonical.meta.instanceId, 'e02a266a-1111-2222-3333-444455556666');
  assert.deepEqual(canonical.meta.gpus, [1]);
  assert.equal(canonical.meta.label, 'Swift 1.5 IQ3_S');
  assert.deepEqual(canonical.meta.aliases, ['Strata', 'swift-iq3_s']);
});

test('alias entries are full rows with alias_of; the canonical tag alias does not duplicate', () => {
  const rows = nativeModelEntries([instance()]);
  const ids = rows.map(r => r.id);
  assert.deepEqual(ids, ['swift-iq3_s', 'Strata'], 'only the distinct alias is added');
  const alias = rows.find(r => r.id === 'Strata');
  assert.equal(alias.alias_of, 'swift-iq3_s');
  assert.equal(alias.meta.n_ctx, 262144, 'alias rows mirror the engine: same full entry');
  assert.deepEqual(alias.architecture, { input_modalities: ['text', 'image'], output_modalities: ['text'] });
});

test('n_ctx falls back to the load-time context, and is omitted when neither is known', () => {
  const noMetrics = nativeModelEntries([instance({ metrics: undefined })]);
  assert.equal(noMetrics[0].meta.n_ctx, 262144, 'context field covers a metrics gap');
  const blind = nativeModelEntries([instance({ metrics: undefined, context: undefined })]);
  assert.equal('n_ctx' in blind[0].meta, false, 'no number is invented when the window is unknown');
});

test('a text-only model advertises text input only', () => {
  const rows = nativeModelEntries([instance({ vision: false, aliases: [] })]);
  assert.equal(rows.length, 1, 'no aliases means exactly one row');
  assert.deepEqual(rows[0].architecture.input_modalities, ['text']);
});

test('every ready instance gets its own rows', () => {
  const rows = nativeModelEntries([
    instance({ id: 'one', apiModelId: 'swift-iq3_s', aliases: ['Strata'] }),
    instance({ id: 'two', apiModelId: 'qwen-iq2_xs', model: 'qwen', label: 'Qwen', vision: false, aliases: [], metrics: { engine: { max_context: 32768 } }, context: 32768 }),
  ]);
  assert.deepEqual(rows.map(r => r.id), ['swift-iq3_s', 'Strata', 'qwen-iq2_xs']);
  assert.equal(rows.find(r => r.id === 'qwen-iq2_xs').meta.n_ctx, 32768);
});

test('the native route in index.mjs uses nativeModelEntries instead of the stripped map', async () => {
  const src = await readFile(fileURLToPath(new URL('../server/index.mjs', import.meta.url)), 'utf8');
  const route = src.split('\n').find(l => l.includes("if(path==='/v1/models'&&method==='GET')"));
  assert.ok(route, 'the native /v1/models route line is found');
  assert.match(route, /nativeModelEntries/, 'the route delegates to the shared mapper');
  assert.doesNotMatch(route, /owned_by:'strata',meta:\{instanceId/, 'the old inline stripped map is gone');
});
