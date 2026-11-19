// #5893: a client that needs the loaded context window (Unsloth Studio sizes its compaction from it) used the engine's private port, whose key
// is new at every load, so every reload broke it. The app's own /v1/models carries the window (PR #4370: meta.n_ctx, the engine's live value
// first) and, here, also as context_length, the field OpenAI-style clients read; the app's API key stays the one the person set.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { nativeModelEntries } from '../server/native-models.mjs';

const instances = [
  { id: 'n1', apiModelId: 'qwen3.8-flash-next', status: 'ready', context: 131072, metrics: { engine: { max_context: 120000 } }, gpus: [0], label: 'Qwen3.8 Flash Next', aliases: ['local'] },
  { id: 'n2', apiModelId: 'other', status: 'loading', context: 32768, gpus: [1], label: 'Other', aliases: [] },
  { id: 'n3', apiModelId: 'configured-only', status: 'ready', context: 65536, gpus: [], label: 'C', aliases: [] },
  { id: 'n4', apiModelId: 'unknown-window', status: 'ready', context: 0, gpus: [], label: 'U', aliases: [] },
];

test('ready models carry their window as context_length and meta.n_ctx; an unknown window is left out', () => {
  const list = nativeModelEntries(instances);
  assert.deepEqual(list.map(m => m.id), ['qwen3.8-flash-next', 'local', 'configured-only', 'unknown-window'], 'a model still loading is not listed; the alias of the ready model rides along as its own row (#6180)');
  const [live, , configured, unknown] = list;
  assert.equal(live.context_length, 120000, 'the engine\'s live window first');
  assert.equal(live.meta.n_ctx, 120000);
  assert.equal(configured.context_length, 65536);
  assert.equal(configured.meta.n_ctx, 65536);
  assert.ok(!('context_length' in unknown) && !('n_ctx' in unknown.meta), 'never reported as 0');
});

test('the host\'s /v1/models answers with these entries', () => {
  const src = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.match(src, /if\(path==='\/v1\/models'&&method==='GET'\)return json\(res,200,\{object:'list',data:nativeModelEntries\(native\.state\(\)\.instances\)\}\);/);
});
