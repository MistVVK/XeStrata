import test from 'node:test';
import assert from 'node:assert/strict';
import { apply as applyContextPlugin } from '../server/agent/plugins/strata-context.mjs';
import { localOutputBudget } from '../server/agent/profile.mjs';

async function localRequestStream(contextWindow) {
  const provider = 'strata-local';
  const maxTokens = localOutputBudget(contextWindow);
  const adapter = {
    modelOf: () => ({ contextWindow, maxTokens }),
    streamWithSnapshot: options => options,
  };
  const llm = {
    adapters: new Map([[provider, { adapter }]]),
    prepareCall: async () => ({ stream: options => adapter.streamWithSnapshot(options) }),
  };
  applyContextPlugin({ llm, inject: () => {} });
  const { stream } = await llm.prepareCall({ provider });
  return {
    maxTokens,
    request: options => stream({ provider, model: 'test-model', ...options }),
  };
}

test('the local reply ceiling still expands the profile default at 512K', async () => {
  const { maxTokens, request } = await localRequestStream(524288);
  assert.equal(maxTokens, 32768);
  assert.equal(request({ maxTokens }).maxTokens, 65536);
  assert.equal(request({}).maxTokens, 65536);
});

test('auxiliary requests and distinct explicit limits keep their output budgets', async () => {
  const { maxTokens, request } = await localRequestStream(65536);
  assert.equal(request({ maxTokens }).maxTokens, 32768);
  assert.equal(request({ maxTokens, purpose: 'compaction' }).maxTokens, maxTokens);
  assert.equal(request({ maxTokens, purpose: 'session-title' }).maxTokens, maxTokens);
  assert.equal(request({ maxTokens: 16384 }).maxTokens, 16384);
});
