import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const connection = { id: 'conn-1', name: 'Other PC', protocol: 'openai-chat', baseUrl: 'http://192.0.2.10:8080/v1', apiKeySet: false, models: [{ id: 'model-a' }] };

async function loadStore() {
  globalThis.window ??= globalThis;
  globalThis.document ??= { addEventListener() {}, hasFocus: () => true, visibilityState: 'visible', documentElement: { dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } } };
  const memory = new Map();
  globalThis.localStorage = {
    get length() { return memory.size; },
    key: (i) => [...memory.keys()][i] ?? null,
    getItem: (k) => memory.get(k) ?? null,
    setItem: (k, v) => { memory.set(k, v); },
    removeItem: (k) => memory.delete(k),
  };
  globalThis.navigator ??= { language: 'en' };
  globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
  globalThis.__providers = { catalog: [], connections: [connection] };
  const mockClient = {
    name: 'mock-client',
    setup(b) {
      b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
      b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({
        loader: 'js',
        contents: `
          export class ApiError extends Error {}
          const reject = async () => { throw new Error('offline'); };
          export const api = { get: reject, post: async () => globalThis.__providers, put: reject, patch: reject, del: reject, delete: reject };
          export function onEvent() { return () => {}; }
          export async function streamPost() {}`,
      }));
    },
  };
  const result = await build({
    entryPoints: [here('../src/lib/store.ts')], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' }, plugins: [mockClient],
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

test('a new chat opened from a chat on an API model starts on that model', async () => {
  const store = await loadStore();
  await store.saveModelConnection({ kind: 'custom' });
  const first = store.newSession();
  store.selectRemoteModel({ connectionId: 'conn-1', modelId: 'model-a' });
  assert.deepEqual(store.getState().sessions.find(s => s.id === first.id)?.preferences?.remoteModel, { connectionId: 'conn-1', modelId: 'model-a' });

  const second = store.newSession();
  assert.equal(store.getState().activeSession, second.id);
  assert.deepEqual(second.preferences?.remoteModel, { connectionId: 'conn-1', modelId: 'model-a' }, 'the new chat keeps the model of the chat it was opened from');
  assert.deepEqual(store.getState().remoteModel, { connectionId: 'conn-1', modelId: 'model-a' }, 'the composer shows that model');

  store.selectRemoteModel(null);
  assert.deepEqual(store.getState().sessions.find(s => s.id === first.id)?.preferences?.remoteModel, { connectionId: 'conn-1', modelId: 'model-a' }, 'changing the new chat leaves the first chat alone');
  const third = store.newSession();
  assert.equal(third.preferences?.remoteModel ?? null, null, 'a chat opened from a local-model chat stays on the local model');
  store.selectSession(first.id);
  assert.deepEqual(store.getState().remoteModel, { connectionId: 'conn-1', modelId: 'model-a' }, 'back in the first chat, its own model shows');
});
