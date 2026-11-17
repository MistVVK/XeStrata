/**
 * QAGUI 0213 #1: one start-up GET that never answers must not leave the app on "Connecting to the local engine" for good.
 * Optional reads (hardware, monitor, skills, ...) fill in later; a hung essential read ends in the offline path within a bounded time.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

globalThis.window ??= globalThis;
globalThis.document ??= { addEventListener() {}, querySelector: () => null, hasFocus: () => true, visibilityState: 'visible', documentElement: { dataset: {}, style: { setProperty() {}, removeProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } } };
const memory = new Map();
globalThis.localStorage = {
  get length() { return memory.size; }, key: (i) => [...memory.keys()][i] ?? null,
  getItem: (k) => memory.get(k) ?? null, setItem: (k, v) => { memory.set(k, v); }, removeItem: (k) => memory.delete(k),
};
globalThis.navigator ??= { language: 'en' };
globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
globalThis.fetch = async () => { throw new Error('offline'); };   // the chat read falls to the browser copy

const answers = {
  '/api/setup': { complete: true, dismissed: false },
  '/api/models': { mode: 'real', variants: [] },
  '/api/engine': { beginner: { contextLength: 8192 }, expert: [] },
  '/api/serving': { state: 'idle' },
  '/api/workspaces': { workspaces: [] },
  '/api/hardware': { gpus: [] },
  '/api/monitor': { cpu: 1 },
  '/api/skills': [],
  '/api/model-providers': { providers: [] },
  '/api/image-models': { models: [] },
  '/api/runtime': { models: [] },
};
globalThis.__hang = new Set();
globalThis.__calls = {};
globalThis.__flaky = {};
const mockClient = {
  name: 'mock-client',
  setup(b) {
    b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
    b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({
      loader: 'js',
      contents: `
        export class ApiError extends Error { constructor(m, status) { super(m); this.status = status; } }
        const none = async () => ({});
        export const api = {
          get: (path) => {
            globalThis.__calls[path] = (globalThis.__calls[path] ?? 0) + 1;
            if (globalThis.__hang.has(path)) return new Promise(() => {});
            if ((globalThis.__flaky[path] ?? 0) > 0) { globalThis.__flaky[path]--; return Promise.reject(new Error('socket hang up')); }
            return Promise.resolve(structuredClone(globalThis.__answers[path]));
          },
          post: none, put: none, patch: none, del: none, delete: none,
        };
        export function onEvent() { return () => {}; }
        export async function streamPost() {}`,
    }));
  },
};
globalThis.__answers = answers;
const result = await build({
  entryPoints: [here('../src/lib/store.ts')], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
  alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' }, plugins: [mockClient],
});
const store = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
Object.assign(store.bootstrapTiming, { timeoutMs: 150, tries: 2, backoffMs: 20 });

const reset = () => { globalThis.__hang.clear(); globalThis.__calls = {}; globalThis.__flaky = {}; };
const snapshot = () => store.getState();

test('an optional GET that never answers does not hold back live; the others still fill in', async () => {
  reset();
  globalThis.__hang.add('/api/skills');
  const t0 = Date.now();
  await store.bootstrap();
  assert.ok(Date.now() - t0 < 140, 'live before the time limit of the hung read');
  assert.equal(snapshot().connection, 'live');
  await new Promise(r => setTimeout(r, 50));
  assert.deepEqual(snapshot().hardware, answers['/api/hardware']);
  assert.equal(globalThis.__calls['/api/skills'], 1, 'the hung read is still waiting for its first answer');
  await new Promise(r => setTimeout(r, 400));
  assert.equal(globalThis.__calls['/api/skills'], 2, 'and is tried again, then given up');
});

test('an essential GET that never answers ends offline within a bounded time', async () => {
  reset();
  globalThis.__hang.add('/api/engine');
  const t0 = Date.now();
  await store.bootstrap();
  const took = Date.now() - t0;
  assert.equal(snapshot().connection, 'offline');
  assert.equal(snapshot().ready, true);
  assert.ok(took < 1500, `bounded (took ${took} ms)`);
  assert.equal(globalThis.__calls['/api/engine'], 2, 'tried the bounded number of times');
});

test('a read that fails once is tried again and the app still goes live', async () => {
  reset();
  globalThis.__flaky['/api/setup'] = 1;
  await store.bootstrap();
  assert.equal(snapshot().connection, 'live');
  assert.equal(globalThis.__calls['/api/setup'], 2);
});
