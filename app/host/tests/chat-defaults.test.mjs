/** Settings → General → "New chats": what a new conversation starts with, and what it must not change. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const here = p => fileURLToPath(new URL(p, import.meta.url));
const PREFS_KEY = 'strata.prefs.v2';
const SESSION_KEY = 'strata.sessions.v1';

let loads = 0;

/**
 * A fresh copy of the store, reading the given localStorage first (the store reads its prefs at import).
 * The bundle is salted: two identical data: URLs are the same module to Node, and the second import
 * would hand back the first copy with its state already changed by an earlier test.
 */
async function load(seed = {}) {
  const memory = new Map(Object.entries(seed));
  globalThis.window ??= globalThis;
  globalThis.document = {
    addEventListener() {}, visibilityState: 'visible',
    querySelector: () => null, querySelectorAll: () => [],
    documentElement: { dataset: {}, style: { setProperty() {}, removeProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } },
  };
  globalThis.localStorage = { getItem: k => memory.get(k) ?? null, setItem: (k, v) => void memory.set(k, v), removeItem: k => void memory.delete(k) };
  globalThis.navigator ??= { language: 'en' };
  globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
  const mockClient = {
    name: 'mock-client',
    setup(b) {
      b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
      b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({
        loader: 'js',
        contents: `export class ApiError extends Error {}
          const reject = async () => { throw new Error('offline'); };
          export const api = { get: reject, post: async () => ({}), put: reject, patch: reject, del: reject, delete: reject };
          export function onEvent() { return () => {}; }
          export async function streamPost() {}`,
      }));
    },
  };
  const result = await build({ entryPoints: [here('../src/lib/store.ts')], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent', alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' }, plugins: [mockClient] });
  const salted = `${result.outputFiles[0].text}\n/* strata-store-copy ${++loads} */\n`;
  const store = await import(`data:text/javascript;base64,${Buffer.from(salted).toString('base64')}`);
  return { store, memory };
}

test('with nothing set, a new chat starts exactly as it did before the setting existed', async () => {
  const { store } = await load();
  const chat = store.newSession();
  assert.deepEqual(chat.preferences.mode, 'manual', 'the composer shows Manual for a chat with no mode');
  assert.deepEqual(chat.preferences.thinking, 'Medium', 'the composer shows Medium for a chat with no effort');
  assert.deepEqual(chat.preferences.thinkingBudget, 0, 'the composer shows No limit for a chat with no budget');
});

test('a chosen default is what the next conversation starts with, and is remembered', async () => {
  const { store, memory } = await load();
  store.setChatDefaults({ mode: 'bypass', thinking: 'High', thinkingBudget: 8192 });
  const chat = store.newSession();
  assert.deepEqual(chat.preferences.mode, 'bypass');
  assert.deepEqual(chat.preferences.thinking, 'High');
  assert.deepEqual(chat.preferences.thinkingBudget, 8192);
  assert.deepEqual(JSON.parse(memory.get(PREFS_KEY)).chatDefaults, { mode: 'bypass', thinking: 'High', thinkingBudget: 8192 }, 'the choice is saved with the other preferences');

  // A second load reads it back from storage, so it survives a restart.
  const again = await load({ [PREFS_KEY]: memory.get(PREFS_KEY), [SESSION_KEY]: memory.get(SESSION_KEY) });
  assert.deepEqual(again.store.newSession().preferences.mode, 'bypass', 'after a restart a new chat still starts in Bypass');
});

test('a default changes new chats only: a conversation already started keeps its own choice', async () => {
  const { store } = await load();
  const before = store.newSession();
  store.setChatDefaults({ mode: 'plan' });
  const after = store.newSession();
  assert.deepEqual(before.preferences.mode, 'manual', 'the chat that existed keeps what it had');
  assert.deepEqual(after.preferences.mode, 'plan', 'the next one takes the new default');
});

test('a stored value the composer cannot offer falls back instead of showing a dead choice', async () => {
  const { store } = await load({ [PREFS_KEY]: JSON.stringify({ chatDefaults: { mode: 'everything-goes', thinking: 'Enormous', thinkingBudget: 999999 } }) });
  const chat = store.newSession();
  assert.deepEqual(chat.preferences.mode, 'manual');
  assert.deepEqual(chat.preferences.thinking, 'Medium');
  assert.deepEqual(chat.preferences.thinkingBudget, 0);
});
