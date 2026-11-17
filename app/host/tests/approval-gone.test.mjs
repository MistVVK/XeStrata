// #5308: after the assistant stopped answering for 20 minutes the turn ended, but its file-write approval card stayed: every Approve or Deny
// was refused by the host (nothing was waiting any more), the card went back to "needs approval" and asked again, for good. A 409 from the
// host now ends the card with the host's sentence; a failure that may pass (the host away) still puts the question back.
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createChatStore } from '../server/chat-store.mjs';
import { installFakeHostFetch } from './helpers/fake-host-chats.mjs';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
async function bundle(entry, plugins = []) {
  const result = await build({
    entryPoints: [here(entry)], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' }, plugins,
  });
  return import(`data:text/javascript;base64,${Buffer.from(`${result.outputFiles[0].text}
// ${Math.random()}`).toString('base64')}`);
}

const ANSWERS = {
  '/api/setup': { complete: true, dismissed: true },
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

async function pageOn(chatStore, { active = { turns: [] }, replies = {} }) {
  const fake = installFakeHostFetch(chatStore);
  globalThis.window ??= globalThis;
  globalThis.document ??= { addEventListener() {}, querySelector: () => null, hasFocus: () => true, visibilityState: 'visible', documentElement: { dataset: {}, style: { setProperty() {}, removeProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } } };
  const memory = new Map();
  globalThis.localStorage = { get length() { return memory.size; }, key: (i) => [...memory.keys()][i] ?? null, getItem: (k) => memory.get(k) ?? null, setItem: (k, v) => { memory.set(k, v); }, removeItem: (k) => memory.delete(k) };
  globalThis.navigator ??= { language: 'en' };
  globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
  globalThis.location ??= { hash: '' };
  const page = { active, replies, attachCalls: [], events: null, posts: [], postFails: [], waiting: [{ callId: 't1' }] };
  globalThis.__page = page;
  globalThis.__answers = ANSWERS;
  const mockClient = {
    name: 'mock-client',
    setup(b) {
      b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
      b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({ loader: 'js', contents: `
        export class ApiError extends Error { constructor(m, status) { super(m); this.status = status; } }
        globalThis.__page.ApiError = ApiError;
        const none = async () => ({});
        export const api = {
          get: async (path) => {
            if (path === '/api/chat/active') return structuredClone(globalThis.__page.active);
            if (path.startsWith('/api/agent/pending')) return { pending: globalThis.__page.waiting ?? [] };
            if (path in globalThis.__answers) return structuredClone(globalThis.__answers[path]);
            throw new Error('offline');
          },
          post: async (path, b) => { if (path !== '/api/chat/approval') return {}; globalThis.__page.posts.push({ path, b }); const f = globalThis.__page.postFails.shift(); if (f) throw f(); return {}; }, put: none, patch: none, del: none, delete: none,
        };
        export function onEvent(handler, onStatus) { globalThis.__page.events = { handler, onStatus }; return () => {}; }
        export async function streamPost(path, body, onFrame) {
          globalThis.__page.attachCalls.push({ path, body });
          if (path !== '/api/chat/attach') return;
          const reply = globalThis.__page.replies[body.requestId];
          if (!reply) throw new ApiError('{"error":"There is no turn to take over.","code":"no-turn"}', 404);
          reply(onFrame);
        }` }));
    },
  };
  const store = await bundle('../src/lib/store.ts', [mockClient]);
  Object.assign(store.bootstrapTiming, { timeoutMs: 2000, tries: 1, backoffMs: 10 });
  return { store, fake, page };
}

const until = async (check, what, page) => { for (let i = 0; i < 300; i++) { if (await check()) return; await new Promise((r) => setTimeout(r, 10)); } assert.fail(`timed out waiting for ${what} (attach calls: ${JSON.stringify(page?.attachCalls)})`); };

const chatWithApproval = (id) => ({
  id, title: 'Docker', createdAt: '2026-11-16T10:00:00.000Z', updatedAt: '2026-11-16T10:01:00.000Z', workspace: 'C:/w', tokensIn: 0, tokensOut: 0,
  messages: [
    { id: id + '-u', role: 'user', content: 'make an example file', at: '2026-11-16T10:00:00.000Z' },
    { id: id + '-a', role: 'assistant', content: 'I will write it.', at: '2026-11-16T10:00:05.000Z', streaming: false,
      tools: [{ id: 't1', name: 'write', scope: 'write', state: 'proposed', live: true, args: { file_path: 'C:/w/example.txt', content: 'x' }, preview: 'example.txt' }] },
  ],
});
const card = (store, id) => store.getState().sessions.find((s) => s.id === id).messages[1].tools[0];

test('an approval the host no longer waits for (409) ends the card; it does not ask again', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-approval-gone-'));
  const chatStore = createChatStore({ dir });
  let rig;
  try {
    await chatStore.save('s_a', JSON.stringify(chatWithApproval('s_a')));
    rig = await pageOn(chatStore, {});
    await rig.store.bootstrap();
    rig.store.selectSession('s_a');
    assert.equal(card(rig.store, 's_a').state, 'proposed');
    rig.page.postFails.push(() => new rig.page.ApiError('That question is no longer waiting for an answer.', 409));
    rig.store.resolveTool('s_a-a', 't1', true);
    await until(() => card(rig.store, 's_a').state === 'error', 'the card to end', rig.page);
    assert.equal(card(rig.store, 's_a').result, 'That question is no longer waiting for an answer.');
    assert.deepEqual(rig.page.posts.map(p => p.b), [{ sessionId: 's_a', toolId: 't1', decision: 'approve' }]);
    rig.store.resolveTool('s_a-a', 't1', true);
    assert.equal(rig.page.posts.length, 1, 'an ended card is not answered again');
  } finally { rig?.fake.restore(); rmSync(dir, { recursive: true, force: true }); }
});

test('a failure that may pass (the host did not answer) puts the question back to try again', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-approval-retry-'));
  const chatStore = createChatStore({ dir });
  let rig;
  try {
    await chatStore.save('s_b', JSON.stringify(chatWithApproval('s_b')));
    rig = await pageOn(chatStore, {});
    await rig.store.bootstrap();
    rig.store.selectSession('s_b');
    rig.page.postFails.push(() => new TypeError('fetch failed'));
    rig.store.resolveTool('s_b-a', 't1', false);
    await until(() => rig.page.posts.length === 1 && card(rig.store, 's_b').state === 'proposed', 'the question back', rig.page);
    rig.store.resolveTool('s_b-a', 't1', false);
    await until(() => card(rig.store, 's_b').state === 'denied', 'the second answer', rig.page);
    assert.equal(rig.page.posts.length, 2);
  } finally { rig?.fake.restore(); rmSync(dir, { recursive: true, force: true }); }
});
