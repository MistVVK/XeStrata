// #5548: the preview panel opened again at every start, because closing it kept the chat's saved previewOpen. Closing the panel now forgets it
// (the tabs stay); picking a tab remembers it again. Driven through the page's real store over a host chat store, then a fresh page load.
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
  const page = { active, replies, attachCalls: [], events: null };
  globalThis.__page = page;
  globalThis.__answers = ANSWERS;
  const mockClient = {
    name: 'mock-client',
    setup(b) {
      b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
      b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({ loader: 'js', contents: `
        export class ApiError extends Error { constructor(m, status) { super(m); this.status = status; } }
        const none = async () => ({});
        export const api = {
          get: async (path) => {
            if (path === '/api/chat/active') return structuredClone(globalThis.__page.active);
            if (path.startsWith('/api/agent/pending')) return { pending: [] };
            if (path in globalThis.__answers) return structuredClone(globalThis.__answers[path]);
            throw new Error('offline');
          },
          post: none, put: none, patch: none, del: none, delete: none,
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

const chatWithPreview = (id) => ({
  id, title: 'Site', createdAt: '2026-11-16T10:00:00.000Z', updatedAt: '2026-11-16T10:01:00.000Z', workspace: 'C:/w', tokensIn: 0, tokensOut: 0, previewOpen: 'a1',
  messages: [
    { id: id + '-u', role: 'user', content: 'make a page', at: '2026-11-16T10:00:00.000Z' },
    { id: id + '-a', role: 'assistant', content: 'Here it is.', at: '2026-11-16T10:00:05.000Z', streaming: false, artifacts: [{ id: 'a1', kind: 'html', title: 'index.html', content: '<p>hi</p>' }] },
  ],
});
const saved = async (chatStore, id) => { const d = JSON.parse(await chatStore.read(id)); return d.session ?? d; };

test('closing the preview panel is remembered, so the next start does not open it; picking a tab opens and remembers it again', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-preview-closed-'));
  const chatStore = createChatStore({ dir });
  let rig, again;
  try {
    await chatStore.save('s_p', JSON.stringify(chatWithPreview('s_p')));
    rig = await pageOn(chatStore, {});
    await rig.store.bootstrap();
    const session = () => rig.store.getState().sessions.find((s) => s.id === 's_p');
    assert.equal(session().previewOpen, 'a1', 'the saved chat had the panel open');
    rig.store.setPreviewPanelClosed('s_p');
    assert.equal(session().previewOpen, undefined);
    assert.deepEqual(session().closedPreviews ?? [], [], 'the tab itself is not closed');
    await until(async () => (await saved(chatStore, 's_p')).previewOpen === undefined, 'the closed panel saved', rig.page);
    rig.fake.restore();
    again = await pageOn(chatStore, {});
    await again.store.bootstrap();
    assert.equal(again.store.getState().sessions.find((s) => s.id === 's_p').previewOpen, undefined, 'the next start leaves the panel closed');
    again.store.setPreviewTabClosed('s_p', 'a1', false, 'a1');
    assert.equal(again.store.getState().sessions.find((s) => s.id === 's_p').previewOpen, 'a1');
  } finally { rig?.fake.restore(); again?.fake.restore(); rmSync(dir, { recursive: true, force: true }); }
});

test('every way the Chat screen closes the panel goes through closePreviewPanel', () => {
  const chat = readFileSync(here('../src/screens/Chat.tsx'), 'utf8');
  assert.match(chat, /function closePreviewPanel\(\) \{ setPreviewPanelClosed\(session\?\.id\); \}/);
  const closes = chat.match(/onClose=\{\(\) => \{? ?setSelectedPreview\(null\)[^}]*\}?\}/g) ?? [];
  assert.equal(closes.length, 5);
  for (const c of closes) assert.match(c, /closePreviewPanel\(\)/, c);
});
