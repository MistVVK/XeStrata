/**
 * The page's side of a reply the agent writes by itself after its turn ended (a background command finished): the host's `chat-wake` event, the
 * same reply asked for again after the event stream came back, and a page that loads while such a turn runs. Each arrives as a new reply under
 * the one before, never appended to it.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { mkdtempSync, rmSync } from 'node:fs';
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

const answeredChat = (id, extra = {}) => ({
  id, title: `Chat ${id}`, createdAt: '2026-11-11T10:00:00.000Z', updatedAt: '2026-11-11T10:01:00.000Z', workspace: 'C:/w', tokensIn: 0, tokensOut: 0,
  messages: [
    { id: `${id}-u`, role: 'user', content: 'run the tests in the background', at: '2026-11-11T10:00:00.000Z' },
    { id: `${id}-a`, role: 'assistant', content: 'Started the job in the background.', at: '2026-11-11T10:00:05.000Z', requestId: 'req-1', streaming: false, usageSource: 'reported', ...extra },
  ],
});

/** A reply the agent wrote by itself, as /api/chat/attach sends it. */
function wakeReply(requestId, text) {
  return (onFrame) => {
    onFrame({ type: 'start', model: 'm', contextMax: 8192, agent: true, resumed: true, requestId, sessionId: 'sess-1', mode: 'manual', skills: [] });
    onFrame({ type: 'resync', content: text, reasoning: '' });
    onFrame({ type: 'done', tokensIn: 1, tokensOut: 2, durationMs: 900, usageSource: 'reported' });
  };
}

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

const until = async (check, what, page) => { for (let i = 0; i < 300; i++) { if (check()) return; await new Promise((r) => setTimeout(r, 10)); } assert.fail(`timed out waiting for ${what} (attach calls: ${JSON.stringify(page?.attachCalls)})`); };

test('a chat-wake event from the host adds the reply the agent wrote by itself as a new message', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-wake-ui-'));
  const chatStore = createChatStore({ dir });
  let rig;
  try {
    await chatStore.save('s_bg', JSON.stringify(answeredChat('s_bg')));
    rig = await pageOn(chatStore, { replies: { 'wake-1': wakeReply('wake-1', 'The job finished: 12 tests pass.') } });
    const { store, page } = rig;
    await store.bootstrap();
    assert.equal(store.getState().connection, 'live');
    const messages = () => store.getState().sessions.find((s) => s.id === 's_bg').messages;
    assert.equal(messages().length, 2);

    page.events.handler({ event: 'chat-wake', data: { chatId: 's_bg', requestId: 'wake-1' } });
    await until(() => messages().length === 3 && messages().at(-1).streaming === false, 'the new reply', page);
    assert.deepEqual(page.attachCalls.map((c) => c.body), [{ sessionId: 's_bg', requestId: 'wake-1' }]);
    assert.equal(messages()[1].content, 'Started the job in the background.', 'the reply before is untouched');
    assert.equal(messages().at(-1).role, 'assistant');
    assert.equal(messages().at(-1).content, 'The job finished: 12 tests pass.');
    assert.equal(messages().at(-1).requestId, 'wake-1');
  } finally { rig?.fake.restore(); rmSync(dir, { recursive: true, force: true }); }
});

test('a reply the agent began while the event stream was down is asked for when the stream comes back', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-wake-ui-'));
  const chatStore = createChatStore({ dir });
  let rig;
  try {
    await chatStore.save('s_bg', JSON.stringify(answeredChat('s_bg')));
    rig = await pageOn(chatStore, { replies: { 'wake-2': wakeReply('wake-2', 'The build finished without errors.') } });
    const { store, page } = rig;
    await store.bootstrap();
    const messages = () => store.getState().sessions.find((s) => s.id === 's_bg').messages;
    assert.equal(page.attachCalls.length, 0, 'nothing to take over at first');

    // The stream drops; the host announces the wake to no one, and holds the finished reply.
    page.events.onStatus('offline');
    page.active = { turns: [], woken: [{ chatId: 's_bg', requestId: 'wake-2' }] };
    page.events.onStatus('live');

    await until(() => messages().length === 3 && messages().at(-1).streaming === false, 'the reply after the reconnect', page);
    assert.deepEqual(page.attachCalls.map((c) => c.body), [{ sessionId: 's_bg', requestId: 'wake-2' }]);
    assert.equal(messages().at(-1).content, 'The build finished without errors.');
    assert.equal(messages()[1].content, 'Started the job in the background.');
  } finally { rig?.fake.restore(); rmSync(dir, { recursive: true, force: true }); }
});

test('a page that loads while such a turn runs shows it as a new reply; one it already started goes on in the same reply', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-wake-ui-'));
  const chatStore = createChatStore({ dir });
  let rig;
  try {
    await chatStore.save('s_new', JSON.stringify(answeredChat('s_new')));
    // This page had already begun the wake reply before it reloaded: the last reply carries its request id.
    const begun = answeredChat('s_old');
    begun.messages.push({ id: 's_old-w', role: 'assistant', content: 'The job fin', at: '2026-11-11T10:02:00.000Z', requestId: 'wake-old', streaming: false });
    await chatStore.save('s_old', JSON.stringify(begun));
    rig = await pageOn(chatStore, {
      active: { turns: [{ chatId: 's_new', requestId: 'wake-new', wake: true }, { chatId: 's_old', requestId: 'wake-old', wake: true }] },
      replies: { 'wake-new': wakeReply('wake-new', 'The job finished: 3 files changed.'), 'wake-old': wakeReply('wake-old', 'The job finished: 7 warnings.') },
    });
    const { store, page } = rig;
    await store.bootstrap();
    const messages = (id) => store.getState().sessions.find((s) => s.id === id).messages;

    await until(() => messages('s_new').at(-1).content === 'The job finished: 3 files changed.' && !messages('s_new').at(-1).streaming, 'the new reply', page);
    assert.equal(messages('s_new').length, 3, 'a new reply, not the end of the one before');
    assert.equal(messages('s_new')[1].content, 'Started the job in the background.');

    await until(() => messages('s_old').at(-1).content === 'The job finished: 7 warnings.' && !messages('s_old').at(-1).streaming, 'the begun reply', page);
    assert.equal(messages('s_old').length, 3, 'the reply it had begun goes on, nothing added');
    assert.equal(messages('s_old').at(-1).id, 's_old-w');

    assert.deepEqual(page.attachCalls.map((c) => c.body).sort((a, b) => a.sessionId.localeCompare(b.sessionId)),
      [{ sessionId: 's_new', requestId: 'wake-new' }, { sessionId: 's_old', requestId: 'wake-old' }]);
  } finally { rig?.fake.restore(); rmSync(dir, { recursive: true, force: true }); }
});
