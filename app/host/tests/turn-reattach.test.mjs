/**
 * A reload does not stop an agent turn. The page that comes back takes the running turn over again: it shows as running (the Stop button), the
 * rest of the answer streams in, the tool card that waits for a yes is there to answer, and the failure the page wrote when it lost the stream
 * goes. A turn that ended while the page was away is shown whole. Nothing to take over leaves the reply as it was.
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
  // A different text each time: the same text would give the same module, with the first test's state in it.
  return import(`data:text/javascript;base64,${Buffer.from(`${result.outputFiles[0].text}
// ${Math.random()}`).toString('base64')}`);
}

const NETWORK = 'Network error. Try sending the message again.';
const chatWithCutOffReply = (id, requestId, extra = {}) => ({
  id, title: `Chat ${id}`, createdAt: '2026-11-04T10:00:00.000Z', updatedAt: '2026-11-04T10:01:00.000Z', workspace: 'C:/w', tokensIn: 0, tokensOut: 0,
  messages: [
    { id: `${id}-u`, role: 'user', content: 'build the thing', at: '2026-11-04T10:00:00.000Z' },
    { id: `${id}-a`, role: 'assistant', content: 'I will start by', at: '2026-11-04T10:00:05.000Z', requestId, streaming: false, error: NETWORK, ...extra },
  ],
});

async function pageOn(chatStore, { active = [], attach }) {
  const fake = installFakeHostFetch(chatStore);
  globalThis.window ??= globalThis;
  globalThis.document ??= { addEventListener() {}, hasFocus: () => true, visibilityState: 'visible', documentElement: { dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } } };
  const memory = new Map();
  globalThis.localStorage = { get length() { return memory.size; }, key: (i) => [...memory.keys()][i] ?? null, getItem: (k) => memory.get(k) ?? null, setItem: (k, v) => { memory.set(k, v); }, removeItem: (k) => memory.delete(k) };
  globalThis.navigator ??= { language: 'en' };
  globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
  globalThis.__attach = attach;
  globalThis.__active = active;
  globalThis.__attachCalls = [];
  const mockClient = {
    name: 'mock-client',
    setup(b) {
      b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
      b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({ loader: 'js', contents: `
        export class ApiError extends Error { constructor(m, status, body) { super(m); this.status = status; this.body = body; } }
        const reject = async () => { throw new Error('offline'); };
        export const api = { get: async (path) => { if (path === '/api/chat/active') return { turns: globalThis.__active }; if (path.startsWith('/api/agent/pending')) return { pending: [] }; throw new Error('offline'); }, post: async () => ({}), put: reject, patch: reject, del: reject, delete: reject };
        export function onEvent() { return () => {}; }
        export async function streamPost(path, body, onFrame, signal) {
          globalThis.__attachCalls.push({ path, body });
          if (path !== '/api/chat/attach') return;
          await globalThis.__attach(body, onFrame, signal, ApiError);
        }` }));
    },
  };
  const store = await bundle('../src/lib/store.ts', [mockClient]);
  return { store, fake };
}
const until = async (check, what) => { for (let i = 0; i < 300; i++) { if (check()) return; await new Promise((r) => setTimeout(r, 10)); } assert.fail(`timed out waiting for ${what} (attach calls: ${JSON.stringify(globalThis.__attachCalls)})`); };

test('reload in the middle of an agent turn: the chat shows as running with a Stop button, streams the rest, keeps the approval card, and ends whole', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-reattach-'));
  const chatStore = createChatStore({ dir });
  let page;
  try {
    // What the page saved before it went away: the question, a cut-off reply, the failure it wrote when the stream broke.
    await chatStore.save('s_run', JSON.stringify(chatWithCutOffReply('s_run', 'req-run')));
    let release;
    const finished = new Promise((r) => { release = r; });
    page = await pageOn(chatStore, {
      active: [{ chatId: 's_run', sessionId: 'sess-1', requestId: 'req-run', mode: 'manual', pending: [{ callId: 'call-9' }] }],
      attach: async (body, onFrame) => {
        assert.deepEqual(body, { sessionId: 's_run' }, 'a running turn is asked for by chat only');
        onFrame({ type: 'start', model: 'm', contextMax: 8192, agent: true, resumed: true, requestId: 'req-run', sessionId: 'sess-1', mode: 'manual', skills: [] });
        onFrame({ type: 'resync', content: 'I will start by reading the folder, then', reasoning: 'thinking so far' });
        onFrame({ type: 'agent-tool', tool: { id: 'call-9', name: 'write', verb: 'Write', target: 'out.txt', args: {}, state: 'proposed', effect: 'Needs your approval.' } });
        await finished;   // the turn goes on after the page is back
        onFrame({ type: 'delta', text: ' writing the file.' });
        onFrame({ type: 'agent-tool', tool: { id: 'call-9', name: 'write', verb: 'Write', target: 'out.txt', args: {}, state: 'done', result: 'wrote out.txt' } });
        onFrame({ type: 'done', tokensIn: 10, tokensOut: 20, durationMs: 684000, usageSource: 'reported' });
      },
    });
    const { store } = page;
    await store.loadChatsFromHost();
    await until(() => page.fake.calls.length && globalThis.__attachCalls.length === 1, 'the attach request');
    store.selectSession('s_run');
    const chat = () => store.useStore((s) => s.sessions).find((s) => s.id === 's_run');
    const reply = () => chat().messages.at(-1);

    // running again: this is what shows the Stop button (the page treats a chat with a live turn as streaming)
    assert.equal(store.useStore((s) => s.streaming), true, 'the Stop button is there');
    assert.equal(reply().streaming, true, 'the reply is running');
    await until(() => reply().content.startsWith('I will start by reading the folder'), 'the catch-up text');
    assert.equal(reply().error, undefined, 'the failure the page wrote is gone');
    assert.equal(reply().reasoning, 'thinking so far');
    const card = reply().tools.find((t) => t.id === 'call-9');
    assert.equal(card.state, 'proposed', 'the approval waiting on the host is there to answer');
    assert.equal(chat().messages.length, 2, 'no second reply was added');

    // the rest of the turn streams into the same reply
    release();
    await until(() => reply().streaming === false, 'the end of the turn');
    assert.equal(reply().content, 'I will start by reading the folder, then writing the file.');
    assert.equal(reply().tools.find((t) => t.id === 'call-9').state, 'done');
    assert.equal(store.useStore((s) => s.streaming), false, 'the Stop button goes away when the turn ends');
    await store.flushSessions();
    const saved = JSON.parse(await chatStore.read('s_run')).session.messages.at(-1);
    assert.equal(saved.content, 'I will start by reading the folder, then writing the file.', 'and the whole answer is what is saved');
    assert.equal(saved.streaming, false);
  } finally { page?.fake.restore(); rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }); }
});

test('a turn that ended while the page was away is shown whole; a chat with nothing to take over keeps its reply and gets no error', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-reattach2-'));
  const chatStore = createChatStore({ dir });
  let page;
  try {
    await chatStore.save('s_done', JSON.stringify(chatWithCutOffReply('s_done', 'req-done')));
    await chatStore.save('s_gone', JSON.stringify(chatWithCutOffReply('s_gone', 'req-gone', { error: 'Response interrupted. Ask again to retry; the partial answer is kept.' })));
    page = await pageOn(chatStore, {
      active: [],
      attach: async (body, onFrame, _signal, ApiError) => {
        if (body.requestId === 'req-done') {
          onFrame({ type: 'start', model: 'm', contextMax: 8192, agent: true, resumed: true, requestId: 'req-done', sessionId: 'sess-2', skills: [] });
          onFrame({ type: 'resync', content: 'I will start by reading the folder. Done: 3 files changed.', reasoning: '' });
          onFrame({ type: 'agent-tool', tool: { id: 'c1', name: 'write', verb: 'Write', target: 'a.txt', args: {}, state: 'done', result: 'ok' } });
          onFrame({ type: 'done', tokensIn: 1, tokensOut: 2, durationMs: 5000, usageSource: 'reported' });
          return;
        }
        throw new ApiError('{"error":"There is no turn to take over.","code":"no-turn"}', 404);   // the host holds nothing for this reply
      },
    });
    const { store } = page;
    await store.loadChatsFromHost();
    await until(() => globalThis.__attachCalls.length >= 2, 'both asked about');
    const sessions = () => store.useStore((s) => s.sessions);
    await until(() => sessions().find((s) => s.id === 's_done').messages.at(-1).usageSource === 'reported', 'the finished turn');
    const done = sessions().find((s) => s.id === 's_done').messages.at(-1);
    assert.equal(done.content, 'I will start by reading the folder. Done: 3 files changed.');
    assert.equal(done.error, undefined);
    assert.equal(done.streaming, false);
    assert.equal(done.tools[0].state, 'done');
    await until(() => !store.useStore((s) => s.sessions).find((s) => s.id === 's_gone').messages.at(-1).streaming, 'the unanswered one settles');
    const gone = sessions().find((s) => s.id === 's_gone').messages.at(-1);
    assert.equal(gone.content, 'I will start by', 'its reply is as the page had it');
    assert.equal(gone.error, 'Response interrupted. Ask again to retry; the partial answer is kept.', 'its error stays');
    assert.equal(store.useStore((s) => s.toasts).length, 0, 'no failure is announced');
    assert.deepEqual(globalThis.__attachCalls.map((c) => c.body.requestId).sort(), ['req-done', 'req-gone'], JSON.stringify(globalThis.__attachCalls));
    assert.equal(store.useStore((s) => s.streaming), false);
    await store.flushSessions();
  } finally { page?.fake.restore(); rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }); }
});
