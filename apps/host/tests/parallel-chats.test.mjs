/**
 * #5008: two chats, one workspace, a tool that writes a unique marker.
 * A folder shared by two chats is shared on purpose; what a chat shows as its own (messages, tool cards, "files this turn made") is private.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DeliverableTracker, FolderClaims } from '../server/agent/deliverables.mjs';
import { createChatStore } from '../server/chat-store.mjs';
import { installFakeHostFetch } from './helpers/fake-host-chats.mjs';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

/* ---------------------------------------------------------------- host side: files a turn made */

/** A "tool" of a chat: writes a file with a unique marker, then asks the chat's tracker about it (what the bridge does on a write/edit card). */
function tool(dir, tracker, name, marker) {
  writeFileSync(join(dir, name), `marker:${marker}\n`);
  return tracker.check(name, { explicit: true });
}

test('two chats in one folder: each turn reports only the files its own tools wrote, interleaved, and a script file nobody can own is withheld', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-5008-'));
  try {
    const claims = new FolderClaims();
    const A = new DeliverableTracker(dir, { claims, owner: 'chat-A' });
    const B = new DeliverableTracker(dir, { claims, owner: 'chat-B' });
    A.snapshot(); claims.begin(dir, 'chat-A');
    B.snapshot(); claims.begin(dir, 'chat-B');
    const reportedA = [], reportedB = [];
    // alternate marked turns
    reportedA.push(tool(dir, A, 'a-1.txt', 'A-one'));
    reportedB.push(tool(dir, B, 'b-1.txt', 'B-one'));
    reportedA.push(tool(dir, A, 'a-2.txt', 'A-two'));
    reportedB.push(tool(dir, B, 'b-2.txt', 'B-two'));
    mkdirSync(join(dir, 'out')); writeFileSync(join(dir, 'out', 'script-made.csv'), 'x,y\n1,2\n');   // a script's output, while both run
    // A ends first: its sweep must not claim B's files
    const sweptA = A.sweep(); claims.end(dir, 'chat-A');
    const sweptB = B.sweep(); claims.end(dir, 'chat-B');
    assert.deepEqual(reportedA.map(d => d.path), ['a-1.txt', 'a-2.txt']);
    assert.deepEqual(reportedB.map(d => d.path), ['b-1.txt', 'b-2.txt']);
    assert.deepEqual(sweptA.map(d => d.path), [], 'chat A is not given chat B\'s files, nor a file whose writer cannot be told');
    assert.deepEqual(sweptB.map(d => d.path), [], 'and the other way round');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a chat working alone still gets the files its scripts made (nothing changes without a second chat)', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-5008b-'));
  try {
    const claims = new FolderClaims();
    const A = new DeliverableTracker(dir, { claims, owner: 'chat-A' });
    A.snapshot(); claims.begin(dir, 'chat-A');
    writeFileSync(join(dir, 'built.js'), 'x\n');
    assert.deepEqual(A.sweep().map(d => d.path), ['built.js']);
    claims.end(dir, 'chat-A');
    // a second chat that starts after the first has ended is not overlapped with it
    const B = new DeliverableTracker(dir, { claims, owner: 'chat-B' });
    B.snapshot(); claims.begin(dir, 'chat-B');
    writeFileSync(join(dir, 'later.js'), 'y\n');
    assert.deepEqual(B.sweep().map(d => d.path), ['later.js']);
    claims.end(dir, 'chat-B');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a chat that is stopped or fails still lets go of the folder, and another folder is never affected', () => {
  const one = mkdtempSync(join(tmpdir(), 'strata-5008c-')), two = mkdtempSync(join(tmpdir(), 'strata-5008d-'));
  try {
    const claims = new FolderClaims();
    const A = new DeliverableTracker(one, { claims, owner: 'chat-A' });
    const B = new DeliverableTracker(two, { claims, owner: 'chat-B' });
    A.snapshot(); claims.begin(one, 'chat-A');
    B.snapshot(); claims.begin(two, 'chat-B');
    writeFileSync(join(one, 'a.js'), 'x\n'); writeFileSync(join(two, 'b.js'), 'y\n');
    assert.deepEqual(A.sweep().map(d => d.path), ['a.js'], 'different folders are not overlapped');
    assert.deepEqual(B.sweep().map(d => d.path), ['b.js']);
    claims.end(one, 'chat-A'); claims.end(one, 'chat-A');   // ending twice (stop, then error) is harmless
    claims.end(two, 'chat-B');
    assert.equal(claims.ownerOf(one, 'a.js'), null);
    assert.equal(claims.overlapped(one, 'chat-A'), false);
  } finally { rmSync(one, { recursive: true, force: true }); rmSync(two, { recursive: true, force: true }); }
});

test('a file both chats wrote with their file tools is reported to each chat that wrote it', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-5008e-'));
  try {
    const claims = new FolderClaims();
    const A = new DeliverableTracker(dir, { claims, owner: 'chat-A' });
    const B = new DeliverableTracker(dir, { claims, owner: 'chat-B' });
    A.snapshot(); claims.begin(dir, 'chat-A'); B.snapshot(); claims.begin(dir, 'chat-B');
    assert.equal(tool(dir, A, 'shared.md', 'A').status, 'created');
    assert.ok(tool(dir, B, 'shared.md', 'B-longer-marker'), 'the second chat\'s own write is its own deliverable');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

/* ---------------------------------------------------------------- app side: messages and tool cards of two live chats */

async function bundle(entry, plugins = []) {
  const result = await build({
    entryPoints: [here(entry)], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' }, plugins,
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

test('two live chats in one workspace keep their messages, tool cards and saved copies apart, interleaved, and after a restart', async () => {
  const hostDir = mkdtempSync(join(tmpdir(), 'strata-5008-host-'));
  const chatStore = createChatStore({ dir: hostDir });
  const fakeHost = installFakeHostFetch(chatStore);
  try {
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
  globalThis.__streams = [];
  const mockClient = {
    name: 'mock-client',
    setup(b) {
      b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
      b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({
        loader: 'js',
        contents: `
          export class ApiError extends Error {}
          const reject = async () => { throw new Error('offline'); };
          export const api = { get: reject, post: async () => ({}), put: reject, patch: reject, del: reject, delete: reject };
          export function onEvent() { return () => {}; }
          export async function streamPost(path, body, onFrame) {
            await new Promise((release) => globalThis.__streams.push({ body, onFrame, release }));
          }`,
      }));
    },
  };
  const store = await bundle('../src/lib/store.ts', [mockClient]);
  await store.loadChatsFromHost();   // the first read of the host's chats (the page's own start-up does this inside bootstrap)
  const wait = async (n) => { for (let i = 0; i < 200 && globalThis.__streams.length < n; i++) await new Promise((r) => setTimeout(r, 5)); assert.ok(globalThis.__streams.length >= n, `stream ${n} started`); };

  const a = store.newSession();
  const turnA = store.sendMessage('question for chat A MARK-A');
  await wait(1);
  const b = store.newSession();
  const turnB = store.sendMessage('question for chat B MARK-B');
  await wait(2);
  const [sa, sb] = globalThis.__streams;
  assert.equal(sa.body.sessionId, a.id);
  assert.equal(sb.body.sessionId, b.id);
  assert.notEqual(sa.body.sessionId, sb.body.sessionId);
  assert.ok(!JSON.stringify(sb.body.history).includes('MARK-A'), 'chat B\'s request carries none of chat A\'s history');

  const tool = (id, marker) => ({ type: 'agent-tool', tool: { id, name: 'write', state: 'done', args: { file_path: `${marker}.txt` }, result: `wrote marker ${marker}` } });
  // interleave: A, B, A, B
  sa.onFrame({ type: 'start', model: 'm', contextMax: 8192, requestId: 'rA', agent: true });
  sb.onFrame({ type: 'start', model: 'm', contextMax: 8192, requestId: 'rB', agent: true });
  sa.onFrame(tool('call-1', 'A-unique-1'));
  sb.onFrame(tool('call-1', 'B-unique-1'));   // the same call id in both chats: ids are only unique inside a chat
  sa.onFrame({ type: 'delta', text: 'answer A ' });
  sb.onFrame({ type: 'delta', text: 'answer B ' });
  sa.onFrame({ type: 'delta', text: 'ends A' });
  sb.onFrame({ type: 'delta', text: 'ends B' });
  sa.onFrame({ type: 'done', tokensIn: 5, tokensOut: 5, durationMs: 10, usageSource: 'reported' });
  sb.onFrame({ type: 'done', tokensIn: 5, tokensOut: 5, durationMs: 10, usageSource: 'reported' });
  sa.release(); sb.release();
  await Promise.all([turnA, turnB]);
  await store.flushSessions();

  const saved = [];
  for (const e of chatStore.list().chats) saved.push(JSON.parse(await chatStore.read(e.id)).session);
  assert.equal(saved.length, 2, JSON.stringify(fakeHost.calls.map((c) => `${c.method} ${c.path}`)));
  const bodyOf = (id) => JSON.stringify(saved.find((s) => s.id === id));
  assert.ok(bodyOf(a.id).includes('MARK-A') && bodyOf(a.id).includes('A-unique-1') && bodyOf(a.id).includes('ends A'));
  assert.ok(!/MARK-B|B-unique|ends B/.test(bodyOf(a.id)), 'chat A holds nothing of chat B');
  assert.ok(bodyOf(b.id).includes('MARK-B') && bodyOf(b.id).includes('B-unique-1') && bodyOf(b.id).includes('ends B'));
  assert.ok(!/MARK-A|A-unique|ends A/.test(bodyOf(b.id)), 'chat B holds nothing of chat A');

  // A second window saves a chat of its own, and then one of this window's chats; this window takes the first over and keeps its own.
  const other = { id: 's_from_other_window', title: 'from window 2', createdAt: '2026-12-01T00:00:00.000Z', updatedAt: '2026-12-01T00:00:01.000Z', messages: [{ id: 'o1', role: 'user', content: 'MARK-OTHER-WINDOW' }] };
  const saved2 = await chatStore.save(other.id, JSON.stringify(other));
  await store.applyChatChange(other.id, saved2.rev, false);
  assert.ok(store.useStore((s) => s.sessions).some((s) => s.id === other.id && s.messages[0].content === 'MARK-OTHER-WINDOW'), 'the chat saved by the other window appears here');
  // while a turn is running in chat A here, the other window's save of chat A does not replace it
  store.selectSession(a.id);
  const liveSend = store.sendMessage('second question MARK-A2');
  await wait(3);
  const [, , sa2] = globalThis.__streams;
  await chatStore.save(a.id, JSON.stringify({ ...JSON.parse(JSON.stringify(store.useStore((s) => s.sessions).find((s) => s.id === a.id))), title: 'renamed in window 2' }));
  await store.applyChatChange(a.id, 99, false);
  assert.notEqual(store.useStore((s) => s.sessions).find((s) => s.id === a.id).title, 'renamed in window 2', 'a chat with a turn running here is not replaced');
  sa2.onFrame({ type: 'done', tokensIn: 1, tokensOut: 1, durationMs: 1, usageSource: 'reported' });
  sa2.release(); await liveSend;
  // the other window deletes its chat
  await chatStore.remove(other.id);
  await store.applyChatChange(other.id, null, true);
  assert.ok(!store.useStore((s) => s.sessions).some((s) => s.id === other.id), 'a chat the other window deleted goes here too');
  } finally { fakeHost.restore(); rmSync(hostDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }); }   // a save that lands while the folder is removed (ENOTEMPTY, seen on a macOS runner): try again
});
