/**
 * #3604 / #3603: the chat list is saved at most once a second and only the chats that changed are sent.
 * #5192: nothing is ever cut out of a saved chat. Chats are files on the host (owner decision 11-04): big chats, restart, the move from the
 * browser's two old layouts, a full disk, a host that is away, two windows.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createChatStore, realIo } from '../server/chat-store.mjs';
import { storeTransport, installFakeHostFetch } from './helpers/fake-host-chats.mjs';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

async function bundle(entry, plugins = []) {
  const result = await build({
    entryPoints: [here(entry)], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' }, plugins,
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

const persist = await bundle('../src/lib/session-persist.ts');
const sync = await bundle('../src/lib/chat-sync.ts');
const { createSessionWriter, serializeSession, readBrowserSessions, exportSessions, TransportError, LEGACY_SESSIONS_KEY, SESSION_INDEX_KEY, SESSION_KEY_PREFIX, HOST_FLAG_KEY, PENDING_KEY } = persist;
const { hydrateChats, keepLastMomentCopy, takeLastMomentCopy } = sync;

/** A Storage with a quota, like the browser's. */
function fakeStorage(limit = Infinity) {
  const map = new Map();
  const used = () => [...map].reduce((n, [k, v]) => n + k.length + v.length, 0);
  return {
    map,
    get length() { return map.size; },
    key: (i) => [...map.keys()][i] ?? null,
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    removeItem: (k) => { map.delete(k); },
    setItem(k, v) {
      const before = map.has(k) ? k.length + map.get(k).length : 0;
      if (used() - before + k.length + String(v).length > limit) { const e = new Error('The quota has been exceeded.'); e.name = 'QuotaExceededError'; throw e; }
      map.set(k, String(v));
    },
  };
}
const chat = (id, n, len, extra = {}) => ({
  id, title: `Chat ${id}`, createdAt: `2026-11-0${(id.charCodeAt(0) % 8) + 1}`, updatedAt: id, workspace: 'C:/w', tokensIn: 0, tokensOut: 0,
  messages: Array.from({ length: n }, (_, i) => ({ id: `${id}-m${i}`, role: i % 2 ? 'assistant' : 'user', content: `${id}:${i}:` + String.fromCharCode(97 + (i % 26)).repeat(len) })),
  ...extra,
});
const host = (io = realIo) => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-persist-'));
  const store = createChatStore({ dir, io });
  const state = { offline: false };
  const transport = storeTransport(store, TransportError, state);
  return { dir, store, state, transport, reopen: () => createChatStore({ dir }), done: () => rmSync(dir, { recursive: true, force: true }) };
};
const writerOn = (h, opts = {}) => createSessionWriter({ transport: h.transport, ...opts });
const saveAll = async (h, sessions, opts) => { const w = writerOn(h, opts); w.schedule(sessions); await w.flush(); return w; };
const fromHost = async (h) => { const out = []; for (const e of h.reopen().list().chats) out.push(JSON.parse(await h.reopen().read(e.id)).session); return out; };

test('2,000 schedules inside a second produce one save, and the newest state wins', async (t) => {
  const h = host();
  try {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let puts = 0;
    const put = h.transport.put;
    h.transport.put = (...a) => { puts++; return put(...a); };
    const writer = writerOn(h, { delayMs: 1_000 });
    for (let i = 0; i < 2_000; i++) writer.schedule([{ id: 's1', seq: i, messages: [] }]);
    assert.equal(puts, 0, 'nothing is sent synchronously');
    t.mock.timers.tick(1_000);
    await writer.flush();
    assert.equal(puts, 1);
    assert.equal((await fromHost(h))[0].seq, 1_999);
    writer.schedule([{ id: 's1', seq: 2_000, messages: [] }]);
    await writer.flush();
    assert.equal(puts, 2);
    assert.equal((await fromHost(h))[0].seq, 2_000);
    await writer.flush();
    assert.equal(puts, 2, 'a flush with nothing pending sends nothing');
  } finally { h.done(); }
});

test('only the chat that changed is serialised and sent again', async () => {
  const h = host();
  try {
    const seen = [], sent = [];
    const put = h.transport.put;
    h.transport.put = (id, ...a) => { sent.push(id); return put(id, ...a); };
    const writer = writerOn(h, { onSerialize: (s) => seen.push(s.id) });
    const a = { id: 'a', updatedAt: '1', messages: [{ id: 'm', content: 'x'.repeat(1000) }] };
    const b = { id: 'b', updatedAt: '2', messages: [] };
    writer.schedule([a, b]); await writer.flush();
    assert.deepEqual(seen.sort(), ['a', 'b']);
    seen.length = 0; sent.length = 0;
    for (let i = 0; i < 50; i++) { writer.schedule([a, { ...b, title: `t${i}` }]); await writer.flush(); }
    assert.equal(seen.filter((id) => id === 'a').length, 0, 'the untouched chat is never stringified again');
    assert.equal(sent.filter((id) => id === 'a').length, 0, 'the untouched chat is never sent again');
    assert.equal(sent.filter((id) => id === 'b').length, 50);
    assert.equal(h.reopen().list().chats.length, 2);
  } finally { h.done(); }
});

/* #5192: the register's own numbers. Twelve messages of 140,000 characters went through the real serializer and six came back. */
test('#5192: 12 messages of 140,000 characters are saved and restored whole, twice over (save, restart, save, restart)', async () => {
  const h = host();
  try {
    const original = chat('s_a', 12, 140_000);
    const before = JSON.stringify(original);
    await saveAll(h, [original]);
    const first = await hydrateChats({ transport: h.transport, browser: fakeStorage(), current: [] });
    assert.equal(first.sessions.length, 1);
    assert.equal(first.sessions[0].messages.length, 12, 'all twelve messages come back');
    assert.equal(JSON.stringify(first.sessions[0]), before, 'byte for byte after one restart');
    const w = createSessionWriter({ transport: h.transport, known: first.revs });
    first.sessions.forEach((s) => w.adopt(s, first.revs.get(s.id)));
    w.schedule([{ ...first.sessions[0], title: 'renamed' }]); await w.flush();
    const second = await hydrateChats({ transport: h.transport, browser: fakeStorage(), current: [] });
    assert.equal(JSON.stringify({ ...second.sessions[0], title: original.title }), before, 'byte for byte after two restarts (apart from the rename)');
    assert.equal(JSON.stringify(original), before, 'the chat in memory was not changed');
  } finally { h.done(); }
});

test('#5192: far more text than the browser could hold (over 10 MB) keeps every chat and every message', async () => {
  const h = host();
  try {
    const chats = ['a', 'b', 'c', 'd', 'e', 'f', 'g'].map((id, i) => chat(`s_${id}`, 8, 100_000, { updatedAt: String(i) }));
    const total = chats.reduce((n, c) => n + JSON.stringify(c).length, 0);
    assert.ok(total > 5_000_000, `fixture is ${total} characters`);
    chats.push({ id: 's_big', createdAt: '2026-11-01', updatedAt: 'z', messages: [{ id: '1', role: 'user', content: 'p'.repeat(250_000) }, { id: '2', role: 'assistant', content: 'ok', tools: [{ name: 'read', result: 'y'.repeat(6_000_000) }] }] });
    await saveAll(h, chats);
    const back = await fromHost(h);
    assert.equal(back.length, chats.length);
    for (const c of chats) assert.equal(JSON.stringify(back.find((s) => s.id === c.id)), JSON.stringify(c));
    assert.equal(serializeSession(chats[0]), JSON.stringify(chats[0]));
    assert.ok(!JSON.stringify(back).includes('shortened'));
  } finally { h.done(); }
});

test('a full disk is reported as such: the chat keeps its last file, the others still save, the unsaved chat is offered whole, and it saves by itself when there is room', async () => {
  let full = false;
  const h = host({ ...realIo, async writeFile(path, data) { if (full && /s_b\.json/.test(path)) throw Object.assign(new Error('no space left on device'), { code: 'ENOSPC' }); return realIo.writeFile(path, data); } });
  try {
    const results = [];
    const writer = writerOn(h, { onResult: (r) => results.push(r), retryMs: 5 });
    const small = chat('s_s', 2, 100), big = chat('s_b', 4, 1_000);
    writer.schedule([small, big]); await writer.flush();
    assert.equal(results.at(-1).ok, true);
    const goodCopy = await h.reopen().read('s_b');
    full = true;
    const grown = { ...big, messages: [...big.messages, { id: 'b-big', role: 'user', content: 'g'.repeat(5_000) }] };
    const fresh = chat('s_n', 2, 100);
    writer.schedule([fresh, small, grown]); const r = await writer.flush();
    assert.equal(r.ok, false);
    assert.deepEqual(r.failed.map((f) => [f.id, f.reason]), [['s_b', 'disk-full']]);
    assert.equal(await h.reopen().read('s_b'), goodCopy, 'the last file of the chat is untouched');
    assert.ok(await h.reopen().read('s_n'), 'the other chats are still saved');
    assert.deepEqual(writer.unsaved().map((s) => s.id), ['s_b']);
    assert.equal(JSON.parse(exportSessions(writer.unsaved()))[0].messages.length, 5, 'the unsaved chat is complete');
    // room is made; the retry saves the chat without being asked again
    full = false;
    for (let i = 0; i < 200 && writer.unsaved().length; i++) await new Promise((res) => setTimeout(res, 10));
    assert.equal(writer.unsaved().length, 0, 'saved by itself');
    assert.equal(results.at(-1).ok, true);
    assert.equal(JSON.parse(await h.reopen().read('s_b')).session.messages.length, 5);
  } finally { h.done(); }
});

test('a refused write (no permission) is reported as permission; a host that is away is reported as offline and caught up when it is back', async () => {
  let denied = true;
  const h = host({ ...realIo, async writeFile(path, data) { if (denied) throw Object.assign(new Error('access denied'), { code: 'EACCES' }); return realIo.writeFile(path, data); } });
  try {
    const results = [];
    const writer = writerOn(h, { onResult: (r) => results.push(r), retryMs: 5 });
    writer.schedule([chat('s_p', 1, 5)]); await writer.flush();
    assert.equal(results.at(-1).failed[0].reason, 'permission');
    denied = false; h.state.offline = true;
    for (let i = 0; i < 100 && results.length < 3; i++) await new Promise((res) => setTimeout(res, 10));
    assert.equal(results.at(-1).failed[0].reason, 'offline', 'a host that does not answer is not a disk problem');
    h.state.offline = false;
    for (let i = 0; i < 200 && writer.unsaved().length; i++) await new Promise((res) => setTimeout(res, 10));
    assert.equal(writer.unsaved().length, 0);
    assert.equal(h.reopen().list().chats.length, 1);
  } finally { h.done(); }
});

test('a chat removed from the list is removed on the host; chats this writer never saw are not; nothing is saved before the writer is enabled', async () => {
  const h = host();
  try {
    await h.store.save('s_other_window', JSON.stringify(chat('s_other_window', 1, 5)));
    const a = chat('s_a', 1, 5), b = chat('s_b', 1, 5);
    const writer = writerOn(h, { enabled: false });
    writer.schedule([a, b]); await writer.flush();
    assert.deepEqual(h.reopen().list().chats.map((c) => c.id), ['s_other_window'], 'off: nothing is sent');
    writer.enable(); await writer.flush();
    assert.deepEqual(h.reopen().list().chats.map((c) => c.id).sort(), ['s_a', 's_b', 's_other_window']);
    writer.schedule([a]); await writer.flush();
    assert.deepEqual(h.reopen().list().chats.map((c) => c.id).sort(), ['s_a', 's_other_window']);
    writer.schedule([]); await writer.flush();
    assert.deepEqual(h.reopen().list().chats.map((c) => c.id), ['s_other_window'], 'an empty list removes the chats this window had, not the other window\'s');
  } finally { h.done(); }
});

/* The move from the browser's storage */
function browserWith({ legacy = [], perChat = [], order = [] } = {}) {
  const storage = fakeStorage();
  if (legacy.length) storage.setItem(LEGACY_SESSIONS_KEY, JSON.stringify(legacy));
  for (const c of perChat) storage.setItem(`${SESSION_KEY_PREFIX}${c.id}`, JSON.stringify(c));
  if (order.length) storage.setItem(SESSION_INDEX_KEY, JSON.stringify({ v: 2, ids: order }));
  return storage;
}
const snapshot = (storage) => JSON.stringify([...storage.map].sort());

test('the browser\'s chats move to the host from both old layouts (the 0.2.10 list and the per-chat entries), once, and the browser copy is left as it is', async () => {
  const h = host();
  try {
    const legacyOnly = chat('s_old', 12, 140_000), both = chat('s_both', 3, 100, { title: 'old list' });
    const bothNewer = chat('s_both', 5, 100, { title: 'per-chat entry' }), perChatOnly = chat('s_new', 2, 100);
    const browser = browserWith({ legacy: [legacyOnly, both], perChat: [bothNewer, perChatOnly], order: ['s_new', 's_both'] });
    const before = snapshot(browser);
    const read = readBrowserSessions(browser);
    assert.deepEqual(read.sessions.map((s) => s.id), ['s_new', 's_both', 's_old']);
    assert.equal(read.sessions.find((s) => s.id === 's_both').title, 'per-chat entry', 'the per-chat entry wins where a chat is in both');
    const result = await hydrateChats({ transport: h.transport, browser, current: read.sessions });
    assert.equal(result.moved, 3);
    assert.equal(result.failed.length, 0);
    assert.equal(browser.getItem(HOST_FLAG_KEY), '1');
    const onHost = Object.fromEntries((await fromHost(h)).map((s) => [s.id, s]));
    assert.equal(JSON.stringify(onHost.s_old), JSON.stringify(legacyOnly), 'the big chat arrived whole');
    assert.equal(onHost.s_both.title, 'per-chat entry');
    assert.deepEqual(result.sessions.map((s) => s.id).sort(), ['s_both', 's_new', 's_old']);
    const after = new Map(browser.map); after.delete(HOST_FLAG_KEY);
    assert.equal(JSON.stringify([...after].sort()), before, 'the browser copy is untouched (it stays as a fallback for one release)');
    // later: the person deletes a chat on the host; the next start does not bring it back from the browser copy
    await h.store.remove('s_old');
    const again = await hydrateChats({ transport: h.transport, browser, current: [] });
    assert.equal(again.moved, 0);
    assert.deepEqual(again.sessions.map((s) => s.id).sort(), ['s_both', 's_new']);
  } finally { h.done(); }
});

test('a move that the host refuses (disk full) is not flagged as done, keeps the chats in the page, and is tried again at the next start', async () => {
  let full = true;
  const h = host({ ...realIo, async writeFile(path, data) { if (full && /s_x\.json/.test(path)) throw Object.assign(new Error('no space'), { code: 'ENOSPC' }); return realIo.writeFile(path, data); } });
  try {
    const browser = browserWith({ legacy: [chat('s_x', 3, 50), chat('s_y', 2, 50)] });
    const current = readBrowserSessions(browser).sessions;
    const first = await hydrateChats({ transport: h.transport, browser, current });
    assert.equal(first.failed.length, 1);
    assert.equal(browser.getItem(HOST_FLAG_KEY), null, 'not done');
    assert.deepEqual(first.sessions.map((s) => s.id).sort(), ['s_x', 's_y'], 'the chat the host refused is still shown');
    full = false;
    const second = await hydrateChats({ transport: h.transport, browser, current: first.sessions });
    assert.equal(second.failed.length, 0);
    assert.equal(browser.getItem(HOST_FLAG_KEY), '1');
    assert.deepEqual((await fromHost(h)).map((s) => s.id).sort(), ['s_x', 's_y']);
  } finally { h.done(); }
});

test('a host that is not answering at start leaves the page with what it has and moves nothing', async () => {
  const h = host();
  try {
    h.state.offline = true;
    const browser = browserWith({ legacy: [chat('s_a', 1, 5)] });
    const current = readBrowserSessions(browser).sessions;
    const result = await hydrateChats({ transport: h.transport, browser, current });
    assert.equal(result.offline, true);
    assert.deepEqual(result.sessions, current);
    assert.equal(browser.getItem(HOST_FLAG_KEY), null);
  } finally { h.done(); }
});

test('the last-moment copy: chats the host had not confirmed come back at the next start when newer, and the copy is used up', async () => {
  const h = host();
  try {
    const browser = fakeStorage();
    await saveAll(h, [chat('s_a', 2, 10, { updatedAt: '2026-11-04T10:00:00Z' })]);
    const unsaved = [chat('s_a', 4, 10, { updatedAt: '2026-11-04T10:05:00Z' }), chat('s_b', 1, 10, { updatedAt: '2026-11-04T10:06:00Z' })];
    assert.equal(keepLastMomentCopy(browser, unsaved), true);
    const result = await hydrateChats({ transport: h.transport, browser, current: [] });
    assert.deepEqual(result.recovered.sort(), ['s_a', 's_b']);
    assert.equal(result.sessions.find((s) => s.id === 's_a').messages.length, 4);
    assert.equal(browser.getItem(PENDING_KEY), null);
    // an older copy than the host's file is dropped
    keepLastMomentCopy(browser, [chat('s_a', 1, 10, { updatedAt: '2026-11-01T00:00:00Z' })]);
    const older = takeLastMomentCopy(browser, [chat('s_a', 4, 10, { updatedAt: '2026-11-04T10:05:00Z' })], new Map());
    assert.deepEqual(older.ids, []);
    // a copy that does not fit the browser is not an error
    assert.equal(keepLastMomentCopy(fakeStorage(10), unsaved), false);
  } finally { h.done(); }
});

test('two windows on one host: neither removes the other\'s chats; the same chat saved from an older copy keeps the displaced version on the host', async () => {
  const h = host();
  try {
    const a = createSessionWriter({ transport: h.transport });
    const b = createSessionWriter({ transport: h.transport });
    const chatA = chat('s_a', 2, 10), chatB = chat('s_b', 2, 10);
    a.schedule([chatA]); await a.flush();
    b.schedule([chatB]); await b.flush();
    assert.deepEqual(h.reopen().list().chats.map((c) => c.id).sort(), ['s_a', 's_b']);
    // window B opens chat A (rev 1) and keeps its copy; window A then saves a longer chat A, then B saves from its older copy
    const opened = await h.transport.get('s_a');
    b.adopt(opened.session, opened.rev);
    const longer = chat('s_a', 6, 10);
    a.schedule([longer, chatB]); await a.flush();
    b.schedule([{ ...opened.session, title: 'edited in window B' }, chatB]); const r = await b.flush();
    assert.equal(r.ok, true);
    assert.equal(JSON.parse(await h.reopen().read('s_a')).session.title, 'edited in window B');
    const kept = await import('node:fs').then((fs) => fs.readdirSync(join(h.dir, 'chats', 'conflicts')));
    assert.equal(kept.length, 1, 'window A\'s version is kept on the host');
    assert.deepEqual(h.reopen().list().chats.map((c) => c.id).sort(), ['s_a', 's_b']);
  } finally { h.done(); }
});

/* End to end through the real store: a 1,500-token streamed answer, the host's files, a restart. */
test('a streamed answer of 1,500 tokens is a handful of saves to the host, the final text is on disk, and a restart brings it back', async () => {
  const h = host();
  const fake = installFakeHostFetch(h.store);
  try {
    globalThis.window ??= globalThis;
    globalThis.document ??= { addEventListener() {}, hasFocus: () => true, visibilityState: 'visible', documentElement: { dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } } };
    const memory = new Map();
    globalThis.localStorage = {
      get length() { return memory.size; }, key: (i) => [...memory.keys()][i] ?? null,
      getItem: (k) => memory.get(k) ?? null, setItem: (k, v) => { memory.set(k, v); }, removeItem: (k) => memory.delete(k),
    };
    globalThis.navigator ??= { language: 'en' };
    globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
    const TOKENS = 1_500;
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
              onFrame({ type: 'start', model: 'm', contextMax: 8192, requestId: 'r1' });
              for (let i = 0; i < ${TOKENS}; i++) onFrame({ type: 'delta', text: 'word ' });
              onFrame({ type: 'done', tokensIn: 10, tokensOut: ${TOKENS}, durationMs: 100, usageSource: 'reported' });
            }`,
        }));
      },
    };
    const store = await bundle('../src/lib/store.ts', [mockClient]);
    await store.loadChatsFromHost();   // the first read of the host's chats (bootstrap does this first; here the rest of it is mocked away)
    const before = fake.calls.filter((c) => c.method === 'PUT').length;
    await store.sendMessage('Write a long answer');
    await store.flushSessions();
    const puts = fake.calls.filter((c) => c.method === 'PUT').length - before;
    assert.ok(puts >= 1 && puts <= 6, `a 1,500-token answer sent the chat ${puts} times`);
    assert.ok(fake.calls.filter((c) => c.method === 'PUT').every((c) => c.headers['x-strata-client'] && c.headers['x-strata-window']), 'every save names the page and the window');
    const saved = (await fromHost(h)).find((s) => s.messages.some((m) => m.role === 'assistant'));
    const answer = saved.messages.find((m) => m.role === 'assistant');
    assert.ok(answer.content.length >= TOKENS * 5 - 10, 'the finished answer is what is on disk');
    assert.equal(answer.streaming, false);
    assert.equal([...memory.keys()].filter((k) => k.startsWith('strata.session')).length, 0, 'nothing is written to the browser storage any more');
  } finally { fake.restore(); h.done(); }
});

// #5923: a reply streamed into a big chat rewrote the whole chat once a second (about 1 MB/s for a 1 MB chat, growing with the chat).
// The wait before a scheduled save now grows with the size of the chats that changed; small chats still save within a second, and a
// turn's end (flush) saves at once.
const countingTransport = () => { const t = { bytes: 0, puts: 0, rev: 0 }; t.put = async (id, text) => { t.bytes += text.length; t.puts++; t.last = text; return { rev: ++t.rev }; }; t.remove = async () => {}; return t; };
async function streamFor(t, writer, base, seconds) {
  let s = base;
  for (let ms = 0; ms < seconds * 1000; ms += 100) {
    s = { ...s, messages: [...s.messages.slice(0, -1), { ...s.messages.at(-1), content: s.messages.at(-1).content + 'tok ' }] };
    writer.schedule([s]);
    t.mock.timers.tick(100);
    for (let i = 0; i < 3; i++) await new Promise(r => setImmediate(r));
  }
  return s;
}

test('#5923: a reply streamed for a minute into a 1 MB chat writes about 12 MB (a save at least every 5 s), not 60; a small chat still saves within a second; the turn end saves at once', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const big = chat('big', 20, 50_000);
  const tr1 = countingTransport();
  const w1 = createSessionWriter({ transport: tr1 });
  w1.adopt(big, 1);
  const last = await streamFor(t, w1, big, 60);
  const size = serializeSession(big).length;
  assert.ok(size > 1_000_000, String(size));
  assert.ok(tr1.bytes <= 13 * size, `a minute of streaming wrote ${(tr1.bytes / 1e6).toFixed(1)} MB for a ${(size / 1e6).toFixed(1)} MB chat (${tr1.puts} saves)`);
  assert.ok(tr1.puts >= 11, `saved at least every 5 s while the reply streams (${tr1.puts} saves)`);
  t.diagnostic(`one minute: ${tr1.puts} saves, ${(tr1.bytes / 1e6).toFixed(1)} MB for a ${(size / 1e6).toFixed(2)} MB chat`);
  const end = { ...last, messages: [...last.messages.slice(0, -1), { ...last.messages.at(-1), content: last.messages.at(-1).content + 'done.' }] };
  w1.schedule([end]);
  await w1.flush();
  assert.equal(JSON.parse(tr1.last).messages.at(-1).content, end.messages.at(-1).content, 'the turn end saves the newest state at once');

  const small = chat('small', 4, 200);
  const tr2 = countingTransport();
  const w2 = createSessionWriter({ transport: tr2 });
  w2.adopt(small, 1);
  await streamFor(t, w2, small, 3);
  assert.ok(tr2.puts >= 2, `a small chat saves about once a second (${tr2.puts} saves in 3 s)`);
});
