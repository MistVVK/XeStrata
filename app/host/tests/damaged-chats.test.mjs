/**
 * #5421: a chat file the host cannot parse (a hand-edited file, a truncated one, a format from a later release) has a surface.
 * The store reports the id, the page reads it through hydrateChats, the sidebar lists it with a delete, and the delete clears the row.
 */
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { build } from 'esbuild';
import { createChatStore, realIo } from '../server/chat-store.mjs';
import { storeTransport, installFakeHostFetch } from './helpers/fake-host-chats.mjs';
import { readFileSync } from 'node:fs';

const here = rel => fileURLToPath(new URL(rel, import.meta.url));
const src = rel => readFileSync(here(`../${rel}`), 'utf8');
const chat = (id, n = 1) => ({
  id, title: `Chat ${id}`, createdAt: '2026-11-04T10:00:00.000Z', updatedAt: '2026-11-04T10:00:00.000Z', workspace: '/w', tokensIn: 0, tokensOut: 0,
  messages: Array.from({ length: n }, (_, i) => ({ id: `${id}-m${i}`, role: i % 2 ? 'assistant' : 'user', content: `${id}:${i}` })),
});
/** Each test here ends in well under a second; a hang fails fast instead of holding a whole host-test run (11-16). */
const TEST_MS = 20_000;
// A test that runs out of time is failed, but what it left waiting (a timer, a poll) keeps this file's process alive, and the run
// waits for the whole file (11-16: until 240 s on Windows and 30 min on Linux). Once every test here has finished, the file ends
// itself a moment later if something is still pending; the timer does not keep a finished file alive.
after(() => { setTimeout(() => process.exit(), 2_000).unref(); });
const truncated = '{"v":1,"rev":1,"savedAt":"2026-11-04T10:00:00.000Z","session":{"id":"s_bad","mess';

/** A store folder with one healthy chat and one file the start-up read cannot parse. */
function scratchHost() {
  const dir = mkdtempSync(join(tmpdir(), 'strata-damaged-'));
  mkdirSync(join(dir, 'chats'));
  writeFileSync(join(dir, 'chats', 's_ok.json'), `{"v":1,"rev":1,"savedAt":"2026-11-04T10:00:00.000Z","session":${JSON.stringify(chat('s_ok', 2))}}`);
  writeFileSync(join(dir, 'chats', 's_bad.json'), truncated);
  const store = createChatStore({ dir, io: realIo });
  return { dir, store, done: () => rmSync(dir, { recursive: true, force: true }) };
}

async function bundle(entry, format = 'esm') {
  const result = await build({
    entryPoints: [here(entry)], bundle: true, format, platform: 'node', write: false, logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' },
  });
  return result.outputFiles[0].text;
}

test('the first read of the host hands the page the damaged ids beside the chats, and a delete takes them away (#5421)', { timeout: TEST_MS }, async () => {
  const h = scratchHost();
  try {
    const sync = await import(`data:text/javascript;base64,${Buffer.from(await bundle('../src/lib/chat-sync.ts')).toString('base64')}`);
    const { TransportError } = await import(`data:text/javascript;base64,${Buffer.from(await bundle('../src/lib/session-persist.ts')).toString('base64')}`);
    const transport = storeTransport(h.store, TransportError);
    const first = await sync.hydrateChats({ transport, browser: { getItem: () => '1', setItem: () => {}, removeItem: () => {} }, current: [] });
    assert.deepEqual(first.damaged, ['s_bad'], 'the damaged file is named to the page');
    assert.deepEqual(first.sessions.map(s => s.id), ['s_ok'], 'the healthy chat is read as a chat');
    await transport.remove('s_bad');
    const second = await sync.hydrateChats({ transport, browser: { getItem: () => '1', setItem: () => {}, removeItem: () => {} }, current: [] });
    assert.deepEqual(second.damaged, [], 'the list is empty once the file is deleted');
    assert.equal(existsSync(join(h.dir, 'chats', 's_bad.json')), false);
    // a host that does not answer reports no damaged chats (nothing was read), and keeps what the page had
    const state = { offline: true };
    const off = await sync.hydrateChats({ transport: storeTransport(h.store, TransportError, state), browser: { getItem: () => '1', setItem: () => {}, removeItem: () => {} }, current: [chat('s_ok', 2)] });
    assert.equal(off.offline, true);
    assert.deepEqual(off.damaged, []);
  } finally { h.done(); }
});

test('the page keeps the damaged list in state, and deleting one clears the row here and on the host (#5421)', { timeout: TEST_MS }, async () => {
  const h = scratchHost();
  const fake = installFakeHostFetch(h.store);
  globalThis.window ??= globalThis;
  // a storage that keeps what it is given, as a browser's does: the first read claims the move of the browser's chats in it and
  // reads the claim back (#5422); a stub that dropped every write left that claim forever with another window, and this test
  // waited in its 250 ms poll until the run's cap (Windows, 11-16)
  const kept = new Map();
  const storage = { getItem: k => kept.has(k) ? kept.get(k) : null, setItem: (k, v) => { kept.set(k, String(v)); }, removeItem: k => { kept.delete(k); },
    key: i => [...kept.keys()][i] ?? null, get length() { return kept.size; } };
  const hadStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', { value: storage, configurable: true, writable: true });
  globalThis.matchMedia ??= () => ({ addEventListener() {}, matches: false });
  try {
    const code = await bundle('../src/lib/store.ts', 'cjs');
    const mod = { exports: {} };
    new Function('require', 'module', 'exports', 'matchMedia', code)(createRequire(import.meta.url), mod, mod.exports, globalThis.matchMedia);
    const store = mod.exports;
    await store.loadChatsFromHost();
    assert.deepEqual(store.getState().sessions.map(s => s.id), ['s_ok'], 'the healthy chat is a row');
    assert.deepEqual(store.getState().damagedChats, ['s_bad'], 'the damaged file is in state');
    await store.deleteDamagedChat('s_bad');
    assert.deepEqual(store.getState().damagedChats, [], 'the row is gone from the page');
    assert.deepEqual(h.store.list().damaged, [], 'and from the host, so the next list() does not put it back');
    assert.equal(existsSync(join(h.dir, 'chats', 's_bad.json')), false, 'the file is unlinked');
    assert.deepEqual(store.getState().sessions.map(s => s.id), ['s_ok'], 'the healthy chat is untouched');
    // a delete in another window (the hub event) clears the row here too
    writeFileSync(join(h.dir, 'chats', 's_worse.json'), truncated);
    const again = createChatStore({ dir: h.dir, io: realIo });
    const swap = installFakeHostFetch(again);
    try {
      await store.loadChatsFromHost();
      assert.deepEqual(store.getState().damagedChats, ['s_worse']);
      await store.applyChatChange('s_worse', null, true);
      assert.deepEqual(store.getState().damagedChats, [], 'the other window deleted it: the row goes without a refresh');
    } finally { swap.restore(); }
  } finally {
    fake.restore();
    if (hadStorage) Object.defineProperty(globalThis, 'localStorage', hadStorage); else delete globalThis.localStorage;
  }
});

test('the sidebar lists the damaged chat files under the workspaces, each with a delete and a confirm (#5421)', { timeout: TEST_MS }, () => {
  const sidebar = src('src/app/Sidebar.tsx');
  assert.match(sidebar, /const damaged = useStore\(s => s\.damagedChats\), newer = useStore\(s => s\.newerChats\);/);
  assert.match(sidebar, /className="ws__group nav-damaged"/);
  assert.match(sidebar, /\{tr\("Unreadable chats"\)\}/);
  assert.match(sidebar, /damaged\.map\(id => <div className="ws__session row" key=\{id\}>/, 'one row per damaged id');
  assert.match(sidebar, /\{id\}\.json/, 'the row names the file, not a chat title');
  assert.match(sidebar, /void deleteDamagedChat\(deleteDamagedId\);/, 'the confirm deletes the file on the host');
  assert.match(sidebar, /tr\("Strata could not read these chat files\. You can delete them here\."\)/, 'one line says what the list is (coordinator wording 11-16)');
  assert.match(sidebar, /<p>\{tr\("Strata could not read this chat file, so its messages are not shown\."\)\}<\/p><\/Dialog>/, 'the question before a delete says what the file is');
});

// GUI-1116 review (coordinator 11-16): a chat saved by a newer Strata is not damaged and is never offered for deletion; dmg09 says what
// happened, so it is the toast after a delete.
test('a chat of a newer Strata is shown with a note to update and never gets a Delete; the toast after a delete says what happened', { timeout: TEST_MS }, () => {
  const sidebar = src('src/app/Sidebar.tsx');
  const row = /\{tr\(newer\.map\(id => <div className="ws__session row" key=\{`newer-\$\{id\}`\} data-newer-chat="">(.*?)<\/div>\)\)\}/.exec(sidebar);
  assert.ok(row, 'one row per newer chat');
  assert.match(row[1], /\{id\}\.json/);
  assert.match(row[1], /tr\("This chat was saved by a newer version of Strata\. Update Strata to open it\."\)/);
  assert.ok(!/deleteDamagedId|setDeleteDamagedId|trash|button/i.test(row[1]), 'no Delete in a newer chat row');
  assert.ok(!/newer\.map[\s\S]{0,400}setDeleteDamagedId/.test(sidebar.slice(sidebar.indexOf('newer.map'), sidebar.indexOf('newer.map') + 600)));
  const store = src('src/lib/store.ts');
  assert.match(store, /set\(\{ damagedChats: state\.damagedChats\.filter\(d => d !== id\) \}\);\s*\/\/[^\n]*\n\s*pushToast\('success', tr\('\{0\} was deleted\. Your other chats are not affected\.', \[`\$\{id\}\.json`\]\)\);/);
  assert.match(store, /set\(\{ sessions, damagedChats: result\.damaged, newerChats: result\.newer \}\);/);
  assert.match(src('src/styles/annotation-fixes.css'), /\.navsession--damaged \{ cursor:default; color:var\(--st-ink-soft\); \}/);
});
