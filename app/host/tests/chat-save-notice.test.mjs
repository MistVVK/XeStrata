/** The notice for chats the host could not write (owner wording 11-04): shown once, with a copy button, in every language, gone when the chats are saved. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { mkdtempSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createChatStore, realIo } from '../server/chat-store.mjs';
import { installFakeHostFetch } from './helpers/fake-host-chats.mjs';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
async function bundle(entry, plugins = []) {
  const result = await build({
    entryPoints: [here(entry)], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' }, plugins,
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

const TITLE = 'Some chats could not be saved';
const BODY = "Strata's chat storage is full, so the newest changes to {0} chat(s) are only in this window. Delete old chats to make room; they then save by themselves.";
const BUTTON = 'Copy unsaved chats';

test('the notice exists in all nine languages, with the placeholder kept', () => {
  for (const lang of ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he']) {
    const file = JSON.parse(readFileSync(here(`../src/locales/sav-${lang}.json`), 'utf8'));
    assert.deepEqual(Object.keys(file), ['sav01', 'sav02', 'sav03'], lang);
    assert.ok(/\{0\}/.test(file.sav02), `${lang} keeps {0}`);
    assert.ok(Object.values(file).every((t) => t.trim().length > 3), lang);
    if (lang === 'en') assert.deepEqual(Object.values(file), [TITLE, BODY, BUTTON], 'the owner\'s exact wording');
    else assert.notEqual(file.sav01, TITLE, `${lang} is translated`);
  }
});

test('the catalog finds the notice: the German page shows it in German, with the number filled in', async () => {
  const i18n = await bundle('../src/lib/i18n.ts');
  await i18n.setTranslationLanguage('de');
  assert.equal(i18n.tr(TITLE), 'Einige Chats konnten nicht gespeichert werden');
  assert.equal(i18n.tr(BUTTON), 'Nicht gespeicherte Chats kopieren');
  assert.match(i18n.tr(BODY, [3]), /^Der Chat-Speicher von Strata ist voll, daher gibt es die neuesten Änderungen an 3 Chat\(s\) nur in diesem Fenster/);
  i18n.setTranslationLanguage('en');
  assert.equal(i18n.tr(BODY, [3]), "Strata's chat storage is full, so the newest changes to 3 chat(s) are only in this window. Delete old chats to make room; they then save by themselves.");
});

test('a full disk raises the notice once with a working copy button; offline raises nothing; saving by itself takes the notice away', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-notice-'));
  let full = false;
  const chatStore = createChatStore({ dir, io: { ...realIo, async writeFile(path, data) { if (full) throw Object.assign(new Error('no space left on device'), { code: 'ENOSPC' }); return realIo.writeFile(path, data); } } });
  const fake = installFakeHostFetch(chatStore);
  try {
    globalThis.window ??= globalThis;
    globalThis.document ??= { addEventListener() {}, hasFocus: () => true, visibilityState: 'visible', documentElement: { dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } } };
    const memory = new Map();
    globalThis.localStorage = { get length() { return memory.size; }, key: (i) => [...memory.keys()][i] ?? null, getItem: (k) => memory.get(k) ?? null, setItem: (k, v) => { memory.set(k, v); }, removeItem: (k) => memory.delete(k) };
    globalThis.navigator ??= { language: 'en' };
    const copied = [];
    Object.defineProperty(globalThis, 'navigator', { value: { language: 'en', clipboard: { writeText: async (t) => { copied.push(t); } } }, configurable: true });
    globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
    const mockClient = {
      name: 'mock-client',
      setup(b) {
        b.onResolve({ filter: /\/client$/ }, () => ({ path: 'mock-client', namespace: 'mock' }));
        b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({ loader: 'js', contents: `
          export class ApiError extends Error {}
          const reject = async () => { throw new Error('offline'); };
          export const api = { get: reject, post: async () => ({}), put: reject, patch: reject, del: reject, delete: reject };
          export function onEvent() { return () => {}; }
          export async function streamPost(path, body, onFrame) { onFrame({ type: 'start', model: 'm', contextMax: 8192 }); onFrame({ type: 'delta', text: 'an answer' }); onFrame({ type: 'done', tokensIn: 1, tokensOut: 2, durationMs: 1, usageSource: 'reported' }); }` }));
      },
    };
    const store = await bundle('../src/lib/store.ts', [mockClient]);
    await store.loadChatsFromHost();
    const toasts = () => store.useStore((s) => s.toasts);
    const notice = () => store.useStore((s) => s.sessionSave);

    store.newSession();
    await store.flushSessions();
    assert.equal(notice(), null);
    full = true;
    await store.sendMessage('a question that cannot be saved');
    await store.flushSessions();
    assert.equal(notice().reason, 'disk-full');
    assert.equal(notice().failedIds.length, 1);
    const shown = toasts().filter((t) => t.title === TITLE);
    assert.equal(shown.length, 1, 'shown once');
    assert.equal(shown[0].tone, 'error');
    assert.equal(shown[0].body, "Strata's chat storage is full, so the newest changes to 1 chat(s) are only in this window. Delete old chats to make room; they then save by themselves.");
    assert.equal(shown[0].action.label, BUTTON);
    await store.flushSessions();
    assert.equal(toasts().filter((t) => t.title === TITLE).length, 1, 'a second failed save does not show it again');

    // the button copies the unsaved chats whole
    assert.ok(store.unsavedSessionsJson().includes('a question that cannot be saved'));
    shown[0].action.run();
    for (let i = 0; i < 20 && !copied.length; i++) await new Promise((r) => setTimeout(r, 10));
    assert.equal(copied.length, 1);
    const copy = JSON.parse(copied[0]);
    assert.equal(copy[0].messages.some((m) => m.content === 'a question that cannot be saved'), true);

    // a host that does not answer is not a full disk: no new notice, nothing claimed
    full = false;
    store.dismissToast(shown[0].id);
    await store.flushSessions();   // there is room now: the chat saves and the notice goes
    assert.equal(notice(), null);
    assert.equal(toasts().filter((t) => t.title === TITLE && !t.leaving).length, 0, 'the notice is taken away when the chats are saved');
    const onDisk = JSON.parse(await chatStore.read(copy[0].id)).session;
    assert.equal(onDisk.messages.some((m) => m.content === 'a question that cannot be saved'), true, 'and the chat is on disk after all');
  } finally { fake.restore(); rmSync(dir, { recursive: true, force: true }); }
});
