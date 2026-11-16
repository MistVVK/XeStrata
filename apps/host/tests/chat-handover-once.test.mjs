/** #5422: two windows opened together on the first start move the browser's chats to the host once, not twice. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createChatStore } from '../server/chat-store.mjs';
import { storeTransport } from './helpers/fake-host-chats.mjs';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
async function bundle(entry) {
  const result = await build({
    entryPoints: [here(entry)], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' },
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}
const { TransportError, LEGACY_SESSIONS_KEY, HOST_FLAG_KEY, readBrowserSessions } = await bundle('../src/lib/session-persist.ts');
const { hydrateChats } = await bundle('../src/lib/chat-sync.ts');

/** One localStorage both windows share (one browser profile). */
function sharedStorage() {
  const map = new Map();
  return {
    map, get length() { return map.size; }, key: (i) => [...map.keys()][i] ?? null,
    getItem: (k) => (map.has(k) ? map.get(k) : null), removeItem: (k) => { map.delete(k); }, setItem: (k, v) => { map.set(k, String(v)); },
  };
}
const chat = (id, n) => ({
  id, title: `Chat ${id}`, createdAt: '2026-11-01', updatedAt: '2026-11-01', workspace: 'C:/w', tokensIn: 0, tokensOut: 0,
  messages: Array.from({ length: n }, (_, i) => ({ id: `${id}-m${i}`, role: i % 2 ? 'assistant' : 'user', content: `${id}:${i}:` + 'x'.repeat(2000) })),
});
function host() {
  const dir = mkdtempSync(join(tmpdir(), 'strata-handover-'));
  const store = createChatStore({ dir });
  const transport = storeTransport(store, TransportError);
  const imports = [];
  const importMany = transport.importMany;
  // The host's import takes a while for a big archive: both windows are inside it at once without the claim.
  transport.importMany = async (sessions) => { imports.push(sessions.map((s) => s.id)); await new Promise((r) => setTimeout(r, 30)); return importMany(sessions); };
  return { store, transport, imports, done: () => rmSync(dir, { recursive: true, force: true }) };
}
const fast = { sleep: (ms) => new Promise((r) => setTimeout(r, Math.min(ms, 5))) };

test('two windows opened together: one sends the archive, the other waits and reads what was moved', async () => {
  const h = host();
  try {
    const browser = sharedStorage();
    const old = [chat('s_a', 4), chat('s_b', 2), chat('s_c', 6)];
    browser.setItem(LEGACY_SESSIONS_KEY, JSON.stringify(old));
    const current = readBrowserSessions(browser).sessions;
    const [one, two] = await Promise.all([
      hydrateChats({ transport: h.transport, browser, current, windowId: 'w_one', ...fast }),
      hydrateChats({ transport: h.transport, browser, current, windowId: 'w_two', ...fast }),
    ]);
    assert.equal(h.imports.length, 1, `import requests: ${JSON.stringify(h.imports)}`);
    assert.equal(one.moved + two.moved, 3);
    assert.equal(browser.getItem(HOST_FLAG_KEY), '1');
    for (const r of [one, two]) {
      assert.deepEqual(r.sessions.map((s) => s.id).sort(), ['s_a', 's_b', 's_c']);
      assert.deepEqual([...r.revs.values()], [1, 1, 1], 'every moved chat is at its first revision in both windows');
    }
  } finally { h.done(); }
});

test('a claim left by a window that closed mid-move runs out, and the next start moves the chats', async () => {
  const h = host();
  try {
    const browser = sharedStorage();
    browser.setItem(LEGACY_SESSIONS_KEY, JSON.stringify([chat('s_a', 2)]));
    browser.setItem(HOST_FLAG_KEY, 'claim:w_gone:1000');
    let clock = 1000 + 5_000;
    const r = await hydrateChats({
      transport: h.transport, browser, current: [], windowId: 'w_next',
      now: () => clock, sleep: async () => { clock += 1_000; },
    });
    assert.equal(r.moved, 1);
    assert.ok(clock - 1000 >= 30_000 && clock - 1000 <= 32_000, `took over after ${clock - 1000} ms of the claim`);
    assert.equal(browser.getItem(HOST_FLAG_KEY), '1');
  } finally { h.done(); }
});

test('a host that goes away during the move gives the claim back, so the next start tries again at once', async () => {
  const h = host();
  try {
    const browser = sharedStorage();
    browser.setItem(LEGACY_SESSIONS_KEY, JSON.stringify([chat('s_a', 2)]));
    h.transport.importMany = async () => { throw new TransportError('The host did not answer.', 0, 'offline', true); };
    const r = await hydrateChats({ transport: h.transport, browser, current: [], windowId: 'w_one', ...fast });
    assert.equal(r.offline, true);
    assert.equal(browser.getItem(HOST_FLAG_KEY), null);
  } finally { h.done(); }
});
