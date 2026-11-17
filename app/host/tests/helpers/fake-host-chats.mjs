/** A stand-in for the host's /api/chats routes that is backed by the real chat store (server/chat-store.mjs), for tests of the page's side. */
import { ChatStoreError, storeFailure } from '../../server/chat-store.mjs';

/** A transport (the page's ChatTransport) straight over a chat store, with a way to make the host go away. */
export function storeTransport(store, TransportError, state = { offline: false }) {
  const guard = async (work) => {
    if (state.offline) throw new TransportError('The host did not answer.', 0, 'offline', true);
    try { return await work(); }
    catch (e) { const f = e instanceof TransportError ? e : storeFailure(e); throw f instanceof TransportError ? f : new TransportError(f.message, f.status, f.code); }
  };
  return {
    state,
    list: () => guard(async () => store.list()),
    get: (id) => guard(async () => { const text = await store.read(id); if (text === null) return null; const d = JSON.parse(text); return { rev: d.rev, session: d.session }; }),
    put: (id, text, baseRev) => guard(() => store.save(id, text, { baseRev })),
    remove: (id) => guard(async () => { await store.remove(id); }),
    importMany: (sessions) => guard(() => store.importMany(sessions)),
  };
}

/** Install a global fetch that answers /api/chats* from the store; every other path is refused like an unreachable host. */
export function installFakeHostFetch(store, { events = [] } = {}) {
  const calls = [];
  const previous = globalThis.fetch;
  globalThis.fetch = async (url, init = {}) => {
    const path = String(url).replace(/^https?:\/\/[^/]+/, '');
    const method = init.method ?? 'GET';
    const headers = Object.fromEntries(Object.entries(init.headers ?? {}).map(([k, v]) => [k.toLowerCase(), v]));
    calls.push({ method, path, headers });
    const reply = (status, body) => ({ ok: status < 200 || status >= 300 ? false : true, status, json: async () => (typeof body === 'string' ? JSON.parse(body) : body), text: async () => (typeof body === 'string' ? body : JSON.stringify(body)) });
    const m = /^\/api\/chats(?:\/([^/?]+))?/.exec(path);
    if (!m) throw new TypeError('fetch failed');
    if (!headers['x-strata-client']) return reply(403, { error: 'Not allowed.' });
    const id = m[1] ? decodeURIComponent(m[1]) : null;
    try {
      if (!id && method === 'GET') return reply(200, store.list());
      if (id === 'import' && method === 'POST') { const r = await store.importMany(JSON.parse(init.body).sessions); return reply(200, r); }
      if (id && method === 'GET') { const t = await store.read(id); return t === null ? reply(404, { error: 'No such chat.', code: 'not-found' }) : reply(200, t); }
      if (id && method === 'PUT') { const base = headers['x-strata-base-rev']; const r = await store.save(id, init.body, { baseRev: base !== undefined ? Number(base) : null }); events.push({ id, rev: r.rev, from: headers['x-strata-window'] }); return reply(200, r); }
      if (id && method === 'DELETE') { const r = await store.remove(id); events.push({ id, deleted: true, from: headers['x-strata-window'] }); return reply(200, r); }
    } catch (e) { const f = e instanceof ChatStoreError ? e : storeFailure(e); return reply(f.status, { error: f.message, code: f.code }); }
    return reply(405, { error: 'Method not allowed.' });
  };
  return { calls, restore() { globalThis.fetch = previous; } };
}
