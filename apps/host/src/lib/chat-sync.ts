import type { Session } from './types';
import {
  HOST_FLAG_KEY, PENDING_KEY, TransportError, readBrowserSessions,
  type ChatTransport, type HostChatEntry, type StorageLike,
} from './session-persist';

/**
 * The page's side of the chat files on the host (server/chat-store.mjs): the HTTP transport, the first read at start (with the move of
 * the browser's old chats to the host), the last-moment copy for a page that goes away, and the merge of what another window saved.
 */

const BASE = () => (globalThis as { __STRATA_HOST__?: string }).__STRATA_HOST__ ?? '';
/** Names this window to the host, so the event of its own save is not mistaken for another window's. */
export const WINDOW_ID = `w_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36).slice(-4)}`;

export function createHttpTransport(fetchImpl: typeof fetch = (...args) => fetch(...args)): ChatTransport {
  const headers = (extra: Record<string, string> = {}) => ({ 'x-strata-client': 'ui', 'x-strata-window': WINDOW_ID, ...extra });
  async function call(method: string, path: string, init: { body?: string; headers?: Record<string, string> } = {}): Promise<Response> {
    let res: Response;
    try { res = await fetchImpl(`${BASE()}${path}`, { method, headers: headers(init.headers), body: init.body }); }
    catch (e) { throw new TransportError(e instanceof Error ? e.message : 'The host did not answer.', 0, 'offline', true); }
    if (res.ok) return res;
    let code: string | undefined, message = `Request failed (${res.status})`;
    try { const data = await res.json() as { error?: string; code?: string }; code = data.code; if (data.error) message = data.error; } catch { /* no body */ }
    throw new TransportError(message, res.status, code);
  }
  return {
    async list() { return await (await call('GET', '/api/chats')).json() as { chats: HostChatEntry[]; damaged?: string[] }; },
    async get(id) {
      try {
        const data = await (await call('GET', `/api/chats/${encodeURIComponent(id)}`)).json() as { rev: number; session: Session };
        return { rev: Number(data.rev) || 0, session: data.session };
      } catch (e) { if (isTransportError(e) && (e as TransportError).status === 404) return null; throw e; }
    },
    async put(id, text, baseRev) {
      const res = await call('PUT', `/api/chats/${encodeURIComponent(id)}`, { body: text, headers: { 'content-type': 'application/json', ...(baseRev !== null ? { 'x-strata-base-rev': String(baseRev) } : {}) } });
      const data = await res.json() as { rev: number; conflict?: boolean };
      return { rev: Number(data.rev) || 0, conflict: data.conflict === true };
    },
    async remove(id) { await call('DELETE', `/api/chats/${encodeURIComponent(id)}`); },
    async importMany(sessions) {
      const res = await call('POST', '/api/chats/import', { body: JSON.stringify({ sessions }), headers: { 'content-type': 'application/json' } });
      return await res.json() as { imported: string[]; skipped: string[]; failed: { id: string; error: string; code?: string }[] };
    },
  };
}

export interface HydrateResult {
  /** The chats to show: the host's, newest first, with this page's own newer ones kept. */
  sessions: Session[];
  /** [id, rev] of every chat read from the host. */
  revs: Map<string, number>;
  /** The host could not be asked: `sessions` is what the page had, nothing was moved. */
  offline: boolean;
  /** Chats moved from the browser's storage to the host this time. */
  moved: number;
  /** Chats the host refused during the move (disk full / permission) or could not be read; they stay in the browser. */
  failed: { id: string; error: string; code?: string }[];
  /** Chats that came back from the last-moment copy because the host's version was older. */
  recovered: string[];
}

/** By name, so it holds across two copies of this module (a bundle per test file). */
const isTransportError = (e: unknown): boolean => e instanceof TransportError || (e as { name?: string } | null)?.name === 'TransportError';
const newer = (a?: string | null, b?: string | null) => String(a ?? '') > String(b ?? '');

/** Read a batch of chats from the host, a few at a time. */
async function readAll(transport: ChatTransport, entries: HostChatEntry[], parallel = 4): Promise<Map<string, { rev: number; session: Session }>> {
  const out = new Map<string, { rev: number; session: Session }>();
  let next = 0;
  const worker = async () => {
    for (;;) {
      const entry = entries[next++];
      if (!entry) return;
      const got = await transport.get(entry.id);
      if (got) out.set(entry.id, got);
    }
  };
  await Promise.all(Array.from({ length: Math.min(parallel, entries.length) }, worker));
  return out;
}

/**
 * The first read of the host's chats at start. Before the host has ever been given the browser's chats (no flag), they are handed over
 * (the host takes the ones it does not have, in one request) and the flag is set; the browser's copy is left untouched. When the host
 * cannot be asked the page keeps what it has and the writer stays off.
 */
export async function hydrateChats(opts: {
  transport: ChatTransport;
  browser: StorageLike;
  /** The chats in memory now (the browser's copy at start, or what this window already holds). */
  current: Session[];
  /** A chat that must not be replaced from the host (a turn is running in it). */
  isLive?: (id: string) => boolean;
  /** Whether a chat in memory has changes the host does not have yet. */
  isDirty?: (session: Session) => boolean;
  /** For tests: this window's name, the clock and the wait while another window moves the browser's chats. */
  windowId?: string;
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
}): Promise<HydrateResult> {
  const { transport, browser } = opts;
  const isLive = opts.isLive ?? (() => false);
  const isDirty = opts.isDirty ?? (() => false);
  let moved = 0;
  const failed: HydrateResult['failed'] = [];
  const owner = opts.windowId ?? WINDOW_ID;
  const now = opts.now ?? Date.now;
  const sleep = opts.sleep ?? ((ms: number) => new Promise<void>(r => setTimeout(r, ms)));
  try {
    let list = await transport.list();
    // Read after the list: two windows opened together both see "not moved" before either has finished (#5422).
    let move = claimHandover(browser, owner, now());
    while (move === 'theirs') {
      // Another window is moving the browser's chats now: wait for it, then read what it moved. A claim it never finishes
      // (the window closed) runs out and this window moves them; one it gave up (the host refused a chat) is taken over at once.
      await sleep(HANDOVER_POLL_MS);
      move = claimHandover(browser, owner, now());
      if (move === 'done') list = await transport.list();
    }
    if (move === 'mine') {
      const mine = readFlag(browser);
      try {
        const old = readBrowserSessions(browser).sessions;
        if (old.length) {
          const result = await transport.importMany(old);
          moved = result.imported.length;
          failed.push(...result.failed);
          list = await transport.list();
        }
      } catch (e) {
        releaseHandover(browser, mine);
        throw e;
      }
      // Done only when nothing was refused: a disk-full move is tried again at the next start.
      if (failed.length) releaseHandover(browser, mine);
      else { try { browser.setItem(HOST_FLAG_KEY, '1'); } catch { /* the flag is a courtesy; the host skips what it has */ } }
    }
    const fromHost = await readAll(transport, list.chats);
    const revs = new Map([...fromHost].map(([id, v]) => [id, v.rev]));
    const mine = new Map(opts.current.map(s => [s.id, s]));
    const merged: Session[] = [];
    for (const entry of list.chats) {
      const remote = fromHost.get(entry.id);
      const local = mine.get(entry.id);
      if (!remote) { if (local) merged.push(local); continue; }
      merged.push(local && (isLive(entry.id) || isDirty(local) || newer(local.updatedAt, remote.session.updatedAt)) ? local : remote.session);
      mine.delete(entry.id);
    }
    // Chats only this page has (made while the host was being read): kept, and saved when the writer starts.
    const hostIds = new Set(list.chats.map(c => c.id));
    const onlyHere = opts.current.filter(s => !hostIds.has(s.id));
    const sessions = [...onlyHere, ...merged];
    const recovered = takeLastMomentCopy(browser, sessions, revs);
    return { sessions: recovered.sessions, revs, offline: false, moved, failed, recovered: recovered.ids };
  } catch (e) {
    if (isTransportError(e)) return { sessions: opts.current, revs: new Map(), offline: true, moved, failed, recovered: [] };
    throw e;
  }
}

/* ---------------------------------------------------------- one window moves the browser's chats */

/** How long one window's claim on the move holds; after that another window takes it over (the first one closed mid-move). */
export const HANDOVER_LEASE_MS = 30_000;
const HANDOVER_POLL_MS = 250;
const CLAIM = 'claim:';

function readFlag(browser: StorageLike): string | null {
  try { return browser.getItem(HOST_FLAG_KEY); } catch { return null; }
}

/**
 * Who moves the browser's chats to the host: 'done' (moved before), 'mine' (this window, now) or 'theirs' (another window that
 * claimed it moments ago). The claim is the flag itself ("claim:<window>:<time>"), written before the move and read back, so two
 * windows opened together do not both send the whole archive (#5422). Storage that cannot be read or written moves as before
 * (the host skips what it has).
 */
export function claimHandover(browser: StorageLike, owner: string, now: number): 'done' | 'mine' | 'theirs' {
  const v = readFlag(browser);
  if (v === '1') return 'done';
  if (v?.startsWith(CLAIM)) {
    const at = Number(v.slice(v.lastIndexOf(':') + 1));
    const who = v.slice(CLAIM.length, v.lastIndexOf(':'));
    if (who !== owner && Number.isFinite(at) && now - at < HANDOVER_LEASE_MS && now >= at) return 'theirs';
  }
  const mine = `${CLAIM}${owner}:${now}`;
  try { browser.setItem(HOST_FLAG_KEY, mine); } catch { return 'mine'; }
  return readFlag(browser) === mine ? 'mine' : 'theirs';
}

/** Gives the move back (a refused chat, the host gone) so it is tried again; never clears another window's claim or a finished move. */
function releaseHandover(browser: StorageLike, mine: string | null) {
  try { if (mine && mine.startsWith(CLAIM) && browser.getItem(HOST_FLAG_KEY) === mine) browser.removeItem(HOST_FLAG_KEY); } catch { /* best effort */ }
}

/* ---------------------------------------------------------- the last-moment copy */

/** The page is going away with chats the host has not confirmed: keep them (best effort, may not fit) for the next start. */
export function keepLastMomentCopy(storage: StorageLike, sessions: Session[]): boolean {
  try {
    if (!sessions.length) { storage.removeItem(PENDING_KEY); return true; }
    storage.setItem(PENDING_KEY, JSON.stringify(sessions));
    return true;
  } catch { return false; }
}

/** At start: chats from the last-moment copy that are newer than the host's (or the host lacks) replace the host's version in memory, to be saved. */
export function takeLastMomentCopy(storage: StorageLike, sessions: Session[], revs: Map<string, number>): { sessions: Session[]; ids: string[] } {
  let kept: Session[] = [];
  try { const raw = storage.getItem(PENDING_KEY); if (raw) { const parsed = JSON.parse(raw) as unknown; if (Array.isArray(parsed)) kept = parsed.filter((s): s is Session => !!s && typeof (s as Session).id === 'string'); } } catch { kept = []; }
  if (!kept.length) return { sessions, ids: [] };
  const byId = new Map(sessions.map(s => [s.id, s]));
  const ids: string[] = [];
  for (const copy of kept) {
    const have = byId.get(copy.id);
    if (!have || newer(copy.updatedAt, have.updatedAt)) { byId.set(copy.id, copy); ids.push(copy.id); }
  }
  try { storage.removeItem(PENDING_KEY); } catch { /* removed at the next start */ }
  if (!ids.length) return { sessions, ids };
  const order = sessions.map(s => s.id);
  const result = order.map(id => byId.get(id)!).concat([...byId.keys()].filter(id => !order.includes(id)).map(id => byId.get(id)!));
  void revs;
  return { sessions: result, ids };
}
