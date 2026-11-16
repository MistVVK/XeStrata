import type { Session } from './types';

/**
 * Saving the chats: by the app host, as files (owner decision 11-04; #5192, #3604, #3603).
 *
 * The chats used to live in the browser's localStorage, which holds about 5 MB for a whole origin. To fit, the writer first cut
 * long fields and dropped old messages and chats (#5192), and after that fix a full storage meant chats that could not be saved
 * at all. Now the host keeps one file per chat in the profile's data folder (server/chat-store.mjs) and the page only sends the
 * chats that changed. The write side keeps the rules of the renderer-memory fix (#3604, #3603):
 *
 *  1. Coalesce: the newest state wins and at most one save happens per `delayMs`. The caller flushes when a turn ends or the
 *     page goes away.
 *  2. Only the chats that changed are serialised again (a chat is replaced by a new object when it changes, the store never
 *     mutates), so the previous JSON of every other chat is reused by object identity, and only those are sent.
 *
 * And, new with the files:
 *  - A chat is sent exactly as it is (JSON.stringify of the chat): no clipping, no dropped messages or chats.
 *  - A save the host refuses (disk full, no permission) or cannot be asked (the host is not answering) leaves the chat whole in
 *    memory and marked unsaved; it is tried again at the next change and on a timer, and the failure is reported through
 *    `onResult` so the caller can say so and offer the unsaved chats (`unsaved()`, `exportSessions`).
 *  - A chat removed from the list is removed on the host. Only chats this writer has seen are ever removed.
 *  - The writer does not save before `enable()` (the first read of the host's chats has to land first, so an older copy in the
 *    page never replaces a newer file).
 *
 * The browser's old storage is read once for the move to files (`readBrowserSessions`) and is never written or emptied again:
 * it stays as a fallback for one release.
 */

/** The old whole-list entry of the released versions (0.2.10 and before). */
export const LEGACY_SESSIONS_KEY = 'strata.sessions.v1';
/** The order of the chats in the short-lived per-chat browser layout (this branch before the files). */
export const SESSION_INDEX_KEY = 'strata.sessions.v2';
/** One chat per entry in that layout: this prefix plus the chat id. */
export const SESSION_KEY_PREFIX = 'strata.session.v2.';
/** Set when the browser's chats were handed to the host. After it, the host's files are the only truth. */
export const HOST_FLAG_KEY = 'strata.chats.host.v1';
/** Chats that were not yet on the host when the page went away (a last-moment copy, taken back at the next start). */
export const PENDING_KEY = 'strata.chats.pending.v1';

/** The part of `Storage` that is read from. */
export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  key(index: number): string | null;
  readonly length: number;
}

/** The JSON text sent for one chat: the chat exactly as it is. */
export function serializeSession(session: Session): string {
  return JSON.stringify(session);
}

/** The chats as one JSON text, for keeping a copy by hand when they cannot be saved. */
export function exportSessions(sessions: Session[]): string {
  return JSON.stringify(sessions);
}

/* ------------------------------------------------------------- the browser's old storage (read only) */

function readIndex(storage: StorageLike): string[] {
  try {
    const raw = storage.getItem(SESSION_INDEX_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as { ids?: unknown };
    return Array.isArray(parsed.ids) ? parsed.ids.filter((id): id is string => typeof id === 'string') : [];
  } catch {
    return [];
  }
}

/**
 * Every chat in the browser's storage, from both layouts: one entry per chat (strata.session.v2.<id>) and the old whole list
 * (strata.sessions.v1). Where a chat is in both, the per-chat entry wins. Nothing is written, moved or removed.
 */
export function readBrowserSessions(storage: StorageLike): { sessions: Session[]; damaged: string[] } {
  const damaged: string[] = [];
  const found = new Map<string, Session>();
  try {
    const raw = storage.getItem(LEGACY_SESSIONS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) {
        parsed.forEach((entry, n) => {
          if (!entry || typeof entry !== 'object') return;
          const s = entry as Session;
          const id = typeof s.id === 'string' && s.id ? s.id : `legacy_${n}`;
          found.set(id, id === s.id ? s : { ...s, id });
        });
      } else damaged.push(LEGACY_SESSIONS_KEY);
    }
  } catch { damaged.push(LEGACY_SESSIONS_KEY); }
  try {
    const keys: string[] = [];
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (key && key.startsWith(SESSION_KEY_PREFIX)) keys.push(key);
    }
    for (const key of keys) {
      try {
        const parsed = JSON.parse(storage.getItem(key) ?? 'null') as Session | null;
        if (parsed && typeof parsed === 'object' && typeof parsed.id === 'string') found.set(parsed.id, parsed);
        else damaged.push(key);
      } catch { damaged.push(key); }
    }
  } catch { /* storage not readable: nothing to load */ }
  const rank = new Map(readIndex(storage).map((id, i) => [id, i]));
  const sessions = [...found.values()].sort((a, b) => {
    const ra = rank.get(a.id), rb = rank.get(b.id);
    if (ra !== undefined && rb !== undefined) return ra - rb;
    if (ra !== undefined) return -1;
    if (rb !== undefined) return 1;
    return String(b.createdAt ?? '') < String(a.createdAt ?? '') ? -1 : 1;
  });
  return { sessions, damaged };
}

/* ------------------------------------------------------------- the host */

export class TransportError extends Error {
  constructor(message: string, readonly status: number, readonly code: string | undefined, readonly network = false) { super(message); this.name = 'TransportError'; }
}

export interface HostChatEntry { id: string; rev: number; createdAt?: string | null; updatedAt?: string | null }

/** What the page asks of the host about chats (chat-sync.ts makes the real one; a test passes a fake). */
export interface ChatTransport {
  list(): Promise<{ chats: HostChatEntry[]; damaged?: string[]; newer?: string[] }>;
  get(id: string): Promise<{ rev: number; session: Session } | null>;
  put(id: string, text: string, baseRev: number | null): Promise<{ rev: number; conflict: boolean }>;
  remove(id: string): Promise<void>;
  importMany(sessions: Session[]): Promise<{ imported: string[]; skipped: string[]; failed: { id: string; error: string; code?: string }[] }>;
}

export type FailureReason = 'disk-full' | 'permission' | 'offline' | 'other';

export interface SaveFailure {
  id: string;
  reason: FailureReason;
  message: string;
}

export interface SaveResult {
  ok: boolean;
  failed: SaveFailure[];
  /** How many chats were saved by this pass. */
  written: number;
  deleted: number;
}

export function failureReason(error: unknown): FailureReason {
  const e = error as Partial<TransportError> | null;
  if (e && e.name === 'TransportError') {
    if (e.network) return 'offline';
    if (e.code === 'disk-full' || e.status === 507) return 'disk-full';
    if (e.code === 'permission' || e.status === 403) return 'permission';
  }
  return 'other';
}

export interface SessionWriterOptions {
  transport: ChatTransport;
  delayMs?: number;
  /**
   * #5923: a scheduled save (a reply being written) waits long enough that the chats it rewrites average at most this many bytes a second;
   * never less than delayMs, never more than maxDelayMs. A turn's end and the page going away still save at once (flush).
   */
  bytesPerSecond?: number;
  maxDelayMs?: number;
  /** After a pass with failures the unsaved chats are tried again after this long. */
  retryMs?: number;
  /** Counts a serialisation of one chat; for tests. */
  onSerialize?: (session: Session) => void;
  /** Called after every pass, with what was saved and which chats the host did not take. */
  onResult?: (result: SaveResult) => void;
  /** The chats the host already has: [id, rev]. Only these (and chats this writer saved) are ever removed from it. */
  known?: Iterable<[string, number]>;
  /** Start saving at once. The page leaves it off until the host's chats have been read. */
  enabled?: boolean;
}

export function createSessionWriter(options: SessionWriterOptions) {
  const transport = options.transport;
  const delayMs = options.delayMs ?? 1_000;
  const retryMs = options.retryMs ?? 8_000;
  const bytesPerSecond = options.bytesPerSecond ?? 64 * 1024;
  const maxDelayMs = Math.max(delayMs, options.maxDelayMs ?? 5_000);
  let scheduled: Session[] = [];
  let enabled = options.enabled ?? true;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let pending: Session[] | null = null;
  let lastSessions: Session[] = [];
  let running: Promise<SaveResult> | null = null;
  let rerun = false;
  let lastResult: SaveResult = { ok: true, failed: [], written: 0, deleted: 0 };
  let failures = 0;
  // Chat object -> its JSON. WeakMap: a replaced chat is forgotten with its object.
  const cache = new WeakMap<Session, string>();
  // The JSON of each chat as the host last confirmed it (or as it was read from the host).
  const sent = new Map<string, string>();
  const revs = new Map<string, number>(options.known ?? []);
  const known = new Set<string>(revs.keys());
  const failedIds = new Set<string>();

  const textOf = (s: Session): string => {
    let text = cache.get(s);
    if (text === undefined) {
      options.onSerialize?.(s);
      text = serializeSession(s);
      cache.set(s, text);
    }
    return text;
  };

  async function pass(): Promise<SaveResult> {
    const sessions = pending;
    pending = null;
    if (!sessions) return lastResult;
    lastSessions = sessions;
    const ids = new Set(sessions.map(s => s.id));
    const failed: SaveFailure[] = [];
    let written = 0, deleted = 0;

    // Removed chats first.
    for (const id of [...known]) {
      if (ids.has(id)) continue;
      try { await transport.remove(id); known.delete(id); sent.delete(id); revs.delete(id); failedIds.delete(id); deleted++; }
      catch (error) { failed.push({ id, reason: failureReason(error), message: error instanceof Error ? error.message : String(error) }); }
    }
    for (const s of sessions) {
      const text = textOf(s);
      if (sent.get(s.id) === text) { failedIds.delete(s.id); continue; }
      try {
        const saved = await transport.put(s.id, text, revs.get(s.id) ?? null);
        revs.set(s.id, saved.rev); known.add(s.id); sent.set(s.id, text); failedIds.delete(s.id);
        written++;
      } catch (error) {
        // The host keeps the last good file of this chat; the chat stays whole in memory and is tried again.
        failedIds.add(s.id);
        failed.push({ id: s.id, reason: failureReason(error), message: error instanceof Error ? error.message : String(error) });
      }
    }
    lastResult = { ok: failedIds.size === 0 && failed.length === 0, failed, written, deleted };
    if (retryTimer !== null) { clearTimeout(retryTimer); retryTimer = null; }
    if (!lastResult.ok) {
      // Tried again, a little later each time (up to a minute); the timer never keeps a closing page or process alive.
      failures = Math.min(failures + 1, 6);
      retryTimer = setTimeout(() => { retryTimer = null; if (pending === null) pending = lastSessions; void flush(); }, Math.min(retryMs * 2 ** (failures - 1), 60_000));
      (retryTimer as unknown as { unref?: () => void }).unref?.();
    } else failures = 0;
    options.onResult?.(lastResult);
    return lastResult;
  }

  const flush = (): Promise<SaveResult> => {
    if (timer !== null) { clearTimeout(timer); timer = null; }
    if (!enabled) return Promise.resolve(lastResult);
    // Nothing new, but a chat the host refused earlier: a flush (a turn ended, the page is going away) tries it again.
    if (pending === null && failedIds.size) pending = lastSessions;
    if (running) { rerun = true; return running; }
    running = (async () => {
      let result = lastResult;
      do { rerun = false; result = await pass(); } while (rerun || pending !== null);
      return result;
    })().finally(() => { running = null; });
    return running;
  };

  /**
   * #5923: each save rewrites a changed chat whole, so a reply streamed into a 1 MB chat wrote about 1 MB a second, growing with the chat. The
   * wait before a scheduled save grows with the size of the chats that changed (their last saved size): a small chat still saves within a
   * second; a chat of about 330 KB or more every 5 s (at most 5 s of a streaming reply to lose if the app crashes; a turn's end saves at once).
   */
  const delayFor = (sessions: Session[]): number => {
    const before = new Map(scheduled.map(s => [s.id, s] as const));
    let bytes = 0;
    for (const s of sessions) if (before.get(s.id) !== s) bytes += sent.get(s.id)?.length ?? 0;
    return Math.max(delayMs, Math.min(maxDelayMs, Math.round((bytes / bytesPerSecond) * 1000)));
  };
  const schedule = (sessions: Session[]) => {
    const wait = timer === null ? delayFor(sessions) : 0;
    scheduled = sessions;
    pending = sessions;
    if (enabled && timer === null) timer = setTimeout(() => { timer = null; void flush(); }, wait);
  };

  return {
    schedule,
    flush,
    hasPending: () => pending !== null || running !== null,
    /** Start saving (the host's chats have been read). Whatever was scheduled meanwhile is saved now. */
    enable() { enabled = true; if (pending !== null) void flush(); },
    /** The chats whose newest state is not on the host. */
    unsaved: (): Session[] => lastSessions.filter(s => failedIds.has(s.id)),
    /** A chat that was read from the host (or arrived from another window): its JSON and revision are what the host has. */
    adopt(session: Session, rev: number | null) { known.add(session.id); if (rev === null) revs.delete(session.id); else revs.set(session.id, rev); sent.set(session.id, serializeSession(session)); failedIds.delete(session.id); },
    /** A chat the host no longer has (removed from another window). */
    forget(id: string) { known.delete(id); revs.delete(id); sent.delete(id); failedIds.delete(id); },
    rev: (id: string) => revs.get(id) ?? 0,
    /** True when the page holds a newer state of this chat than the host has (scheduled, in flight or refused). */
    dirty(session: Session) { return sent.get(session.id) !== textOf(session); },
    /** The chats not confirmed by the host right now: what to keep if the page goes away this instant. */
    notOnHost(): Session[] {
      const list = pending ?? lastSessions;
      return list.filter(s => sent.get(s.id) !== textOf(s));
    },
  };
}
