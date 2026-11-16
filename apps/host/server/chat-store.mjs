/**
 * The chats, saved by the host as files (owner decision 11-04: the browser's 5 MB storage limit is gone).
 *
 *   <data folder>/chats/index.json         { v: 1, chats: [{ id, rev, createdAt, updatedAt, bytes, title }] }
 *   <data folder>/chats/<id>.json          { "v": 1, "rev": N, "savedAt": ISO, "session": { ...the chat exactly as the page sent it... } }
 *   <data folder>/chats/conflicts/         the version a later save displaced when two windows saved the same chat from an older copy
 *
 * - One file per chat. A save writes a temporary file next to it, flushes it and renames it over the chat: a crash or a full disk
 *   in the middle leaves the older file whole. Temporary files left by a crash are removed at the next start.
 * - `rev` counts the saves of a chat. A save that names a `baseRev` older than the stored one (a second window saved first) is still
 *   written, but the version it displaces is kept in conflicts/ first, so nothing is lost.
 * - The index is a derived list. At start it is rebuilt from the chat files, which own their revisions even if the last index write
 *   failed; a file that cannot be read is reported and left exactly as it is.
 * - Nothing is cut: the chat is stored byte for byte as received (a text splice, no re-serialising).
 * - Owner-only: the folder is created private (POSIX 0700 and files 0600; on Windows the folder gets a protected ACL that its files
 *   inherit), the same level as the rest of the profile.
 */
import { mkdirSync, readdirSync, readFileSync, statSync, unlinkSync } from 'node:fs';
import { open, readFile, rename, unlink, mkdir, copyFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';

export const CHAT_FORMAT = 1;
const ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;
const RESERVED = /^(con|prn|aux|nul|com\d|lpt\d|index|import|conflicts)(\.|$)/i;
const KEEP_CONFLICTS = 5;

export class ChatStoreError extends Error {
  constructor(message, status, code) { super(message); this.status = status; this.code = code; }
}

/** The disk operations, replaceable in tests to make a disk full, a permission failure or a crash in the middle of a write. */
export const realIo = {
  async writeFile(path, text) {
    const handle = await open(path, 'w', 0o600);
    try { await handle.writeFile(text, 'utf8'); await handle.sync(); } finally { await handle.close(); }
  },
  rename, unlink, copyFile, readFile, readdir, mkdir,
};

export const validChatId = id => typeof id === 'string' && ID.test(id) && !RESERVED.test(id);

/** Map a failed disk operation to what the page is told: a full disk and a refused write are different words for the same "not saved". */
export function storeFailure(error) {
  if (error instanceof ChatStoreError) return error;
  const code = error?.code;
  if (code === 'ENOSPC' || code === 'EDQUOT' || code === 'EFBIG') return new ChatStoreError('The disk is full.', 507, 'disk-full');
  if (code === 'EACCES' || code === 'EPERM' || code === 'EROFS') return new ChatStoreError('Strata may not write its chat folder.', 403, 'permission');
  return new ChatStoreError(error?.message ?? 'The chat could not be written.', 500, code ?? 'write-failed');
}

export function createChatStore({ dir, io = realIo, privateDir = () => {}, now = () => new Date() }) {
  const root = join(dir, 'chats');
  const conflictsDir = join(root, 'conflicts');
  mkdirSync(root, { recursive: true, mode: 0o700 });
  try { privateDir(root); } catch { /* the folder is inside the profile; the tighter ACL is best effort */ }
  let tmpCount = 0;
  let index = new Map();   // id -> entry
  const damaged = new Set();

  const fileOf = id => join(root, `${id}.json`);
  const bytesOf = text => Buffer.byteLength(text, 'utf8');
  const entryOf = (id, rev, session, bytes) => ({ id, rev, createdAt: session?.createdAt ?? null, updatedAt: session?.updatedAt ?? null, bytes, title: typeof session?.title === 'string' ? session.title.slice(0, 200) : '' });

  // A chain, so saves of one host never overlap: the file, the index and the conflicts folder change one save at a time.
  let chain = Promise.resolve();
  const queue = work => { const run = chain.then(work, work); chain = run.catch(() => {}); return run; };

  function loadSync() {
    // Temporary files from a write that never finished.
    let names = [];
    try { names = readdirSync(root); } catch { /* empty */ }
    for (const name of names) if (/\.tmp-\d+-\d+$/.test(name)) { try { unlinkSync(join(root, name)); } catch { /* left */ } }
    const files = names.filter(n => n.endsWith('.json') && n !== 'index.json' && !/\.tmp-/.test(n)).map(n => n.slice(0, -5)).filter(validChatId);
    const map = new Map();
    // A chat rename can succeed before its index write fails or the host exits. Even a valid index can be behind for an existing id:
    // recover from every chat envelope so the next save cannot reuse a revision or miss an older window's conflict.
    for (const id of files) {
      try {
        const parsed = JSON.parse(readFileSync(fileOf(id), 'utf8'));
        if (parsed?.v !== CHAT_FORMAT || !parsed.session || parsed.session.id !== id) throw new Error('not a chat file');
        map.set(id, entryOf(id, Number(parsed.rev) || 1, parsed.session, statSync(fileOf(id)).size));
      } catch { damaged.add(id); }
    }
    index = map;
  }
  loadSync();

  async function writeAtomic(path, text) {
    const temp = `${path}.tmp-${process.pid}-${++tmpCount}`;
    try {
      await io.writeFile(temp, text);
      for (let attempt = 0; ; attempt++) {
        try { await io.rename(temp, path); break; }
        catch (e) { if (attempt >= 5 || !['EPERM', 'EBUSY', 'EACCES'].includes(e?.code)) throw e; await new Promise(r => setTimeout(r, 40 * (attempt + 1))); }   // an antivirus scan holds the file for a moment on Windows
      }
    } catch (e) {
      try { await io.unlink(temp); } catch { /* nothing to remove */ }
      throw e;
    }
  }

  async function writeIndex() {
    const list = [...index.values()];
    try { await writeAtomic(join(root, 'index.json'), JSON.stringify({ v: CHAT_FORMAT, savedAt: now().toISOString(), chats: list })); }
    catch { /* only a list: it is rebuilt from the files when it does not match */ }
  }

  async function keepDisplaced(id, rev) {
    try {
      await io.mkdir(conflictsDir, { recursive: true });
      await io.copyFile(fileOf(id), join(conflictsDir, `${id}.rev${rev}.json`));
      const mine = (await io.readdir(conflictsDir)).filter(n => n.startsWith(`${id}.rev`)).sort((a, b) => Number(a.match(/rev(\d+)/)?.[1]) - Number(b.match(/rev(\d+)/)?.[1]));
      for (const old of mine.slice(0, Math.max(0, mine.length - KEEP_CONFLICTS))) { try { await io.unlink(join(conflictsDir, old)); } catch { /* left */ } }
    } catch (e) {
      throw storeFailure(e);   // not saved rather than saved over the only copy of the other window's work
    }
  }

  return {
    root,
    /** The chats, newest first. */
    list() {
      const chats = [...index.values()].sort((a, b) => String(b.createdAt ?? '').localeCompare(String(a.createdAt ?? '')) || a.id.localeCompare(b.id));
      return { v: CHAT_FORMAT, chats, damaged: [...damaged] };
    },
    has: id => index.has(id),
    rev: id => index.get(id)?.rev ?? 0,
    /** The stored file text of a chat ({"v","rev","savedAt","session"}), or null. */
    async read(id) {
      if (!validChatId(id) || !index.has(id)) return null;
      try { return await io.readFile(fileOf(id), 'utf8'); }
      catch (e) { if (e?.code === 'ENOENT') { index.delete(id); return null; } throw storeFailure(e); }
    },
    /**
     * Save one chat. `sessionText` is the chat's JSON as the page wrote it. Resolves { id, rev, bytes, conflict }.
     * Rejects with a ChatStoreError (status 400 for a chat that is not a chat, 507 disk full, 403 refused, 500 other): nothing was changed.
     */
    save(id, sessionText, { baseRev = null } = {}) {
      if (!validChatId(id)) return Promise.reject(new ChatStoreError('That is not a chat id.', 400, 'bad-id'));
      let session;
      try { session = JSON.parse(sessionText); } catch { return Promise.reject(new ChatStoreError('The chat must be JSON.', 400, 'bad-json')); }
      if (!session || typeof session !== 'object' || Array.isArray(session) || session.id !== id) return Promise.reject(new ChatStoreError('The chat does not match its id.', 400, 'bad-chat'));
      const body = sessionText.trim();
      return queue(async () => {
        const current = index.get(id);
        const rev = (current?.rev ?? 0) + 1;
        let conflict = false;
        if (current && baseRev !== null && Number(baseRev) !== current.rev) { conflict = true; await keepDisplaced(id, current.rev); }
        const text = `{"v":${CHAT_FORMAT},"rev":${rev},"savedAt":${JSON.stringify(now().toISOString())},"session":${body}}`;
        try { await writeAtomic(fileOf(id), text); } catch (e) { throw storeFailure(e); }
        const bytes = bytesOf(text);
        index.set(id, entryOf(id, rev, session, bytes));
        damaged.delete(id);
        await writeIndex();
        return { id, rev, bytes, conflict };
      });
    },
    /** Delete a chat (the person deleted it). A chat that is not there is already gone. */
    remove(id) {
      if (!validChatId(id)) return Promise.reject(new ChatStoreError('That is not a chat id.', 400, 'bad-id'));
      return queue(async () => {
        try { await io.unlink(fileOf(id)); } catch (e) { if (e?.code !== 'ENOENT') throw storeFailure(e); }
        const had = index.delete(id);
        await writeIndex();
        return { id, removed: had };
      });
    },
    /** Chats from the browser's old storage: every one the host does not have yet is saved; the ones it has are left alone. */
    async importMany(sessions) {
      const imported = [], skipped = [], failed = [];
      for (const session of Array.isArray(sessions) ? sessions : []) {
        const id = session?.id;
        if (!validChatId(id)) { failed.push({ id: String(id ?? ''), error: 'not a chat id' }); continue; }
        if (index.has(id)) { skipped.push(id); continue; }
        try { await this.save(id, JSON.stringify(session)); imported.push(id); }
        catch (e) { failed.push({ id, error: e.message, code: e.code }); if (e.code === 'disk-full') break; }
      }
      return { imported, skipped, failed };
    },
    /** Wait until every save queued so far is on disk (tests, shutdown). */
    idle: () => chain,
  };
}
