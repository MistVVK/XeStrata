/**
 * The person's own rules for every answer (#5084): one plain-text file, `<data>/agent/home/AGENTS.md`.
 *
 * That is the user-global AGENTS.md the Harness already reads in agent mode (its agent-instructions plugin looks in DSH_HOME, which the
 * host sets to `<data>/agent/home`), so the coding assistant picks the rules up by itself. The plain chat paths (the local model and an
 * API connection) get the same text through `promptBlock()`. Empty or missing rules add nothing at all, so the system prompt stays
 * byte-identical to the one without this feature and the engine's prompt cache is unaffected.
 *
 * Writes carry the revision the editor read: a save over a file another window changed since is refused (409), never merged silently.
 */
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

/** The Harness plugin's own limit (preset.mjs: agent-instructions maxBytes 65536); the editor keeps to it so agent mode reads all of it. */
export const RULES_MAX_BYTES = 65536;

export class RulesError extends Error {
  constructor(message, status, code) { super(message); this.status = status; this.code = code; }
}

const revisionOf = text => createHash('sha256').update(text, 'utf8').digest('hex').slice(0, 16);

export function rulesFile(dataDir) { return join(dataDir, 'agent', 'home', 'AGENTS.md'); }

export function createUserRules({ file }) {
  let cache = null; // { mtimeMs, size, text }

  function current() {
    let st;
    try { st = statSync(file); } catch { cache = null; return ''; }
    if (cache && cache.mtimeMs === st.mtimeMs && cache.size === st.size) return cache.text;
    let text = '';
    try { text = readFileSync(file, 'utf8'); } catch { text = ''; }
    if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
    cache = { mtimeMs: st.mtimeMs, size: st.size, text };
    return text;
  }

  /** What the editor shows, and the revision a save must name. */
  function read() {
    const text = current();
    return { text, revision: revisionOf(text), bytes: Buffer.byteLength(text, 'utf8'), maxBytes: RULES_MAX_BYTES, path: file };
  }

  /** Save the editor's text. `revision` is the one `read()` gave; a different file on disk now means another window saved first. */
  function write(text, revision) {
    if (typeof text !== 'string') throw new RulesError('The rules must be text.', 400, 'bad-text');
    const normalized = text.replace(/\r\n?/g, '\n');
    if (Buffer.byteLength(normalized, 'utf8') > RULES_MAX_BYTES) throw new RulesError('These rules are too long. Make them shorter than 64 KB, then save again.', 413, 'too-long');
    if (revision !== revisionOf(current())) throw new RulesError('The rules were changed in another window. Close this and open them again to see the newest version.', 409, 'changed');
    mkdirSync(dirname(file), { recursive: true });
    const tmp = `${file}.${process.pid}.tmp`;
    writeFileSync(tmp, normalized, 'utf8');
    renameSync(tmp, file);
    cache = null;
    return read();
  }

  /** The block the plain chat paths add after Strata's own prompt: '' when there are no rules, so nothing changes then. */
  function promptBlock() {
    const text = current().trim();
    return text ? `The person you are helping set these rules for every answer:\n${text}` : '';
  }

  return { read, write, promptBlock, file };
}
