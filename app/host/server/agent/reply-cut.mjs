/**
 * A reply the model could not finish (AGENT-LOST, 11-01): pure helpers, no I/O.
 *
 * A model reply has a ceiling (the "output budget"). A thinking model that then writes a whole file in one tool call can spend all of it
 * before the call is closed. The Harness drops a tool call that was not closed (its arguments are half a JSON text), the turn ends with
 * the reason `max-tokens`, and until now the person saw nothing: no file, no card, no sentence. Here:
 *
 *   - `replyCap(window)`      how much one reply may produce (the request's ceiling), much larger than the budget the Harness reserves
 *                             for compaction (profile.mjs `localOutputBudget`): the Harness still clamps it to what the window has left.
 *   - `ToolDraft`             what a tool call that is being written looks like so far: its name, the file it is for, a size, a tail of the text.
 *   - `cutNotice(...)`        the plain sentence for the chat, and `cutNudge(...)` the message that lets the agent carry on in smaller pieces.
 */

/** Half the window, at most 64K tokens, and never less than the budget compaction reserves. The Harness clamps it to the room left (window minus prompt). */
export function replyCap(window, reserved = 0) {
  const w = Number.isFinite(window) && window > 0 ? Math.floor(window) : 32768;
  return Math.max(Math.floor(reserved) || 0, Math.min(Math.floor(w / 2), 65536));
}

const ESCAPES = { n: '\n', t: '\t', r: '', '"': '"', '\\': '\\', '/': '/', b: '', f: '' };

/** A tail of a JSON string body, read back to text: `\n` becomes a newline, `\"` a quote. A half-written escape at the end is dropped. */
export function unescapeJsonTail(raw) {
  let text = String(raw ?? '');
  const cut = /\\(u[0-9a-fA-F]{0,3})?$/.exec(text);
  if (cut) text = text.slice(0, cut.index);
  return text.replace(/\\(u[0-9a-fA-F]{4}|[nrtbf"\\/])/g, (_, e) => (e[0] === 'u' ? String.fromCharCode(parseInt(e.slice(1), 16)) : ESCAPES[e] ?? e));
}

const TARGET = /"(?:file_path|path|filePath)"\s*:\s*"((?:[^"\\]|\\.)*)"/;
const URL_ARG = /"url"\s*:\s*"((?:[^"\\]|\\.)*)"/;
/** The long text argument of the common writing tools; the tail of the arguments is previewed after the first one of these. */
const BODY = /"(?:content|new_string|new_str|text|code|body|command|file_text)"\s*:\s*"/;

/** One tool call being written: the model's streamed arguments, kept small. */
export class ToolDraft {
  constructor(id, name = '') {
    this.id = id; this.name = name;
    this.bytes = 0; this.lines = 0;
    this.head = '';            // the first 4,000 characters: the body argument usually starts here
    this.where = '';           // the file (or address) the call is for, once seen: a model may write it before or after the text
    this.carry = '';           // the last 300 characters, so a name split between two pieces is still seen
    this.tail = '';            // the last 2,400 characters: what is being written now
    this.at = 0;               // when a frame was last sent for it
  }

  add(piece) {
    const text = String(piece ?? '');
    if (!text) return;
    this.bytes += text.length;
    for (let i = text.indexOf('\\n'); i >= 0; i = text.indexOf('\\n', i + 2)) this.lines += 1;
    if (this.head.length < 4000) this.head += text.slice(0, 4000 - this.head.length);
    this.tail = (this.tail + text).slice(-2400);
    if (!this.where) {
      const seen = this.carry + text;
      const raw = (TARGET.exec(seen) ?? URL_ARG.exec(seen))?.[1];
      if (raw !== undefined) { try { this.where = JSON.parse(`"${raw}"`); } catch { this.where = unescapeJsonTail(raw); } }
      this.carry = seen.slice(-300);
    }
  }

  get target() { return this.where; }

  /** The text of the body so far, only its last `max` characters; empty until the body argument has started. */
  preview(max = 1600) {
    const start = BODY.exec(this.head);
    if (!start) return '';
    const bodyAt = start.index + start[0].length;                    // where the body begins, counted in streamed characters
    if (bodyAt >= this.bytes) return '';
    const tailAt = this.bytes - this.tail.length;                    // where the kept tail begins
    const body = unescapeJsonTail(this.tail.slice(Math.max(0, bodyAt - tailAt)));
    return body.length > max ? body.slice(-max) : body;
  }

  /** The frame for the page. */
  frame({ done = false, ...extra } = {}) {
    return { type: 'agent-tool-draft', draft: { id: this.id, name: this.name, target: this.target, bytes: this.bytes, lines: this.lines, preview: done ? '' : this.preview(), ...extra, done } };
  }
}

const kb = n => (n >= 10_000 ? `${Math.round(n / 1000)} KB` : `${(n / 1000).toFixed(1)} KB`);

/** Calls the model was still writing when its reply was cut: [{ name, target, bytes }]. */
export function cutCalls(drafts) {
  return [...drafts].filter(d => d.bytes > 0).map(d => ({ name: d.name, target: d.target, bytes: d.bytes }));
}

const withTokens = tokens => (tokens > 0 ? ` (${Number(tokens).toLocaleString('en-US')} tokens)` : '');

/** The sentence for the person, with the catalog slots it needs: { text, key, values }. `again` is true when the agent was already asked to carry on. */
export function cutNotice({ calls = [], tokens = 0, again = false, retrying = false } = {}) {
  const call = calls.find(c => c.target) ?? calls[0] ?? null;
  const file = call?.target ? String(call.target).split(/[\\/]/).filter(Boolean).at(-1) : '';
  const what = file || (/^(write|edit)$/.test(call?.name ?? '') ? 'a file' : call?.name);
  if (call && retrying) {
    const key = 'That reply reached its length limit{0} while writing {1} ({2}), so that write did not happen. I asked the assistant to look at what is on disk and write the rest in smaller parts.';
    const values = [withTokens(tokens), what, kb(call.bytes)];
    return { key, values, text: fill(key, values) };
  }
  if (call) {
    const key = 'The reply reached its length limit{0} while writing {1} ({2}), so that write did not happen. Say “continue” and it will look at what is on disk and go on in smaller parts.';
    const values = [withTokens(tokens), what, kb(call.bytes)];
    return { key, values, text: fill(key, values) };
  }
  if (retrying) {
    const key = 'That reply reached its length limit{0} and was cut off. I asked the assistant to carry on.';
    return { key, values: [withTokens(tokens)], text: fill(key, [withTokens(tokens)]) };
  }
  const key = 'The reply reached its length limit{0} and was cut off. Say “continue” to go on.';
  return { key, values: [withTokens(tokens)], text: fill(key, [withTokens(tokens)]) };
}

export function fill(template, values) { return String(template).replace(/\{(\d+)\}/g, (_, i) => String(values[Number(i)] ?? '')); }

/** The message sent to the assistant so it carries on. It is not shown as the person's message. */
export function cutNudge({ calls = [], tokens = 0 } = {}) {
  const call = calls.find(c => c.target) ?? calls[0] ?? null;
  if (!call) {
    return 'Your last reply was cut off by the output limit before it was finished. Carry on from where it stopped without repeating what you already wrote, and keep your thinking short.';
  }
  const file = call.target || 'the file';
  return `Your last reply was cut off by the output limit${tokens > 0 ? ` (${Number(tokens).toLocaleString('en-US')} tokens)` : ''} while you were writing ${file} in one tool call, so that call did not run and nothing was changed by it. First look at what is on disk now (list the folder, read the end of the file). Do not delete or start over a file that is already complete. If it is not complete, write what is missing in several smaller parts (about 150 lines or fewer each) with edits or appends, and read the end of the file back when you are done. Keep your thinking short.`;
}
