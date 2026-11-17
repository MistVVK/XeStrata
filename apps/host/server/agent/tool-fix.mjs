/**
 * TOOLCALL-B: a reply whose tool call could not run does not end the agent's turn.
 *
 * Why: in one person's 499 agent turns, 43 ended as "completed" because the model's tool call could not run (a tool name it was never given,
 * such as exec or psh, or a call cut off before it closed). The person saw the agent stop. The Strata server already tries such a reply once
 * more by itself; when the reply it finally sends still carries `strata_rejected_tool_calls` or `strata_unread_tool_markup`, the app tells the
 * model what went wrong, as a tool error, and lets it go on. Nothing guessed ever runs: a close declared name is only named in the text.
 *
 * Pure (no I/O). The forwarder (llm-shape.mjs) reads the two fields from the reply and records them here; the bridge (agent-bridge.mjs) asks
 * at the end of a turn whether one belongs to that turn.
 *
 *   rejected   [{name, reason: "undeclared", closest_declared?}]
 *   unread     {reason: "cut_off" | "unreadable", chars}
 */

/** Corrections in a row per turn; a tool call that runs starts the count again. The next bad reply ends the turn with END_NOTE. */
export const MAX_TOOL_FIXES = 2;
/** Markers kept at most (oldest dropped first), and how long one stays usable. */
export const MARKERS_KEPT = 64;
export const MARKER_TTL_MS = 30 * 60_000;

/** The status line while the model is asked again (catalog key; guic family). */
export const FIXING_STATUS = 'Asking the model to fix its tool call…';
/** The sentence a turn ends with when the model wrote calls that could not run, twice in a row (catalog key; guic family). */
export const END_NOTE = 'The model wrote tool calls that Strata could not run, twice in a row, so Strata stopped this turn. Nothing was run. Send your message again, or choose another model.';

/** The `delta` frame the turn ends with (the page shows it as the assistant's own line and translates it by `key`, as the turn guard's notes). */
export function endNote() {
  return { type: 'delta', own: true, text: `\n\n${END_NOTE}`, key: END_NOTE, noteReason: 'tool-calls' };
}

const clean = s => String(s ?? '').replace(/\s+/g, ' ').trim().slice(0, 200);

/**
 * The message the model reads (English, model-facing; not in the catalogs). One rejected name gets the whole sentence with the declared list;
 * every further name gets its own short line. A cut-off or unreadable call gets its own sentence.
 */
export function correctionText({ rejected = null, unread = null, declared = [] } = {}) {
  const lines = [];
  const seen = new Set();
  const list = [...new Set((declared ?? []).map(clean).filter(Boolean))];
  for (const r of Array.isArray(rejected) ? rejected : []) {
    const name = clean(r?.name);
    if (!name || seen.has(name)) continue;
    seen.add(name);
    const hint = clean(r?.closest_declared);
    let line = `Tool error: no such tool: ${name}. Nothing was run.`;
    if (seen.size === 1) line += ` The tools you can call are: ${list.length ? list.join(', ') : '(none)'}.`;
    if (hint) line += ` Did you mean ${hint}?`;
    lines.push(line);
  }
  if (unread) lines.push('Tool error: your tool call could not be read (it was cut off or not complete), so nothing was run. Write the whole call again, or answer in plain text.');
  return lines.join('\n');
}

const norm = s => String(s ?? '').replace(/\r\n/g, '\n').trim();

/** Does a request (its user-role texts) carry one of the texts a turn sent? */
export function carries(users, texts) {
  const own = [...texts].map(norm).filter(Boolean);
  if (!own.length) return false;
  return (users ?? []).some(u => { const t = norm(u); return t && own.some(x => t === x || t.includes(x)); });
}

/**
 * The markers the forwarder recorded, until a turn takes one. A marker belongs to a turn when (a) one of its request's user messages is a text
 * this turn sent (its prompt, a correction, a nudge, a queued message), which keeps parallel chats apart, and (b) it was recorded after the
 * turn's newest own event (`since`: the start, the last tool card, the last text sent, the last notice), which keeps out an earlier step's reply
 * whose valid call ran anyway and a helper's reply from before. A marker that fits more than one running turn is used by none.
 */
export class MarkerStore {
  constructor({ now = Date.now, kept = MARKERS_KEPT, ttlMs = MARKER_TTL_MS } = {}) {
    this.now = now; this.kept = kept; this.ttlMs = ttlMs;
    this.list = [];
  }

  add(marker) {
    if (!marker || (!marker.rejected && !marker.unread)) return;
    this.list.push({ ...marker, at: Number.isFinite(marker.at) ? marker.at : this.now() });
    while (this.list.length > this.kept) this.list.shift();
  }

  #prune() { const old = this.now() - this.ttlMs; this.list = this.list.filter(m => m.at >= old); }

  /**
   * The newest marker for `turn` ({texts, since}); `others` are the other running turns ({texts, since}). Returns {marker} (taken out of the
   * store), {ambiguous: marker} (taken out too, acted on by nobody), or null.
   */
  take(turn, others = []) {
    this.#prune();
    for (let i = this.list.length - 1; i >= 0; i--) {
      const m = this.list[i];
      if (m.at < turn.since || !carries(m.users, turn.texts)) continue;
      this.list.splice(i, 1);
      const shared = others.some(o => m.at >= (o.since ?? 0) && carries(m.users, o.texts));
      return shared ? { ambiguous: m } : { marker: m };
    }
    return null;
  }

  get size() { return this.list.length; }
}
