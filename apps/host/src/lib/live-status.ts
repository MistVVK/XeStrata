import { useSyncExternalStore } from 'react';
import { tr } from './i18n';

/**
 * The one transient line under the newest message while the model works (frames from the host: `status`, see notes-split-agent3.md).
 * Only the LATEST frame of a turn is kept; it is removed on `idle`, on `done` and `error`, and when the stream ends, so the line can never
 * show something that stopped being true.
 */
export interface StatusFrame {
  seq: number;
  phase: string;
  code: string;
  text: string;
  args?: Record<string, unknown>;
  tokens?: number;
  elapsedMs: number;
  /** When this page received the frame; elapsed time keeps counting from it between frames. */
  receivedAt: number;
}

const frames = new Map<string, StatusFrame>();
const listeners = new Set<() => void>();
const emit = () => listeners.forEach(fn => fn());

/** Take a status frame for a chat. An older frame (lower `seq`) never replaces a newer one. Returns whether the line changed. */
export function applyStatusFrame(sessionId: string, raw: Record<string, unknown>, now = Date.now()): boolean {
  const seq = Number(raw.seq);
  const phase = String(raw.phase ?? '');
  const current = frames.get(sessionId);
  if (Number.isFinite(seq) && current && seq < current.seq) return false;
  if (phase === 'idle' || !phase) { const had = frames.delete(sessionId); if (had) emit(); return had; }
  frames.set(sessionId, {
    seq: Number.isFinite(seq) ? seq : (current?.seq ?? 0) + 1,
    phase, code: String(raw.code ?? phase), text: String(raw.text ?? ''),
    args: raw.args && typeof raw.args === 'object' ? raw.args as Record<string, unknown> : undefined,
    tokens: typeof raw.tokens === 'number' ? raw.tokens : undefined,
    elapsedMs: Number(raw.elapsedMs) || 0, receivedAt: now,
  });
  emit();
  return true;
}

export function clearStatus(sessionId: string) { if (frames.delete(sessionId)) emit(); }
export const statusOf = (sessionId: string | null | undefined): StatusFrame | null => (sessionId ? frames.get(sessionId) ?? null : null);

export function useLiveStatus(sessionId: string | null | undefined): StatusFrame | null {
  return useSyncExternalStore(
    fn => { listeners.add(fn); return () => { listeners.delete(fn); }; },
    () => statusOf(sessionId),
    () => null,
  );
}

/** The playful words shown while the model is thinking, in the order they rotate. Translated through the catalog. */
export const PLAYFUL_VERBS = [
  'Thinking…', 'Pondering…', 'Cooking up an answer…', 'Mulling it over…', 'Connecting the dots…', 'Chasing an idea…',
  'Sketching it out…', 'Weighing the options…', 'Reading between the lines…', 'Brewing a reply…', 'Putting it together…', 'Following a hunch…',
] as const;

/** Phases where the playful words rotated. Since 0.2.16 (owner) the line always says what the model does now, so nothing rotates any more;
 *  the words stay in the catalog. */
export function isPlayful(_frame: StatusFrame): boolean {
  return false;
}

/** A tool's name as the person reads it: an MCP tool without its "mcp__server__" prefix. */
export const shortToolName = (name: unknown): string => String(name ?? '').replace(/^mcp__[^_]+(?:_[^_]+)*?__/, '') || String(name ?? '');
const fileName = (path: string) => path.split(/[\\/]/).filter(Boolean).pop() ?? path;
const FILE_WRITERS = new Set(['write', 'edit']);
const SHELLS = new Set(['pwsh', 'bash']);

/** The words for a frame. `rotation` counts how many times the playful word has advanced (the page advances it every few seconds). */
export function statusWords(frame: StatusFrame, rotation = 0): string {
  if (isPlayful(frame)) return tr(PLAYFUL_VERBS[((rotation % PLAYFUL_VERBS.length) + PLAYFUL_VERBS.length) % PLAYFUL_VERBS.length]);
  const a = frame.args ?? {};
  const target = typeof a.target === 'string' && a.target ? a.target : '';
  const tool = typeof a.tool === 'string' ? a.tool : '';
  // 0.2.16 (owner): what the model does now, in place of the playful words
  if (frame.phase === 'thinking') return tr('Thinking…');
  if (frame.phase === 'waiting' && frame.code === 'model-working') return tr('Waiting for the model…');
  // the model is writing a tool call: the tool's name, or the file it is writing
  if (frame.code === 'preparing-tool' && tool) return FILE_WRITERS.has(tool) && target ? tr('Writing file {0}…', [fileName(target)]) : tr('Calling {0}…', [shortToolName(tool)]);
  // a file tool runs (or waits for your answer): the file's name, not its whole path
  if (frame.code === 'tool' && FILE_WRITERS.has(tool) && target) return tr('Writing file {0}…', [fileName(target)]);
  // a tool runs: a command tool, a tool without words of its own and a tool its server runs say its name
  if (frame.code === 'tool' && tool && (SHELLS.has(tool) || !a.verb || a.verb === 'Using a tool')) return tr('Running {0}…', [shortToolName(tool)]);
  switch (frame.code) {
    case 'starting': return tr('Starting…');
    case 'reading-prompt': return tr('Reading your message…');
    case 'answering': return tr('Writing the answer…');
    case 'preparing-tool': case 'tool': {
      const verb = typeof a.verb === 'string' && a.verb ? tr(a.verb) : '';
      return verb ? `${verb}${target ? ` · ${target}` : ''}…` : tr(frame.text);
    }
    case 'waiting-approval': return tr('Waiting for your answer…');
    case 'helpers': return tr('Helpers are working ({0})…', [Number(a.n) || 1]);
    case 'goal-step': return tr('Working on the goal…');
    default: return tr(frame.text);
  }
}

/** "14s", "1m 04s": how long the turn has been going. */
export function elapsedText(startedAt: number, now: number): string {
  const total = Math.max(0, Math.round((now - startedAt) / 1000));
  return total < 60 ? `${total}s` : `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, '0')}s`;
}
