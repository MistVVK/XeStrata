import type { ActivityGroup } from './deliverables';
import type { ToolDraft } from './types';

/**
 * A tool call the model is still writing (`message.drafts`, the host's `agent-tool-draft` frames: a whole file in one call, say) drawn through the
 * open activity line: the verb and the file, its growing size and line count, and the tail of its text when that reads well.
 */

/** "18 KB", the way the "cut" sentence says it (thousands, not 1,024s), one decimal under 10 KB. */
export function draftSize(bytes: number): string {
  if (!(bytes > 0)) return '0 KB';
  if (bytes < 100) return `${Math.round(bytes)} B`;
  return bytes >= 10_000 ? `${Math.round(bytes / 1000)} KB` : `${(bytes / 1000).toFixed(1)} KB`;
}

/** The last lines of a draft's text, if they read well (words and code, not a wall of escapes); at most `count` lines of 160 characters, else none. */
export function draftTail(preview: string | undefined, count = 3): string[] {
  const lines = String(preview ?? '').replace(/\t/g, '  ').split(/\r?\n/).map(l => l.replace(/\s+$/, '')).filter(l => l.trim()).slice(-count).map(l => l.slice(0, 160));
  const text = lines.join('');
  if (text.length < 12) return [];
  const printable = [...text].filter(c => c >= ' ' && c !== '\u007f').length / text.length;
  const wordy = (text.match(/[A-Za-z]{2,}/g) ?? []).length;
  return printable > 0.95 && wordy >= 2 ? lines : [];
}

/** Whether a draft has a place for its text (a file or a command): the box is kept from the start so the line does not grow when the text arrives. */
export const draftHasTail = (d: Pick<ToolDraft, 'scope'>) => d.scope === 'write' || d.scope === 'shell';

/** The activity line for one draft: `label` is the plain sentence ("Writing pagoda.html"), already in the person's language. */
export function groupFromDraft(draft: ToolDraft, label: string): ActivityGroup {
  const tail = draftHasTail(draft) ? draftTail(draft.preview) : [];
  return { id: `draft-${draft.id}`, closed: false, line: label, added: 0, removed: 0, files: [], live: { label, bytes: draft.bytes, lines: draft.lines, ...(draftHasTail(draft) ? { tail: tail.join('\n') } : {}) } };
}
