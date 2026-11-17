/**
 * #5960: a picture pasted (Ctrl+V) or a file dropped while an answer is being written or a file is being read was thrown away without a word. The
 * composer now says so. What it says depends on what came in: a picture (the sentence the person met in the report) or any other file.
 */
export type BusyDrop = 'picture' | 'file';

export function busyDropKind(files: readonly { type?: string }[] | undefined | null): BusyDrop | null {
  if (!files || !files.length) return null;
  return files.some(f => (f.type ?? '').startsWith('image/')) ? 'picture' : 'file';
}
