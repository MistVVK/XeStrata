import { useSyncExternalStore } from 'react';
import type { PreviewArtifact } from './artifacts';

/**
 * Code blocks that can be shown in the sidebar preview: an HTML page, or Markdown. The block's id comes from the message and the code itself,
 * so two blocks in one answer are two previews, and the block's button can tell whether its own preview is the one open.
 */
const hash = (text: string) => { let h = 5381; for (let i = 0; i < text.length; i++) h = ((h << 5) + h + text.charCodeAt(i)) | 0; return (h >>> 0).toString(36); };

export function codePreviewKind(language: string): 'website' | 'markdown' | null {
  const name = language.trim().toLowerCase();
  return name === 'html' ? 'website' : name === 'markdown' || name === 'md' ? 'markdown' : null;
}

export function codePreview(base: string, language: string, code: string): PreviewArtifact | null {
  const kind = codePreviewKind(language);
  if (!kind) return null;
  return { id: `${base}-${hash(code)}`, kind, title: kind === 'website' ? 'HTML preview' : 'Markdown preview', content: code };
}

/** The one complete page in a finished answer, or null: exactly one fenced html block that is a whole document (a snippet stays a button). */
export function mainCodePreview(base: string, content: string): PreviewArtifact | null {
  const blocks = [...content.matchAll(/^[ \t]*```([^\n`]*)\n([\s\S]*?)\n[ \t]*```[ \t]*$/gm)].map(m => ({ language: m[1].trim(), code: m[2] })).filter(b => codePreviewKind(b.language));
  if (blocks.length !== 1 || codePreviewKind(blocks[0].language) !== 'website') return null;
  return /<!doctype\s+html|<html[\s>]/i.test(blocks[0].code) ? codePreview(base, blocks[0].language, blocks[0].code) : null;
}

let selected: string | null = null;
const listeners = new Set<() => void>();
/** The chat tells which preview is showing in the sidebar; only the code blocks' buttons listen, so no message re-renders for it. */
export function setSelectedPreviewId(id: string | null) {
  if (id === selected) return;
  selected = id;
  listeners.forEach(fn => fn());
}
export const useSelectedPreviewId = () => useSyncExternalStore(fn => { listeners.add(fn); return () => { listeners.delete(fn); }; }, () => selected, () => null);
