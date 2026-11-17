/**
 * Copy text for the person. The browser clipboard call can be refused (the desktop web view asks for permission, or has no clipboard
 * object); the older copy command through a hidden field still works there. The answer is always a toast: "Copied", or one plain
 * sentence and what to do next (#3731).
 */
export interface ClipboardDeps {
  clipboard?: { writeText(text: string): Promise<void> } | null;
  legacyCopy?: (text: string) => boolean;
  toast: (tone: 'success' | 'warn', title: string, body?: string) => void;
}

function legacyCopy(text: string): boolean {
  if (typeof document === 'undefined') return false;
  const field = document.createElement('textarea');
  field.value = text;
  field.setAttribute('readonly', '');
  field.style.cssText = 'position:fixed;top:0;left:0;opacity:0;pointer-events:none';
  document.body.appendChild(field);
  const active = document.activeElement as HTMLElement | null;
  try {
    field.select();
    return document.execCommand('copy');
  } catch { return false; }
  finally { field.remove(); active?.focus?.(); }
}

export async function copyText(text: string, label: string, deps: ClipboardDeps): Promise<boolean> {
  const clipboard = deps.clipboard === undefined ? (typeof navigator === 'undefined' ? null : navigator.clipboard) : deps.clipboard;
  let done = false;
  try { if (clipboard) { await clipboard.writeText(text); done = true; } } catch { /* the fallback is next */ }
  if (!done) { try { done = (deps.legacyCopy ?? legacyCopy)(text); } catch { done = false; } }
  if (done) deps.toast('success', 'Copied', label);
  else deps.toast('warn', 'Couldn’t copy', 'Select the text and copy it with Ctrl+C.');
  return done;
}
