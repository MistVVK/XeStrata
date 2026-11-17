import { api } from './client';
import { appModeKnown, loadAppMode, type AppMode } from './app-mode';

export interface ExternalDeps {
  known: () => AppMode | null;
  load: () => Promise<AppMode>;
  post: (path: string, body: unknown) => Promise<unknown>;
  open: (url: string) => void;
}

/**
 * Opens a web address in the person's own browser (#4721). The app window has no handler for a new window, so window.open does nothing
 * there: the host hands the address to the default browser. In a browser tab the browser opens it itself. The answer to "which window am I
 * in" is read once at start, so the click can open the tab at once (a popup opened late can be blocked); if it has not arrived yet, the
 * click waits for it.
 */
export async function openExternal(url: string, deps: Partial<ExternalDeps> = {}): Promise<void> {
  const d: ExternalDeps = {
    known: appModeKnown,
    load: loadAppMode,
    post: (path, body) => api.post(path, body),
    open: u => { window.open(u, '_blank', 'noopener,noreferrer'); },
    ...deps,
  };
  const mode = d.known() ?? await d.load();
  if (mode.mode === 'window') {
    try { await d.post('/api/artifacts/open', { action: 'open', url }); return; }
    catch { /* the host could not: the page tries as well */ }
  }
  d.open(url);
}
