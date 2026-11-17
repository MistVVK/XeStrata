import { useCallback, useEffect, useRef, useState } from 'react';
import { copyText } from './clipboard';

/** How long a copy button shows that it worked (#4748). */
export const COPIED_MS = 1800;
/** A menu item shows it a little shorter and then closes its menu. */
export const COPIED_MENU_MS = 900;

/**
 * One behaviour for every Copy control: copy the text (with the older fallback of copyText; only a failure gets a toast), and when it worked,
 * `copied` is true for a moment so the control itself can show it. A failed copy leaves the control as it was and the toast says so.
 * The text may be a function, for a value that has to be fetched first.
 */
export function useCopy(hold = COPIED_MS) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);
  const copy = useCallback(async (text: string | (() => string | Promise<string>), label: string): Promise<boolean> => {
    // The toast store needs a browser, so it is loaded when someone copies: a renderer using this hook must render without one.
    const { pushToast } = await import('./store');
    let value: string;
    try { value = typeof text === 'function' ? await text() : text; }
    catch { pushToast('warn', 'Couldn’t copy'); return false; }
    // A copy that worked is shown on the control itself (the icon turns into a check): no "Copied" popover at the corner. A copy that failed still says so.
    const ok = await copyText(value, label, { toast: (tone, title, body) => { if (tone !== 'success') pushToast(tone, title, body); } });
    if (ok) {
      setCopied(true);
      clearTimeout(timer.current);
      timer.current = setTimeout(() => setCopied(false), hold);
    }
    return ok;
  }, [hold]);
  return { copied, copy };
}
