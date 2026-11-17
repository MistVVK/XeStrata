import { useState } from 'react';
import { api, ApiError } from '../lib/client';
import { localeTag, tr } from '../lib/i18n';
import { Btn, Dialog } from './ui';

/** What the host answers for /api/rules (server/user-rules.mjs). */
interface Rules { text: string; revision: string; bytes: number; maxBytes: number }

export const RULES_MAX_BYTES = 65536;
const TOO_LONG = 'These rules are too long. Make them shorter than 64 KB, then save again.';
const CHANGED = 'The rules were changed in another window. Close this and open them again to see the newest version.';
export const ruleBytes = (text: string) => new TextEncoder().encode(text.replace(/\r\n?/g, '\n')).length;
// rounded up to a tenth, so a few rules never read as 0.0 KB
const kb = (bytes: number) => `${(Math.ceil(bytes / 102.4) / 10).toLocaleString(localeTag(), { minimumFractionDigits: 1, maximumFractionDigits: 1 })} KB`;

/**
 * Settings > General > Your rules (#5084): the person's own rules for every chat. The host keeps them in the AGENTS.md the coding
 * assistant already reads and adds them to the plain chat's prompt too. A save names the version it opened, so a newer save from
 * another window is never overwritten.
 */
export function UserRulesSetting() {
  const [open, setOpen] = useState<Rules | null>(null);
  const [text, setText] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const bytes = ruleBytes(text), tooLong = bytes > RULES_MAX_BYTES;

  async function start() {
    setError('');
    try { const rules = await api.get<Rules>('/api/rules'); setText(rules.text); setOpen(rules); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  }
  async function save() {
    if (!open || tooLong) return;
    setBusy(true); setError('');
    try { await api.put<Rules>('/api/rules', { text, revision: open.revision }); setOpen(null); }
    catch (e) { setError(e instanceof ApiError && e.status === 409 ? CHANGED : e instanceof ApiError && e.status === 413 ? TOO_LONG : e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  return <>
    <Btn size="sm" onClick={() => void start()}>{tr("Edit rules")}</Btn>
    {!open && error ? <p className="small" role="alert">{tr(error)}</p> : null}
    <Dialog open={!!open} title={tr("Your rules")} onClose={() => setOpen(null)} actions={<><Btn onClick={() => setOpen(null)}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={busy || tooLong} onClick={() => void save()}>{tr("Save rules")}</Btn></>}>
      <label className="input-field"><span className="sr-only">{tr("Your rules")}</span><textarea className="st-input" rows={12} aria-label={tr("Your rules")} value={text} placeholder={tr("Write one rule per line.")} onChange={e => setText(e.target.value)} /></label>
      <p className="small muted" data-rules-bytes={bytes}>{tr("{0} of 64 KB used", [kb(bytes)])}</p>
      {tooLong || error ? <p className="small" role="alert">{tr(tooLong ? TOO_LONG : error)}</p> : null}
    </Dialog>
  </>;
}
