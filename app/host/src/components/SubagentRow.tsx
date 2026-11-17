import { tr } from '../lib/i18n';
import { useRef, useState } from 'react';
import { Btn, Dialog, type IconName } from './ui';
import { api } from '../lib/client';
import { plainFailureText } from '../lib/plain-errors';
import { compactTokens, elapsed } from '../lib/chat-metrics';
import type { Session } from '../lib/types';
import { helperRanLong } from '../lib/helper-limit';
type Agent = NonNullable<Session['agents']>[number];
const states: Record<Agent['status'], {
    label: string;
    icon: IconName;
}> = {
    working: { label: 'In progress', icon: 'clock' }, completed: { label: 'Completed', icon: 'check' },
    failed: { label: 'Failed', icon: 'error' }, queued: { label: 'Queued', icon: 'clock' }, cancelled: { label: 'Cancelled', icon: 'x' },
};
export function SubagentRow({ agent: a, now = Date.now(), sessionId }: {
    agent: Agent;
    now?: number;
    /** The chat the helper belongs to: with it the row can take a message for the helper (#4749). */
    sessionId?: string;
}) {
    const [log, setLog] = useState(false);
    const [writing, setWriting] = useState(false), [draft, setDraft] = useState(''), [busy, setBusy] = useState(false), [sent, setSent] = useState(false), [problem, setProblem] = useState('');
    // A working helper gets it at its next step (steer); one that has finished or waits gets a new turn with it (queue).
    async function send() {
        const message = draft.trim();
        if (!message || !sessionId || busy) return;
        setBusy(true); setProblem(''); setSent(false);
        try { await api.post('/api/chat/helper', { sessionId, helperId: a.id, message, delivery: a.status === 'working' ? 'steer' : 'queue' }); setDraft(''); setSent(true); }
        catch (e) { setProblem(plainFailureText(e instanceof Error ? e.message : String(e)).message); }
        finally { setBusy(false); }
    }
    const observedAt = useRef(Date.now());
    const duration = a.durationMs + (a.status === 'working' ? Math.max(0, now - (a.updatedAt ?? observedAt.current)) : 0);
    const state = states[a.status] ?? states.queued;
    return <div className="subagent-row" data-status={a.status}>
    <span className="subagent-row__dot" aria-label={tr(state.label)}/>
    <div className="subagent-row__copy"><div className="subagent-row__title"><strong title={a.name}>{a.name}</strong><span className="subagent-row__status">{tr(state.label)}</span></div><p title={a.description}>{tr(a.description)}</p>{tr(helperRanLong(a.status, duration) ? <p className="small subagent-row__note" role="status">{tr("This helper has been working for over 30 minutes.")}</p> : null)}{tr(a.status === 'failed' || a.steps?.length ? <button className="text-action" onClick={() => setLog(true)}>{tr("View log")}</button> : null)}{tr(sessionId && a.status !== 'queued' && a.status !== 'cancelled' ? <>{writing ? <form className="subagent-row__reply" onSubmit={e => { e.preventDefault(); void send(); }}><input className="st-input" value={draft} maxLength={8000} aria-label={tr("Message to {0}", [a.name])} placeholder={tr("Write to {0}\u2026", [a.name])} disabled={busy} onChange={e => { setDraft(e.target.value); setSent(false); }}/><Btn size="sm" variant="primary" type="submit" disabled={busy || !draft.trim()}>{tr("Send")}</Btn></form> : <button className="text-action" onClick={() => setWriting(true)}>{tr("Message this helper")}</button>}{tr(sent ? <p className="small subagent-row__note" role="status">{tr("Sent to {0}.", [a.name])}</p> : null)}{tr(problem ? <p className="small inline-error subagent-row__note" role="alert">{problem}</p> : null)}</> : null)}</div>
    <div className="subagent-row__metrics"><span>{tr(compactTokens(a.tokens))}{tr(" tok")}</span><span>{tr(elapsed(duration))}</span></div>
  <Dialog open={log} title={tr("{0} \u00B7 log", [a.name])} onClose={() => setLog(false)} actions={<Btn onClick={() => setLog(false)}>{tr("Done")}</Btn>}><p>{a.description}</p><p className="small muted">{tr("Status: ")}{tr(state.label)} · {tr(elapsed(a.durationMs))} · {tr(compactTokens(a.tokens))}{tr(" tokens")}</p>{tr(a.steps?.length ? <ol className="subagent-log" aria-label={tr("{0} \u00B7 log", [a.name])}>{tr(a.steps.map((step, i) => <li key={i} className={step.said ? 'subagent-log__said' : 'subagent-log__step'}>{tr(step.said ? <span>{step.said}</span> : <><strong>{tr(step.verb ?? '')}</strong> <span>{step.target ?? ''}</span></>)}</li>))}</ol> : null)}</Dialog></div>;
}
