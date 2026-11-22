import { tr } from '../lib/i18n';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Btn, Dialog, type IconName } from './ui';
import { api } from '../lib/client';
import { useImeKeys } from '../lib/ime-keys';
import { plainFailureText } from '../lib/plain-errors';
import { compactTokens, elapsed } from '../lib/chat-metrics';
import type { Session, SessionAgentReply } from '../lib/types';
import { helperRanLong } from '../lib/helper-limit';
import { helperRecordsChanged, updateHelperRecords, type HelperAction, type HelperMessage, type HelperRecord } from '../lib/helper-records';

type Agent = NonNullable<Session['agents']>[number];

export interface HelperThreadItem {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  at?: string;
  delivery?: 'queue' | 'steer';
}

/** Merge user-sent messages and helper replies into chronological order for the work view (#6516). */
export function buildHelperThread(
  messages: HelperMessage[],
  replies?: SessionAgentReply[],
  steps?: Agent['steps'],
): HelperThreadItem[] {
  const userItems = messages.map((m, idx) => ({
    id: `u:${m.id}`,
    role: 'user' as const,
    text: m.text,
    at: m.at,
    delivery: m.delivery,
    ts: Date.parse(m.at) || 0,
    order: idx,
  }));
  const assistantSource = replies?.length
    ? replies.map((r, idx) => ({
      id: `a:${r.id}`,
      role: 'assistant' as const,
      text: r.text,
      at: r.at,
      ts: Date.parse(r.at) || 0,
      order: idx,
    }))
    : (steps ?? []).filter(s => typeof s.said === 'string' && s.said.trim()).map((s, idx) => ({
      id: `s:${idx}`,
      role: 'assistant' as const,
      text: s.said!,
      at: undefined,
      ts: Number.MAX_SAFE_INTEGER - 1000 + idx,
      order: idx,
    }));
  return [...userItems, ...assistantSource]
    .sort((a, b) => (a.ts !== b.ts ? a.ts - b.ts : a.role === b.role ? a.order - b.order : a.role === 'user' ? -1 : 1))
    .map(({ ts: _ts, order: _order, ...item }) => item);
}

const states: Record<Agent['status'], { label: string; icon: IconName }> = {
  working: { label: 'In progress', icon: 'clock' },
  completed: { label: 'Completed', icon: 'check' },
  failed: { label: 'Failed', icon: 'error' },
  queued: { label: 'Queued', icon: 'clock' },
  cancelled: { label: 'Cancelled', icon: 'x' },
};

export function SubagentRow({ agent: a, now = Date.now(), sessionId, record, onRemove, onOpenWork, panel = false }: {
  agent: Agent;
  now?: number;
  /** The chat the helper belongs to: with it the row can take a message for the helper (#4749). */
  sessionId?: string;
  record?: HelperRecord;
  onRemove?: () => void;
  /** Opens this helper's standalone work view surface (#6516). */
  onOpenWork?: () => void;
  panel?: boolean;
}) {
  const [log, setLog] = useState(false);
  const [writing, setWriting] = useState(false);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);
  const [sentMessages, setSentMessages] = useState<HelperMessage[]>([]);
  const [updating, setUpdating] = useState(false);
  const [problem, setProblem] = useState('');
  const [stopping, setStopping] = useState(false);
  const ime = useImeKeys();
  const logRef = useRef<HTMLOListElement>(null);
  const followLog = useRef(true);
  const seenLog = useRef<string | null>(null);

  // A working helper gets it at its next step (steer); one that has finished or waits gets a new turn with it (queue).
  async function send() {
    const message = draft.trim();
    if (!message || !sessionId || busy) return;
    setBusy(true); setProblem(''); setSent(false);
    try {
      const delivery = a.status === 'working' ? 'steer' : 'queue';
      const result = await api.post<{ message?: HelperMessage; historySaved?: boolean }>('/api/chat/helper', { sessionId, helperId: a.id, message, delivery });
      const item = result.message ?? { id: crypto.randomUUID(), text: message, at: new Date().toISOString(), delivery };
      setSentMessages(previous => [...previous, item]);
      setDraft(''); setSent(true);
      if (result.historySaved === false) setProblem(tr('The message was sent, but its history could not be saved.'));
      helperRecordsChanged(sessionId);
    } catch (e) {
      setProblem(plainFailureText(e instanceof Error ? e.message : String(e)).message);
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    if (!sessionId || stopping) return;
    setStopping(true); setProblem('');
    try {
      await api.post('/api/chat/helper/stop', { sessionId, helperId: a.id });
    } catch (e) {
      setProblem(plainFailureText(e instanceof Error ? e.message : String(e)).message);
      setStopping(false);
    }
  }

  async function openNativeWindow() {
    if (!sessionId) { showLog(); return; }
    try {
      const { opened } = await api.post<{ opened: boolean }>('/api/chat/helper/window', { sessionId, helperId: a.id, name: a.name });
      if (!opened) showLog();
    } catch (e) {
      setProblem(plainFailureText(e instanceof Error ? e.message : String(e)).message);
    }
  }

  async function changeArchive(action: HelperAction) {
    if (!sessionId || updating) return;
    setUpdating(true); setProblem('');
    try { await updateHelperRecords(sessionId, action, [a.id]); }
    catch (e) { setProblem(plainFailureText(e instanceof Error ? e.message : String(e)).message); }
    finally { setUpdating(false); }
  }

  const observedAt = useRef(Date.now());
  const duration = a.durationMs + (a.status === 'working' ? Math.max(0, now - (a.updatedAt ?? observedAt.current)) : 0);
  const state = states[a.status] ?? states.queued;
  const canStop = a.status === 'working' || a.status === 'queued';
  useEffect(() => { if (!canStop) setStopping(false); }, [canStop]);
  const archived = !canStop && record?.archived;
  const canMessage = !!sessionId && !archived && a.status !== 'cancelled';
  const savedMessages = record?.messages ?? [];
  const messages = [...savedMessages, ...sentMessages.filter(message => !savedMessages.some(saved => saved.id === message.id))];
  const thread = buildHelperThread(messages, a.replies, a.steps);
  const threadHistory = thread.length ? <section className="subagent-messages subagent-thread" aria-label={tr('Helper conversation')}>
    <h2>{tr('Conversation')}</h2>
    <ol>{thread.map(item => <li key={item.id} className={`subagent-thread__item subagent-thread__item--${item.role}`}>
      <div className="subagent-thread__meta">
        <strong>{item.role === 'user' ? tr('You') : a.name}</strong>
        {item.delivery && <span className="subagent-thread__badge">{tr(item.delivery === 'steer' ? 'Live steer' : 'Follow-up')}</span>}
        {item.at && <time dateTime={item.at}>{new Date(item.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</time>}
      </div>
      <div className="subagent-thread__text">{item.text}</div>
    </li>)}</ol>
  </section> : null;
  const replyForm = canMessage ? <form className="subagent-row__reply" onSubmit={e => { e.preventDefault(); void send(); }}>
    <textarea className="st-input" rows={2} value={draft} maxLength={8000}
      aria-label={tr('Message to {0}', [a.name])} placeholder={tr('Write to {0}…', [a.name])}
      disabled={busy || updating} onChange={e => { setDraft(e.target.value); setSent(false); }}
      {...ime.events}
      onKeyDown={e => { if (ime.owns(e)) return; if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send(); } }}/>
    <Btn size="sm" variant="primary" type="submit" disabled={busy || updating || !draft.trim()}>{tr('Send')}</Btn>
  </form> : null;
  // ChildProjector merges consecutive text chunks into one step. Its count can
  // stay unchanged while the visible log grows, so follow the rendered content,
  // not merely the number of list items.
  const logRevision = JSON.stringify(a.steps ?? []);
  function showLog() {
    followLog.current = true;
    seenLog.current = null;
    if (onOpenWork) onOpenWork();
    else setLog(true);
  }
  function onRowClick(e: React.MouseEvent<HTMLDivElement>) {
    if (panel) return;
    const target = e.target as Element | null;
    if (target?.closest('button, input, textarea, a, form, [role="dialog"]')) return;
    showLog();
  }
  // A helper window opens on its current work, not on an old first step. Keep
  // following new work only while the reader has stayed at the end, so looking
  // back through the log never snaps the scroll position away.
  useLayoutEffect(() => {
    const log = logRef.current;
    if (!log) return;
    if (seenLog.current === null || followLog.current)
      log.scrollTop = log.scrollHeight;
    seenLog.current = logRevision;
  }, [log, logRevision, panel]);
  const trackLogPosition = () => {
    const log = logRef.current;
    if (!log) return;
    followLog.current = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
  };
  const logEntries = a.steps?.length ? <ol ref={logRef} onScroll={trackLogPosition} className="subagent-log" aria-label={tr('{0} · log', [a.name])}>
    {a.steps.map((step, i) => <li key={i} className={step.said ? 'subagent-log__said' : 'subagent-log__step'}>{step.said
      ? <span>{step.said}</span>
      : <><strong>{tr(step.verb ?? '')}</strong> <span>{step.target ?? ''}</span></>}</li>)}
  </ol> : null;

  return <div className={`subagent-row${!panel ? ' subagent-row--clickable' : ''}`} data-status={a.status} data-panel={panel || undefined} onClick={onRowClick}>
    <span className="subagent-row__dot" aria-label={tr(state.label)}/>
    <div className="subagent-row__copy">
      <div className="subagent-row__title">
        <strong title={a.name}>{a.name}</strong>
        <span className="subagent-row__status">{tr(state.label)}{archived && <> · {tr('Archived')}</>}</span>
      </div>
      <p title={a.description}>{tr(a.description)}</p>
      <div className="subagent-row__actions">
        {!panel && (a.status === 'failed' || a.steps?.length || messages.length) ? <Btn size="sm" onClick={showLog}>{tr('View log')}</Btn> : null}
        {!panel && sessionId && <Btn size="sm" onClick={() => void openNativeWindow()}>{tr('Open window')}</Btn>}
        {sessionId && canStop && <Btn size="sm" variant="danger" disabled={stopping} onClick={() => void stop()}>{tr(stopping ? 'Stopping…' : 'Stop helper')}</Btn>}
        {canMessage && !panel && <Btn size="sm" aria-expanded={writing} onClick={() => setWriting(v => !v)}>{tr('Message this helper')}</Btn>}
        {sessionId && !canStop && <Btn size="sm" disabled={updating || busy}
          onClick={() => void changeArchive(archived ? 'restore' : 'archive')}>{tr(archived ? 'Restore' : 'Archive')}</Btn>}
        {onRemove && !canStop && <Btn size="sm" variant="ghost" disabled={updating || busy} onClick={onRemove}>{tr('Remove')}</Btn>}
      </div>
      {!panel && writing && replyForm}
      {tr(helperRanLong(a.status, duration) ? <p className="small subagent-row__note" role="status">{tr("This helper has been working for over 30 minutes.")}</p> : null)}
      {tr(sent ? <p className="small subagent-row__note" role="status">{tr('Sent to {0}.', [a.name])}</p> : null)}
      {tr(problem ? <p className="small inline-error subagent-row__note" role="alert">{problem}</p> : null)}
    </div>
    <div className="subagent-row__metrics"><span>{tr(compactTokens(a.tokens))}{tr(' tok')}</span><span>{tr(elapsed(duration))}</span></div>
    {panel ? <>{threadHistory}{logEntries}<div className="subagent-work__composer">{replyForm}</div></> : !onOpenWork && <Dialog open={log} title={tr('{0} · log', [a.name])} wide onClose={() => setLog(false)} actions={<>{sessionId && canStop && <Btn size="sm" variant="danger" disabled={stopping} onClick={() => void stop()}>{tr(stopping ? 'Stopping…' : 'Stop helper')}</Btn>}<Btn onClick={() => setLog(false)}>{tr('Done')}</Btn></>}>
      <p>{a.description}</p>
      <p className="small muted">{tr('Status: ')}{tr(state.label)} · {tr(elapsed(duration))} · {tr(compactTokens(a.tokens))}{tr(' tokens')}</p>
      {threadHistory}
      {logEntries}
      {replyForm}
      {tr(sent ? <p className="small subagent-row__note" role="status">{tr('Sent to {0}.', [a.name])}</p> : null)}
      {tr(problem ? <p className="small inline-error subagent-row__note" role="alert">{problem}</p> : null)}
    </Dialog>}
  </div>;
}
