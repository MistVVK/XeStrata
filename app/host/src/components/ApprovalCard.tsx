/** The card that asks the person to approve or deny one action of the assistant, and the dock that keeps it in view. */
import { useMemo } from 'react';
import { tr } from '../lib/i18n';
import { Btn, Icon } from './ui';
import { askQuestion, askSub, askIcon } from '../lib/ask';
import { resolveTool, useStore } from '../lib/store';
import { approvalDetail, pendingApprovals } from '../lib/approvals';
import { QuestionCard } from './QuestionCard';
import type { AgentAsk } from '../lib/agent-ask';
import type { ToolCall } from '../lib/types';

const NO_ASKS: AgentAsk[] = [];
export const SCOPE: Record<ToolCall['scope'], {
    label: string;
    tone: 'plain' | 'info' | 'warn';
}> = {
    read: { label: 'Reads', tone: 'plain' },
    write: { label: 'Writes', tone: 'warn' },
    shell: { label: 'Runs a command', tone: 'warn' },
    connected: { label: 'Connected tool', tone: 'info' },
};

/** The one place the Approve and Deny buttons live. Chat shows it in the dock above the box, not in the middle of a long reply (#4725). */
export function ApprovalCard({ messageId, tool: t }: {
    messageId: string;
    tool: ToolCall;
}) {
    const scope = SCOPE[t.scope];
    if (t.ask)
        return <section className="askcard" data-state={t.state} aria-label={tr(askQuestion(t.ask))}>
      <div className="askcard__top"><span className="askcard__icon"><Icon name={askIcon(t.ask)} size={18}/></span>
        <div className="askcard__text"><p className="askcard__q">{askQuestion(t.ask)}</p>{askSub(t.ask) ? <p className="askcard__why" title={askSub(t.ask)}>{askSub(t.ask)}</p> : null}</div>
        <div className="askcard__acts"><Btn size="sm" onClick={() => resolveTool(messageId, t.id, false)}>{tr("Deny")}</Btn><Btn variant="primary" size="sm" onClick={() => resolveTool(messageId, t.id, true)}>{tr("Approve")}</Btn></div></div>
      <details className="askcard__details"><summary><Icon name="chevron" size={11}/>{tr("Details")}</summary><pre tabIndex={0}>{approvalDetail(t)}</pre></details>
    </section>;
    return <section className="toolcard toolcard--compact" data-state={t.state} aria-label={t.name}>
    <details className="toolcard__disclosure">
      <summary><Icon name={t.scope === 'shell' ? 'bolt' : 'tool'} size={17}/><span className="toolcard__title">{tr(t.verb ?? t.name)}</span><span className="toolcard__scope xs muted">{tr(scope.label)}</span><Icon name="chevron" className="toolcard__chevron" size={15}/></summary>
      <div className="toolcard__body"><code className="toolcard__preview">{t.preview}</code><p className="small muted">{tr(t.effect)}</p>{t.result ? <p className="small">{t.state === 'error' ? tr(t.result) : t.result}</p> : null}</div>
    </details>
    {t.target ? <code className="toolcard__what" title={t.target}>{t.target}</code> : null}
    <div className="toolcard__quick-actions"><span className="xs muted">{tr("Needs your approval")}</span><Btn size="sm" onClick={() => resolveTool(messageId, t.id, false)}>{tr("Deny")}</Btn><Btn variant="primary" size="sm" icon="check" onClick={() => resolveTool(messageId, t.id, true)}>{tr("Approve")}</Btn></div>
  </section>;
}

/**
 * Keeps an approval where the person is looking. A reply can run to many screens; the card used to sit above its text, so a
 * person reading the end of it never saw that the assistant had stopped to ask. The dock sits directly above the message box and
 * does not scroll with the conversation. Where the card was, the conversation says that it is waiting.
 */
export function ApprovalDock({ sessionId }: {
    sessionId: string | undefined;
}) {
    const session = useStore(s => s.sessions.find(x => x.id === sessionId));
    const items = useMemo(() => pendingApprovals(session), [session]);
    const asks = useStore(s => (sessionId ? s.agentAsks[sessionId] : undefined)) ?? NO_ASKS;
    if (!items.length && !asks.length)
        return null;
    return <section className="approvaldock" role="group" aria-label={tr("Waiting for your answer")} data-count={items.length}>
      {asks.map(a => <QuestionCard key={a.callId} sessionId={sessionId!} ask={a}/>)}
      {items.map(({ messageId, tool }) => <ApprovalCard key={tool.id} messageId={messageId} tool={tool}/>)}
    </section>;
}
