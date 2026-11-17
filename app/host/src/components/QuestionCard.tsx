/** A question the agent asked the person, in the style of the approval card: options to pick, the person's own words, one answer or a skip (#4244). */
import { useState } from 'react';
import { tr } from '../lib/i18n';
import { Btn, Icon } from './ui';
import { answerAgentAsk } from '../lib/store';
import { plainFailureText } from '../lib/plain-errors';
import { answerComplete, toAnswers, type AgentAsk, type AgentQuestion } from '../lib/agent-ask';

type Draft = Record<string, { selected: string[]; custom: string }>;

export function QuestionCard({ sessionId, ask }: { sessionId: string; ask: AgentAsk }) {
    const [draft, setDraft] = useState<Draft>({});
    const [busy, setBusy] = useState(false), [problem, setProblem] = useState('');
    const current = (q: AgentQuestion) => draft[q.id] ?? { selected: [], custom: '' };
    const pick = (q: AgentQuestion, label: string) => setDraft(d => {
        const cur = d[q.id] ?? { selected: [], custom: '' };
        const selected = q.multiSelect ? (cur.selected.includes(label) ? cur.selected.filter(x => x !== label) : [...cur.selected, label]) : [label];
        // One choice for a single-select question: picking an option replaces words typed instead.
        return { ...d, [q.id]: { selected, custom: q.multiSelect ? cur.custom : '' } };
    });
    const type = (q: AgentQuestion, custom: string) => setDraft(d => {
        const cur = d[q.id] ?? { selected: [], custom: '' };
        return { ...d, [q.id]: { selected: q.multiSelect || !custom.trim() ? cur.selected : [], custom } };
    });
    async function send(skip: boolean) {
        if (busy) return;
        setBusy(true); setProblem('');
        try { await answerAgentAsk(sessionId, ask.callId, skip ? null : toAnswers(ask.questions, draft)); }
        catch (e) { setProblem(plainFailureText(e instanceof Error ? e.message : String(e)).message); setBusy(false); }
    }
    return <section className="askcard askq" role="group" aria-label={tr("The assistant has a question for you")}>
      {ask.questions.map(q => {
        const cur = current(q);
        return <div className="askq__item" key={q.id}>
          <div className="askcard__top"><span className="askcard__icon"><Icon name="chat" size={18}/></span>
            <div className="askcard__text">{q.header ? <p className="xs muted">{q.header}</p> : null}<p className="askcard__q">{q.question}</p>{q.detail ? <p className="small muted askq__detail">{q.detail}</p> : null}</div></div>
          {q.options.length ? <div className="askq__opts" role={q.multiSelect ? 'group' : 'radiogroup'} aria-label={q.question}>{q.options.map(o => {
            const on = cur.selected.includes(o.label);
            return <button type="button" key={o.label} role={q.multiSelect ? 'checkbox' : 'radio'} aria-checked={on} className="askq__opt" data-on={on || undefined} disabled={busy} onClick={() => pick(q, o.label)}><span>{o.label}</span>{o.description ? <small className="muted">{o.description}</small> : null}</button>;
          })}</div> : null}
          <input className="st-input askq__own" aria-label={tr("Type your own answer")} placeholder={tr("Type your own answer")} value={cur.custom} maxLength={2000} disabled={busy} onChange={e => type(q, e.target.value)}/>
        </div>;
      })}
      {problem ? <p className="small inline-error askq__problem" role="alert">{problem}</p> : null}
      <div className="askq__acts"><Btn size="sm" disabled={busy} onClick={() => void send(true)}>{tr("Skip")}</Btn><Btn variant="primary" size="sm" disabled={busy || !answerComplete(ask.questions, draft)} onClick={() => void send(false)}>{tr("Send answer")}</Btn></div>
    </section>;
}
