/** The agent's own questions (#4244): what the host sends, and the answer that goes back. */
export interface AgentQuestion { id: string; question: string; detail?: string; header?: string; options: { label: string; description?: string }[]; multiSelect: boolean }
export interface AgentAsk { callId: string; questions: AgentQuestion[] }
export interface AgentAnswer { id: string; selected: string[]; custom?: string }

/** Reads the frame of the host defensively: a question that cannot be shown is left out. */
export function readAsk(frame: { callId?: unknown; questions?: unknown }): AgentAsk | null {
  const callId = typeof frame.callId === 'string' ? frame.callId : '';
  const questions = (Array.isArray(frame.questions) ? frame.questions : []).flatMap((q): AgentQuestion[] => {
    if (!q || typeof q !== 'object') return [];
    const o = q as Record<string, unknown>;
    if (typeof o.id !== 'string' || !o.id || typeof o.question !== 'string' || !o.question.trim()) return [];
    const options = (Array.isArray(o.options) ? o.options : []).flatMap((x): { label: string; description?: string }[] => {
      const p = x as Record<string, unknown> | null;
      return p && typeof p.label === 'string' && p.label ? [{ label: p.label, ...(typeof p.description === 'string' && p.description ? { description: p.description } : {}) }] : [];
    });
    return [{ id: o.id, question: o.question, ...(typeof o.detail === 'string' && o.detail ? { detail: o.detail } : {}), ...(typeof o.header === 'string' && o.header ? { header: o.header } : {}), options, multiSelect: o.multiSelect === true }];
  });
  return callId && questions.length ? { callId, questions } : null;
}

/** Whether every question has a choice or the person's own words, so the answer can be sent. */
export function answerComplete(questions: readonly AgentQuestion[], answers: Record<string, { selected: string[]; custom: string }>): boolean {
  return questions.every(q => { const a = answers[q.id]; return !!a && (a.selected.length > 0 || a.custom.trim().length > 0); });
}

/** The answer in the shape the host takes: the offered labels the person ticked, and their own words when they typed some. */
export function toAnswers(questions: readonly AgentQuestion[], answers: Record<string, { selected: string[]; custom: string }>): AgentAnswer[] {
  return questions.map(q => { const a = answers[q.id] ?? { selected: [], custom: '' }; const custom = a.custom.trim(); return { id: q.id, selected: a.selected, ...(custom ? { custom } : {}) }; });
}
