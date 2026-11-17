/**
 * The agent's own questions (the Harness's `ask_user_question` tool, #4244). The Harness raises the `user-questions/request` event with
 * `{ questions: [{ id, question, detail?, header?, options?: [{ label, description? }], multiSelect?, intent? }], wait?: { callId } }` and wants
 * `{ answers: [{ id, selected: string[], custom? }] }` back (AskUserQuestionRequestEvent / AskUserQuestionAnswer of the pinned Harness).
 * This file only reads the first into something the page can draw, and checks the second before it goes out.
 */
const text = (v, max) => (typeof v === 'string' ? v.trim().slice(0, max) : '');

/** The questions of a request, bounded and plain; an empty list when the request carries nothing that can be shown. */
export function parseQuestions(raw) {
  const out = [];
  for (const q of Array.isArray(raw) ? raw.slice(0, 6) : []) {
    const id = text(q?.id, 120), question = text(q?.question, 800);
    if (!id || !question) continue;
    const options = (Array.isArray(q.options) ? q.options.slice(0, 12) : [])
      .map(o => ({ label: text(o?.label, 200), description: text(o?.description, 400) || undefined })).filter(o => o.label);
    out.push({ id, question, detail: text(q.detail, 800) || undefined, header: text(q.header, 80) || undefined, options, multiSelect: q.multiSelect === true });
  }
  return out;
}

/**
 * The answer the person gave, checked against the questions that were asked: every question answered, only the offered labels (or the person's own
 * words), one choice for a single-select question. Throws a plain Error with a code on the first thing that does not fit.
 */
export function buildAnswer(questions, given) {
  const byId = new Map((Array.isArray(given) ? given : []).map(a => [String(a?.id ?? ''), a]));
  const answers = [];
  for (const q of questions) {
    const a = byId.get(q.id);
    const labels = new Set(q.options.map(o => o.label));
    const selected = (Array.isArray(a?.selected) ? a.selected : []).map(s => String(s)).filter(s => labels.has(s));
    const custom = text(a?.custom, 2000);
    if (!selected.length && !custom) throw Object.assign(new Error('Answer every question, or skip them.'), { code: 'question-incomplete' });
    if (!q.multiSelect && selected.length > 1) throw Object.assign(new Error('Choose one answer for each question.'), { code: 'question-invalid' });
    answers.push({ id: q.id, selected: [...new Set(selected)], ...(custom ? { custom } : {}) });
  }
  return { answers };
}
