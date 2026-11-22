import { parseQuestions } from '../user-questions.mjs';

export const name = 'strata-questions';
export const inject = ['agents', 'tools', 'userQuestions'];

export function apply(ctx) {
  ctx.on('tools/execute', async (exec, next) => {
    const child = exec.agent;
    const header = child?.session.header;
    if (exec.name !== 'ask_user_question' || header?.origin !== 'subagent' || !header.parentSession) return next();
    const parent = ctx.agents.get(header.parentSession);
    // The Harness lets only a live root ask the person. Relay through the exact
    // live owner, keeping the question tied to the child's cancellable tool call.
    if (!parent || !ctx.agents.roots().includes(parent) || !ctx.agents.isOwnedBy(child.id, parent)) return next();
    const raw = exec.arguments?.questions;
    if (!Array.isArray(raw)) return next();
    const questions = parseQuestions(raw.map(q => ({ ...q, multiSelect: q?.multi_select === true })));
    if (!questions.length) return next();
    // Strip unset optional fields before the gateway checks for lossless JSON.
    const wireQuestions = JSON.parse(JSON.stringify(questions));
    const value = await ctx.userQuestions.ask({ agent: parent, questions: wireQuestions, signal: exec.signal, strataHelperId: child.id });
    return { isError: false, value, content: [{ type: 'text', text: JSON.stringify(value) }] };
  });
}
