import { randomUUID } from 'node:crypto';
import { parseArguments } from '../events.mjs';

export const name = 'strata-working-memory';
export const inject = ['jobs'];

const HISTORY_EVENTS = 2048;
const MAX_BRIEF_BYTES = 3000;
const FILE_TOOLS = new Set(['read', 'edit', 'write']);
const WORK_TOOLS = new Set([...FILE_TOOLS, 'bash', 'pwsh', 'job_output', 'job_kill']);

const clip = (text, limit) => typeof text === 'string' ? text.slice(0, limit).replace(/\s+/g, ' ').trim() : '';
const at = time => Number.isFinite(time) && Math.abs(time) <= 8.64e15 ? new Date(time).toISOString() : '';
const human = message => message?.source?.kind === 'user' && typeof message.source.rpcId === 'string';
const firstText = message => Array.isArray(message?.content) ? message.content.find(part => part?.type === 'text')?.text : '';

function pathKey(path) {
  const key = path.replace(/^@/u, '').replaceAll('\\', '/').replace(/^\.\//u, '');
  return process.platform === 'win32' ? key.toLowerCase() : key;
}

/** Match paths already observed in this session. A short name shared by
 * different observed paths does not identify evidence for either file. */
function referencedPaths(paths, texts) {
  const matched = new Set();
  for (const text of texts) {
    if (typeof text !== 'string') continue;
    for (const match of text.slice(0, 8000).matchAll(/(["'\x60])([^\r\n]+?)\1|[^\s"'\x60<>(){}[\],;!?]+/gu)) {
      const reference = pathKey(match[2] ?? match[0].replace(/[.:]+$/u, ''));
      if (!reference) continue;
      const candidates = paths.filter(path => path === reference || path.endsWith('/' + reference));
      if (candidates.length === 1) matched.add(candidates[0]);
    }
  }
  return matched;
}

/** A small view of the current session's durable events. Never carry raw command lines,
 * file contents, model reasoning, or a tool's stdout into the working brief. */
export function workingBrief(log, jobs = [], currentDirections = []) {
  if (!Array.isArray(log)) return '';
  const requests = [];
  const calls = new Map();
  const outcomes = new Map();
  let checklist;
  let ended;
  // A long tool run may fill the tail without a new human instruction or
  // checklist. Look those up in the full log without replaying tool payloads.
  for (let index = log.length - 1; index >= 0; index--) {
    const { type, data = {}, seq, time } = log[index];
    if (type === 'user/message' && requests.length < 2 && human(data)) {
      const text = clip(firstText(data), 360);
      if (text) requests.unshift({ text, at: at(time), seq });
    }
    if (type === 'todo/write' && !checklist) {
      checklist = { items: Array.isArray(data.todos) ? data.todos : [], seq, at: at(time) };
    }
    if (type === 'turn/end' && !ended) ended = { status: clip(data.reason?.kind, 32), at: at(time), seq };
    if (requests.length === 2 && checklist && ended) break;
  }
  const items = checklist?.items ?? [];
  const inProgress = items.filter(item => item?.status === 'in_progress');
  const openTasks = [...inProgress, ...items.filter(item => item?.status === 'pending')].slice(0, 4);
  for (const event of log.slice(-HISTORY_EVENTS)) {
    const data = event.data ?? {};
    switch (event.type) {
      case 'tool/call': {
        if (!WORK_TOOLS.has(data.name) || typeof data.callId !== 'string') break;
        const args = parseArguments(data.arguments);
        const target = FILE_TOOLS.has(data.name) ? args.file_path ?? args.path : args.job_id ?? args.description;
        calls.set(data.callId, { tool: data.name, target: clip(target, 200), seq: event.seq, at: at(event.time) });
        break;
      }
      case 'tool/result': {
        const callId = data.message?.toolCallId;
        const call = calls.get(callId);
        if (!call) break;
        calls.delete(callId);
        const isError = data.message.isError === true;
        const label = FILE_TOOLS.has(call.tool) ? call.tool + ':' + call.target : callId;
        outcomes.delete(label);
        outcomes.set(label, { ...call, result: isError ? 'error' : 'returned', seq: event.seq, at: at(event.time) });
        break;
      }
    }
  }

  if (outcomes.size === 0 && calls.size === 0 && openTasks.length === 0 && jobs.length === 0) return '';
  const recent = [...outcomes.values()].reverse();
  const fileOutcomes = recent.filter(outcome => FILE_TOOLS.has(outcome.tool) && outcome.target);
  const paths = [...new Set(fileOutcomes.map(outcome => pathKey(outcome.target)))];
  const requested = referencedPaths(paths, currentDirections);
  const focus = requested.size ? requested : referencedPaths(paths, inProgress.map(task => task.content));
  const focusedOutcomes = fileOutcomes.filter(outcome => focus.has(pathKey(outcome.target)));
  const rows = [
    'Strata working state from this chat (historical evidence). Use it if the current request continues this work. Follow the current user request first.',
    'Before changing a previously touched file, use read on its current contents. Use new file text to construct edits. Check whether an unfinished action already took effect before retrying it.',
  ];
  const add = (label, value) => {
    const row = '\n' + label + ' ' + JSON.stringify(value);
    if (Buffer.byteLength(rows.join('\n') + row, 'utf8') <= MAX_BRIEF_BYTES) rows.push(row.slice(1));
  };
  for (const request of requests) add('Prior user request', request);
  for (const item of openTasks) {
    add('Open task', { text: clip(item.content, 160), status: item.status, seq: checklist.seq, at: checklist.at });
  }
  for (const outcome of focusedOutcomes.slice(0, 6)) add('Earlier tool outcome', outcome);
  if (ended?.status) add('Last recorded turn', ended);
  for (const job of jobs.slice(-3)) {
    add('Active job in this session', { id: clip(job.id, 64), label: clip(job.label, 120), status: job.status, at: at(job.startedAt) });
  }
  for (const call of [...calls.values()].slice(-3)) add('Tool call without a recorded result', call);
  if (!focusedOutcomes.length) {
    for (const outcome of recent.slice(0, 6)) add('Earlier tool outcome', outcome);
  }

  return rows.join('\n');
}

export function apply(ctx) {
  ctx.on('agent/pre-step', async ({ agent, step, signal }, next) => {
    const decision = await next();
    // Only attach memory to a turn the user is already starting. A memory note
    // never wakes the agent or starts a new model request at the end of a turn.
    if (decision.kind !== 'enter' || signal.aborted || step !== 1) return decision;
    const directions = decision.messages.filter(human);
    if (!directions.length) return decision;
    const session = agent.session;
    // A fork or helper may inherit another session's log. Recalling it here
    // would promote the parent's work to evidence for the child's task.
    if (session.header?.parentSession || session.header?.origin === 'subagent') return decision;
    const jobs = ctx.jobs.list(session.id).filter(job => job.owner === session.id && ['running', 'stopping'].includes(job.status));
    const text = workingBrief(session.log, jobs, directions.map(firstText));
    if (!text) return decision;
    const note = {
      id: randomUUID(),
      role: 'user',
      source: { kind: name, form: 'recall' },
      content: [{ type: 'text', text }],
    };
    return { ...decision, messages: [note, ...decision.messages] };
  });
}
