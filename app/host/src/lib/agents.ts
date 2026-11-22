import type { SessionAgent } from './types';

const statuses = new Set(['working', 'completed', 'failed', 'queued', 'cancelled']);
const metric = (value: unknown, fallback = 0) => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : fallback;

/** The steps of a helper as the log shows them: only well-formed ones, at most 20, each short. A frame without steps keeps the ones the row had. */
export const MAX_STEPS = 20;
export const MAX_REPLIES = 50;
const short = (value: unknown, max: number) => typeof value === 'string' && value.trim() ? value.slice(0, max) : undefined;
export function stepsOf(value: unknown): { steps?: NonNullable<SessionAgent['steps']> } {
  if (!Array.isArray(value)) return {};
  const steps = value.flatMap((item): NonNullable<SessionAgent['steps']> => {
    if (!item || typeof item !== 'object') return [];
    const o = item as Record<string, unknown>, said = short(o.said, 400), verb = short(o.verb, 40), target = short(o.target, 160);
    return said ? [{ said }] : verb || target ? [{ ...(verb ? { verb } : {}), ...(target ? { target } : {}) }] : [];
  }).slice(-MAX_STEPS);
  return steps.length ? { steps } : {};
}

export function repliesOf(value: unknown): { replies?: NonNullable<SessionAgent['replies']> } {
  if (!Array.isArray(value)) return {};
  const replies = value.flatMap((item): NonNullable<SessionAgent['replies']> => {
    if (!item || typeof item !== 'object') return [];
    const o = item as Record<string, unknown>;
    const id = short(o.id, 180), text = typeof o.text === 'string' && o.text.trim() ? o.text.slice(0, 4000) : undefined, at = short(o.at, 64);
    return id && text && at ? [{ id, text, at }] : [];
  }).slice(-MAX_REPLIES);
  return replies.length ? { replies } : {};
}

/** Merge partial agent frames without replacing other agents or their readings. */
export function updateAgent(agents: SessionAgent[], frame: unknown, now = Date.now()): SessionAgent[] {
  if (!frame || typeof frame !== 'object') return agents;
  const next = frame as Record<string, unknown>;
  if (typeof next.id !== 'string' || !next.id.trim()) return agents;
  const previous = agents.find(a => a.id === next.id);
  const at = metric(next.updatedAt, now);
  if (previous?.updatedAt && at < previous.updatedAt) return agents;
  const previousDuration = (previous?.durationMs ?? 0) + (previous?.status === 'working' && previous.updatedAt ? Math.max(0, at - previous.updatedAt) : 0);
  const agent: SessionAgent = {
    id: next.id,
    name: typeof next.name === 'string' ? next.name : previous?.name ?? 'Subagent',
    description: typeof next.description === 'string' ? next.description : previous?.description ?? '',
    status: statuses.has(String(next.status)) ? next.status as SessionAgent['status'] : previous?.status ?? 'queued',
    durationMs: metric(next.durationMs, previousDuration),
    tokens: metric(next.tokens, previous?.tokens),
    updatedAt: at,
    ...stepsOf(next.steps ?? previous?.steps),
    ...repliesOf(next.replies ?? previous?.replies),
  };
  return previous ? agents.map(a => a.id === agent.id ? agent : a) : [...agents, agent];
}

export const runningAgents = (agents: SessionAgent[]) => agents.filter(a => a.status === 'working').length;

/** Working vs not-working counts and working helper names for the toolbar button and hover summary (#6516). */
export function agentCounts(agents: SessionAgent[]) {
  const workingList = agents.filter(a => a.status === 'working');
  const working = workingList.length;
  const notWorking = agents.length - working;
  return {
    working,
    notWorking,
    total: agents.length,
    workingList,
  };
}

/** A page that was closed or reloaded holds no live helper: a saved row still "working" is shown as stopped until a newer frame says otherwise
 *  (updateAgent takes any frame newer than the row's own time, so a helper that really runs comes back by itself). #4218. */
export const settleAgents = (agents: SessionAgent[] | undefined): SessionAgent[] | undefined =>
  agents?.some(a => a.status === 'working' || a.status === 'queued')
    ? agents.map(a => a.status === 'working' || a.status === 'queued'
      ? { ...a, status: 'cancelled' as const, description: 'This helper is no longer running.' }
      : a)
    : agents;
