/**
 * The engine's Windows commit-limit warning (#1607) as the model card shows it. On Windows the engine compares what a run plans to allocate on the host with the
 * commit that is left (RAM + page file) and warns when it will not fit: it can end at the limit with no message while RAM is still free. The server reads the
 * engine's log line and reports the numbers in /metrics `engine.commit_warning` (serve/commit_limit.py); the card says it in plain words with one next step.
 */
export interface CommitWarning { left: string; limit: string }

const gb = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null);

/** The warning of one running model's /metrics `engine` object, with its numbers as the card shows them (one decimal), or null when the engine did not warn. */
export function commitWarning(engine: unknown): CommitWarning | null {
  const w = (engine && typeof engine === 'object' ? (engine as Record<string, unknown>).commit_warning : null) as Record<string, unknown> | null | undefined;
  if (!w || typeof w !== 'object') return null;
  const left = gb(w.left_gib), limit = gb(w.limit_gib);
  if (left === null || limit === null || limit <= 0) return null;
  return { left: left.toFixed(1), limit: limit.toFixed(1) };
}
