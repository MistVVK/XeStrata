/**
 * #5091: how much room a chat has before the assistant summarises its older messages by itself. The point comes from the host (GET /api/agent/compaction,
 * `model`: the window and the token count where summarising starts for the model that answers now, from the same arithmetic the agent uses), so there is
 * no second percentage in the page. It is shown only when that model's window is the window this chat runs with.
 */
export interface CompactionModel { context?: number; thresholdTokens?: number }

export function roomBeforeSummary(used: number | null | undefined, max: number | null | undefined, model: CompactionModel | null | undefined): number | null {
  if (typeof used !== 'number' || !Number.isFinite(used) || typeof max !== 'number' || !(max > 0)) return null;
  const point = model?.thresholdTokens, window = model?.context;
  if (typeof point !== 'number' || !(point > 0) || window !== max) return null;
  return Math.max(0, Math.round(point - used));
}
