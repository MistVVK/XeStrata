/**
 * #5996: the helper time limit. The person may stop any helper that keeps working past 30 minutes, 1 hour or 2 hours (Settings > General); Off, the
 * default, lets helpers run as long as they need. The host counts and stops (server/agent/children.mjs helperLimitMin, the same three values);
 * whatever the setting, a helper's row says so once it has worked more than 30 minutes.
 */
export const HELPER_LIMITS = [30, 60, 120] as const;
export type HelperLimit = typeof HELPER_LIMITS[number];
export const validHelperLimit = (value: unknown): HelperLimit | null => HELPER_LIMITS.find(m => m === value) ?? null;
/** A working helper's row carries a plain note after this long, whatever the limit. */
export const HELPER_NOTE_MS = 30 * 60_000;
export const helperRanLong = (status: string, durationMs: number) => status === 'working' && durationMs > HELPER_NOTE_MS;
