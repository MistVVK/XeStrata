/**
 * How often the host asks the graphics card driver for its readings (#4066).
 * It used to start two nvidia-smi processes every 3 s for as long as the app was open, even with nothing loaded and
 * the Monitor out of sight; on some PCs that was felt as a stutter while typing. Now the 3 s pace is kept while a model
 * runs or the Monitor is being looked at, and an idle app asks once every 30 s.
 */
export const ACTIVE_MS = 3000;
export const IDLE_MS = 30000;
/** The Monitor screen asks again every 10 s while it is open; this is how long one ask keeps the fast pace. */
export const WATCH_MS = 25000;

/** `live`: a model is loaded or an engine runs. `watchedUntil`: ms timestamp until which the Monitor was last seen open. */
export function sensorPace({ live = false, watchedUntil = 0, now = Date.now() } = {}) {
  const watched = now < watchedUntil;
  return { intervalMs: live || watched ? ACTIVE_MS : IDLE_MS, transfer: !!live };
}
