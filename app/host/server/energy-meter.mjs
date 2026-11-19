/**
 * GPU energy since the host started, in Wh (#3628): the power reading held for the time since the last tick, added up.
 * A long gap (the PC slept, the host was paused) counts for at most 5 s, so a sleep never adds a made-up hour of use.
 */
const meter = { wh: 0, at: 0 };

export function addEnergy(powerW, at) {
  const dt = meter.at ? Math.min((at - meter.at) / 1000, 5) : 0;
  meter.at = at;
  if (typeof powerW === 'number' && Number.isFinite(powerW) && powerW >= 0 && dt > 0) meter.wh += powerW * dt / 3600;
  return meter.wh;
}

export const energyWh = () => meter.wh;

/**
 * The power a running engine reports for its own cards (#3727): an AMD card on Windows has no power reading of the host's own, so the meter stayed at 0 Wh
 * while the Monitor's Power tile (the engine's reading) showed 235 to 285 W. The host's poll of an engine's metrics notes the watts here, per instance;
 * a reading older than `ENGINE_POWER_MAX_AGE_MS` (an engine that stopped answering) counts for nothing.
 */
export const ENGINE_POWER_MAX_AGE_MS = 6000;
const enginePower = new Map();

/** Note what one engine instance reports (a number of watts, or null when it reports none). */
export function noteEnginePower(instanceId, powerW, at = Date.now()) {
  if (typeof powerW === 'number' && Number.isFinite(powerW) && powerW >= 0) enginePower.set(instanceId, { powerW, at });
  else enginePower.delete(instanceId);
}

/** The sum of the engines' fresh readings, or null when no engine reports one. */
export function enginePowerW(at = Date.now()) {
  let sum = null;
  for (const [id, r] of enginePower) {
    if (at - r.at > ENGINE_POWER_MAX_AGE_MS) { enginePower.delete(id); continue; }
    sum = (sum ?? 0) + r.powerW;
  }
  return sum;
}

/**
 * What the meter adds: the host's own sum over the cards that report a power, or the engines' when that is more (the host has no reading of its own for
 * some cards; the same card in both is not added twice).
 */
export function meteredPowerW(hostW, at = Date.now()) {
  const host = typeof hostW === 'number' && Number.isFinite(hostW) && hostW >= 0 ? hostW : null;
  const engines = enginePowerW(at);
  return host === null ? engines : engines === null ? host : Math.max(host, engines);
}

/** The watts in an engine's /metrics answer: the sum of `hardware.gpus[].power`, else the top-level `hardware.gpu_power`; null when it has none. */
export function metricsPowerW(metrics) {
  const hardware = metrics?.hardware;
  const values = (Array.isArray(hardware?.gpus) ? hardware.gpus : []).map(g => g?.power).filter(v => typeof v === 'number' && Number.isFinite(v) && v >= 0);
  if (values.length) return values.reduce((n, v) => n + v, 0);
  const total = hardware?.gpu_power;
  return typeof total === 'number' && Number.isFinite(total) && total >= 0 ? total : null;
}

/**
 * The power the meter adds up (#6012): the sum over the GPUs that report one. A GPU with no power counter (an AMD iGPU
 * on Windows, a unified-memory GPU) adds nothing instead of making the whole sum unknown; with no reading at all, null.
 * PR #4430: the Monitor's total power and power limit are summed the same way (the `key` argument), so the tile shows what the meter counts.
 */
export function reportedPowerW(readings, key = 'powerW') {
  const values = (readings ?? []).map(g => g?.[key]).filter(v => typeof v === 'number' && Number.isFinite(v) && v >= 0);
  return values.length ? values.reduce((n, v) => n + v, 0) : null;
}
