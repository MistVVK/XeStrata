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
 * The power the meter adds up (#6012): the sum over the GPUs that report one. A GPU with no power counter (an AMD iGPU
 * on Windows, a unified-memory GPU) adds nothing instead of making the whole sum unknown; with no reading at all, null.
 * PR #4430: the Monitor's total power and power limit are summed the same way (the `key` argument), so the tile shows what the meter counts.
 */
export function reportedPowerW(readings, key = 'powerW') {
  const values = (readings ?? []).map(g => g?.[key]).filter(v => typeof v === 'number' && Number.isFinite(v) && v >= 0);
  return values.length ? values.reduce((n, v) => n + v, 0) : null;
}
