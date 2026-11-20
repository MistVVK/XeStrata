/**
 * Tencent Hy4-Flash runs on its own engine program (strata-hy4, the component "strata-hy4"), which takes none of the Flash-Next options the app writes for the Strata engine
 * (--vram-reserve-mib, --pool-workers, --resident-budget-gib, --batch, --mtp, --spec ...) and refuses a flag it does not know. This is the one test the app's settings code asks;
 * the run config says it with its arch (terminal/serve/hy4_kind.py is the Python side of the same test).
 */
export const HY4_ARCH = 'hy4flash';
export const HY4_FAMILY = 'hy4-flash';
export const isHy4 = config => config?.arch === HY4_ARCH || /(^|[\\/:])strata-hy4(\.exe)?$/i.test(String(config?.exe ?? ''));
export const isHy4Family = family => family === HY4_FAMILY;
/** Return the `--vision <path>` shard from a Hy4-Flash run config's args, or null if absent. */
export function hy4VisionShard(config) {
  if (!isHy4(config) || !Array.isArray(config?.args)) return null;
  const at = config.args.indexOf('--vision');
  if (at < 0 || at + 1 >= config.args.length) return null;
  const val = config.args[at + 1];
  return typeof val === 'string' && val.length > 0 && !val.startsWith('--') ? val : null;
}
/** Strip `--vision` and its following non-flag `.strata` shard path from a Hy4-Flash args list. */
export function withoutHy4Vision(args) {
  const list = (Array.isArray(args) ? args : []).map(String), out = [];
  for (let i = 0; i < list.length; i++) {
    if (list[i] === '--vision') {
      if (i + 1 < list.length && !list[i + 1].startsWith('--')) i++;
      continue;
    }
    out.push(list[i]);
  }
  return out;
}
