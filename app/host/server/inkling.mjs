/**
 * Thinking Machines Inkling-2-Mini runs on its own shape engine program (strata-inkling, component "strata-inkling"),
 * which uses its own flags (--native, --vision, --max-context, --parallel 1..64, --draft dspark|off, --vram-experts-gb,
 * --ram-experts-gb) rather than Qwen Flash-Next flags.
 */
export const INKLING_ARCH = 'inkling2_loop';
export const INKLING_FAMILY = 'inkling-2-mini';
export const isInkling = config => config?.arch === INKLING_ARCH || config?.exe === 'component:strata-inkling' || /(^|[\\/:])strata-inkling(\.exe)?$/i.test(String(config?.exe ?? ''));
export const isInklingFamily = family => family === INKLING_FAMILY;
/** Return the `--vision <path>` shard from an Inkling-2-Mini run config's args, or null if absent. */
export function inklingVisionShard(config) {
  if (!isInkling(config) || !Array.isArray(config?.args)) return null;
  const at = config.args.indexOf('--vision');
  if (at < 0 || at + 1 >= config.args.length) return null;
  const val = config.args[at + 1];
  return typeof val === 'string' && val.length > 0 && !val.startsWith('--') ? val : null;
}
/** Strip `--vision` and its following non-flag `.strata` shard path from an Inkling-2-Mini args list. */
export function withoutInklingVision(args) {
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
