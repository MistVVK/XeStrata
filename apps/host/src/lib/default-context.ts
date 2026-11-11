import { BASE_CONTEXT, cacheGb, DENSE_GB, MTP_VRAM_GB, ownPathFits, type ContextFit } from './native-fit';

/**
 * The conversation context a model opens with: 128K for every configuration, unless the person picks another (owner, 11-01). Where the
 * model's own largest choice is under 128K, that largest choice. The engine's own recommendation (catalog.recommendation.context, which the
 * Terminal still uses) is not what the app opens with.
 */
export const DEFAULT_CONTEXT = 131072;
/** What the Models page used to open with: a saved model that holds exactly this has never had a context chosen for it. */
export const OLD_DEFAULT_CONTEXT = 32768;

/** The default among the sizes a model offers: 128K, or the largest size under it. */
export function defaultContext(choices: readonly number[] | null | undefined): number {
  const sizes = (choices ?? []).filter(n => Number.isFinite(n) && n > 0);
  const under = sizes.filter(n => n <= DEFAULT_CONTEXT);
  return under.length ? Math.max(...under) : sizes.length ? Math.min(...sizes) : DEFAULT_CONTEXT;
}

/**
 * What a model that is already on this computer opens with: the context it was prepared with when someone chose it, else the default. A choice made
 * from this update on is saved with a marker (`chosen`), whatever its size, 32K included; only an old saved 32K without a marker (the old default) moves
 * to the new default.
 */
export function openingContext(saved: number | null | undefined, choices: readonly number[] | null | undefined, chosen = false, oldDefaults: readonly number[] = [OLD_DEFAULT_CONTEXT]): number {
  return saved && saved > 0 && (chosen || !oldDefaults.includes(saved)) ? saved : defaultContext(choices);
}
/** What DeepSeek-V4.1-Flash used to open with (the one context its engine was checked at): an unmarked saved 8K is that default, not a choice. */
export const OLD_DEEPSEEK_CONTEXT = 8192;

/**
 * Graphics memory left for the model's own data after the conversation memory of this context (the app's estimate, the same figures as the This
 * computer card: dense weights and buffers, the conversation cache, the draft layer). Under ONE_GB the card is tight at this context.
 */
export function roomAfterContext(vramGb: number, context: number, kv = 'int8'): number {
  const dense = Math.max(DENSE_GB - cacheGb(BASE_CONTEXT), 0) + cacheGb(context, kv);
  return vramGb - dense - MTP_VRAM_GB;
}
/** A context past 32K that leaves a card under 1 GB for the model (the 32K default would leave more); for Qwen3.6 (`fit`, from the catalog), a context
 *  that does not fit the card by its own rule (#5715, strata_control/own_path_fit.py). Said as a warning; nothing is refused or lowered. */
export function contextIsTight(vramGb: number | null | undefined, context: number, kv = 'int8', fit?: ContextFit | null): boolean {
  const vram = Number(vramGb);
  if (!Number.isFinite(vram) || vram <= 0) return false;
  if (fit) return !ownPathFits(fit, vram, context);
  return context > BASE_CONTEXT && roomAfterContext(vram, context, kv) < 1;
}
