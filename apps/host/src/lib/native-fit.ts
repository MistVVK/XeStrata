/**
 * What a model version asks of this computer, from the catalog's own figures and the hardware report.
 *
 * The numbers are RAM + VRAM together WITHOUT the n-gram table (it stays on the SSD and is read when needed). The experts are held
 * where there is room: first in the graphics memory beyond about 5 GB (the dense weights, buffers and a 32K cache), then in RAM beyond
 * about 10 GB left for the system; whatever is left is read from the SSD. The same rules setup uses to choose its low-RAM mode
 * (setup.py: low_ram_gpu_gb, LOW_RAM_HEADROOM_GB), so the page and the engine agree.
 */
export interface FitInput { ram_needed_gb?: number; experts_gb?: number | null; memory_gb?: number; mem_total?: boolean; /** the weight files without the lookup table, when the catalog states them (DeepSeek: `memory_gb` is the CPU's share) */ fit_gb?: number | null; /** Qwen3.6: its own context rule (below) */ context_fit?: ContextFit | null }
export interface FitHardware { ram_gb?: number | null; vram_gb?: number | null }
export interface Fit {
  /** RAM + VRAM the version needs (the n-gram table not counted). */
  memoryGb: number;
  /** About what it takes of the graphics memory / of RAM on this PC. */
  vramUseGb: number;
  ramUseGb: number;
  /** About how much is read from the SSD while answering; 0 when it all fits in memory. */
  spillGb: number;
  known: boolean;
}

const DENSE_GB = 5, SYSTEM_ROOM_GB = 10, DENSE_EXTRA_GB = 1;
export { DENSE_GB };
/** The draft layer (MTP) takes about 1.08 GB of graphics memory (1082 MiB: the layer 840 + draft head 213 + the rest); it comes out of the expert cache while MTP is on. DFlash2 alone takes 0.89 GB (890 MiB, the MTP layer not loaded beside it): the fit uses the drafter that is chosen
 *  (phaseA-tests/blackwell/RESULTS-DF2MEM.md section 1; strata_control/drafting.py DRAFTER_GB). */
export const MTP_VRAM_GB = 1.08;
export const DFLASH2_VRAM_GB = 0.89;
/** DFlash2 alone with the converted pack (the selector's codebooks as Q4_0 rows: 715 MiB; strata_control/drafting.py DRAFTER_MIB "dflash2_selq4"). The page passes `dflash2_selq4` when `capabilities.dflash2Selq4` is true. */
export const DFLASH2_SELQ4_VRAM_GB = 0.72;
export const DRAFTER_VRAM_GB: Record<string, number> = { mtp: MTP_VRAM_GB, dflash2: DFLASH2_VRAM_GB, dflash2_selq4: DFLASH2_SELQ4_VRAM_GB, off: 0 };
const round1 = (n: number) => Math.round(n * 10) / 10;

export function memoryNeeded(v: FitInput): number {
  if (typeof v.memory_gb === 'number' && Number.isFinite(v.memory_gb)) return v.memory_gb;
  if (v.mem_total && typeof v.ram_needed_gb === 'number') return v.ram_needed_gb;
  return round1((v.experts_gb ?? v.ram_needed_gb ?? 0) + DENSE_EXTRA_GB);
}

/** The conversation cache of Flash-Next: 13 attention layers, bytes per token per layer by cache format, 5.4% over the bare cache
 *  (strata_control/vram_fit.py KV_BYTES, KV_MEASURED_FACTOR; setup.py UD_FIT_KV_BYTES). */
const KV_BYTES: Record<string, number> = { fp16: 2112, int8: 1056, q4_0: 576, k8v4: 816 };
const KV_LAYERS = 13, KV_FACTOR = 1.054;
export const BASE_CONTEXT = 32768;
export function cacheGb(context: number, kv = 'int8'): number {
  return (121 + context * KV_LAYERS * (KV_BYTES[String(kv).toLowerCase()] ?? KV_BYTES.int8) * KV_FACTOR / 2 ** 20) / 1024;   // 121 MiB fixed per session
}
/** Qwen3.6-35B-A3B (its own engine path, #5715): the catalog's figures from strata_control/own_path_fit.py, the rule setup's context menu uses. Its cache
 *  is fp32 for 11 layers, about 2.6x the Flash-Next int8 figure above, and fits a card when the fixed part plus the cache is at most the card's memory.
 *  Pinned to the Python by tests/own-path-fit.test.mjs. */
export interface ContextFit { bytes_per_token: number; fixed_gb: number }
export function ownPathCacheGb(fit: ContextFit, context: number): number { return context * fit.bytes_per_token / 2 ** 30; }
export function ownPathFits(fit: ContextFit, vramGb: number, context: number): boolean { return fit.fixed_gb + ownPathCacheGb(fit, context) <= vramGb; }
/** The conversation cache of this model at this context: its own rule where the catalog gives one, else the Flash-Next figure. */
export function contextCacheGb(context: number, kv = 'int8', fit?: ContextFit | null): number { return fit ? ownPathCacheGb(fit, context) : cacheGb(context, kv); }
export interface CacheChoice { context: number; kv?: string; /** the drafter this run loads (resolveDrafter in lib/drafting.ts); absent: the MTP, as the `mtp` argument says */ drafter?: string }

export function fitOn(v: FitInput, hw: FitHardware | null | undefined, mtp = true, cache?: CacheChoice | null): Fit {
  const memoryGb = memoryNeeded(v);
  // DENSE_GB already holds a 32K cache in the default format; the person's context and format replace it (#3832).
  const denseGb = cache && cache.context > 0 ? Math.max(DENSE_GB - cacheGb(BASE_CONTEXT), 0) + contextCacheGb(cache.context, cache.kv, v.context_fit) : DENSE_GB;
  const ram = Number(hw?.ram_gb), vram = Number(hw?.vram_gb) || 0;
  if (!Number.isFinite(ram) || ram <= 0) return { memoryGb, vramUseGb: 0, ramUseGb: 0, spillGb: 0, known: false };
  const experts = Math.max(0, memoryNeeded(v) - DENSE_EXTRA_GB);
  const drafterGb = cache?.drafter !== undefined ? (DRAFTER_VRAM_GB[cache.drafter] ?? 0) : mtp ? MTP_VRAM_GB : 0;
  const onGpu = Math.max(0, Math.min(experts, vram - denseGb - (vram > 0 ? drafterGb : 0)));
  const ramRoom = Math.max(0, ram - SYSTEM_ROOM_GB);
  const spillGb = Math.max(0, experts - onGpu - ramRoom);
  const inRam = Math.max(0, experts - onGpu - spillGb);
  const vramUseGb = vram > 0 ? Math.min(vram, denseGb + drafterGb + onGpu) : 0;
  return { memoryGb, vramUseGb: round1(vramUseGb), ramUseGb: round1(inRam + DENSE_EXTRA_GB * (vram > 0 ? 0 : 1)), spillGb: round1(spillGb), known: true };
}

/** The family groups of the version list, in the order shown. `family` is the engine's family id. */
export const VERSION_GROUPS: { id: string; title: string; families: string[] }[] = [
  { id: 'swift', title: 'Swift models', families: ['swift'] },
  { id: 'gsq', title: 'GSQ-RCO models', families: ['qwen', 'coder'] },
  { id: 'nvfp4', title: 'NVFP4 models', families: ['nvfp4'] },
  { id: 'gyro', title: 'Gyro models', families: ['gyro'] },
  { id: 'unsloth', title: 'Unsloth models', families: ['unsloth'] },
];

/** The fixed "Recommended" picks, in order: engine family + model name + the short note shown beside the name. */
export const RECOMMENDED: { family: string; model: string; note: string }[] = [
  { family: 'gyro', model: 'Gyro-S', note: 'the smallest' },
  { family: 'swift', model: 'IQ2_XS', note: 'good balance' },
  { family: 'qwen', model: 'IQ3_S', note: 'best quality for its size' },
  { family: 'unsloth', model: 'UD-Q4_K_XL', note: 'more quality' },
];

/** Is a group of the version list open? What the person chose wins; until they touch it, only the group holding the selected version is open
 *  (the running version's group used to stay open whatever was clicked). */
export const groupOpen = (opened: Record<string, boolean>, id: string, containsSelected: boolean): boolean => opened[id] ?? containsSelected;

/** What the engine's own pick for this PC is (catalog.recommendation: the memory tiers of setup.py and tools/strata_mcp.py, the same answer `strata launch` shows). */
export interface PcPick { family?: string | null; model?: string | null; why?: string }
export const PICK_NOTE = 'the pick for this PC';
export const OTHER_NOTE = 'the other choice';

/**
 * The recommended set, found in a version list (a version the catalog does not offer is skipped). The four fixed cards are only a display list;
 * the pick comes from the engine's own recommendation for this PC (RAM + VRAM tiers: 40 GB Swift 1.5 IQ2_XS, 48 GB IQ3_XXS, 64 GB and up IQ3_S), so
 * the page and `strata launch` agree. The pick comes first and is marked; when the engine names Gyro-M as the other choice (40 GB tier, NVIDIA and AMD only,
 * which the engine decides) that card follows it. Without a pick the list is the fixed four.
 */
export function recommendedVersions<V extends { sourceFamily: string; model: string }>(versions: V[], pick?: PcPick | null): { r: typeof RECOMMENDED[number]; v: V }[] {
  type Row = { r: typeof RECOMMENDED[number]; v: V };
  const find = (family: string, model: string) => versions.find(v => v.sourceFamily === family && v.model === model);
  const rows: Row[] = RECOMMENDED.map(r => ({ r, v: find(r.family, r.model) })).filter((x): x is Row => !!x.v);
  if (!pick?.family || !pick.model) return rows;
  const chosen = find(pick.family, pick.model);
  if (!chosen) return rows;
  const rest = rows.filter(x => x.v !== chosen);
  const first: Row = { r: { family: pick.family, model: pick.model, note: PICK_NOTE }, v: chosen };
  const out: Row[] = [first];
  const gyroM = /Gyro-M/.test(pick.why ?? '') ? find('gyro', 'Gyro-M') : undefined;
  if (gyroM) out.push({ r: { family: 'gyro', model: 'Gyro-M', note: OTHER_NOTE }, v: gyroM });
  return [...out, ...rest.filter(x => x.v !== gyroM)];
}

/**
 * The version list as the menu shows it, for one base model: first the versions to look at (the recommended ones, in the order given, then the ones already
 * on this computer, each by download size), then every other version by download size. A version is in exactly one of the two, so none is shown twice
 * and none is lost; the ones in `hidden` (shown elsewhere, as on the first-run page) are left out. `recommended` is one id or several.
 */
export function versionGroups<V extends { id: string; download_gb: number }>(versions: V[], hidden: ReadonlySet<string>, downloaded: ReadonlySet<string> | undefined, recommended?: string | readonly string[]): { top: V[]; more: V[] } {
  const seen = new Set<string>();
  const list = versions.filter(v => !hidden.has(v.id) && !seen.has(v.id) && !!seen.add(v.id));
  const bySize = (a: V, b: V) => a.download_gb - b.download_gb;
  const ids = typeof recommended === 'string' ? [recommended] : [...(recommended ?? [])];
  const picked = ids.map(id => list.find(v => v.id === id)).filter((v): v is V => !!v);
  const have = list.filter(v => !ids.includes(v.id) && downloaded?.has(v.id)).sort(bySize);
  const top = [...picked, ...have];
  const inTop = new Set(top.map(v => v.id));
  return { top, more: list.filter(v => !inTop.has(v.id)).sort(bySize) };
}

/**
 * What a version is on this computer (owner's one rule, 2026-11-08): its files, WITHOUT the n-gram / PLE / Engram lookup table's file (the download size minus
 * that file: `fit_gb`, else `memory_gb`), against the graphics memory plus the RAM, nothing taken off either side. They fit: 'fits', and the list says nothing.
 * They do not: 'ssd', and the list says "may run slower" (part is read from the SSD while it answers). Nothing here refuses anything: no card, or a Mac with
 * little memory, gets the same two words. (Not enough free disk is a different question; the download says it.)
 *
 * `fitOn` (above) still places the experts for the This computer card; `on_this_pc` no longer decides anything here.
 */
export type VersionFit = 'fits' | 'ssd' | 'unknown';
export function versionFit(v: FitInput & { on_this_pc?: string | null }, hw: FitHardware | null | undefined, cache?: CacheChoice | null): VersionFit {
  const fit = fitOn(v, hw, true, cache);
  if (!fit.known) return 'unknown';
  const size = typeof v.fit_gb === 'number' && Number.isFinite(v.fit_gb) ? v.fit_gb : fit.memoryGb;
  return size <= Number(hw?.ram_gb) + (Number(hw?.vram_gb) || 0) ? 'fits' : 'ssd';
}

/** A card as the installer's hardware report lists it. */
export interface RunCard { index: number; name: string; vram_gb: number; usable?: boolean; vendor?: string; arch?: string; uma?: boolean; integrated?: boolean; unified_gb?: number; /** a unified-memory chip: its BIOS carve-out (setup.py amd_apply_uma) */ dedicated_gb?: number }

/**
 * The graphics memory that counts for the fit label, of the largest card: strata_control/fit_label.py card_room_gb, the same rule (the Terminal and the
 * catalog's “on_this_pc” use it). A chip that shares the PC’s memory (Strix Halo: uma) counts only its own carve-out, because its “shared” part is the
 * RAM, already counted (QAGUI-STRIX 11-09: Q8_0, 133.8 GB, against 63.6 GB RAM + 95.8 GB carve-out-plus-shared said nothing; Control says “may run slower”
 * against 63.6 + 64). Pinned to the Python by tests/fit-label-rule.test.mjs.
 */
export function cardRoomGb(cards: readonly Pick<RunCard, 'vram_gb' | 'uma' | 'dedicated_gb'>[] | null | undefined): number {
  let best = 0;
  for (const c of cards ?? []) best = Math.max(best, c.uma ? Number(c.dedicated_gb) || 0 : Number(c.vram_gb) || 0);
  return best;
}

/**
 * The cards a model can be placed on: the ones this engine can drive (strata_control/gpu_fit.py `usable_cards` and `for_vendor`, same rule).
 * A card the engine has no code for, a card of the other make (an engine drives one make; the report numbers each make from 0, so
 * two cards could share a number) and a chip that shares the system memory while the computer has a card of its own are left out.
 */
export function runCards<T extends RunCard>(gpus: T[] | null | undefined, engineVendor?: string | null): T[] {
  const sharesMemory = (g: RunCard) => !!(g.integrated || g.uma || g.unified_gb);
  const mine = (gpus ?? []).filter(g => g.usable !== false && (!engineVendor || !g.vendor || g.vendor === engineVendor));
  const hasReal = mine.some(g => !sharesMemory(g) && g.vram_gb >= 4);
  return mine.filter(g => !(hasReal && sharesMemory(g)));
}
