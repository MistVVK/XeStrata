import type { Hardware } from './types';

/** Inventory total only. Placement and fit continue to use separate budgets. */
export function memoryInventory(hardware: Hardware) {
  const ram = hardware.memory.totalGb;
  const dedicated = hardware.gpus.filter(g => !g.unified && g.backend !== 'cpu');
  const vram = dedicated.reduce((sum, gpu) => sum + Math.max(0, gpu.vramGb || 0), 0);
  return { ram, vram, total: ram + vram, shared: hardware.accelerator.unified && !vram };
}

/**
 * The engine's size for each card the host reports with no dedicated memory (an old Windows probe read 0 for AMD cards, #3727).
 * Each engine card is used once: same name first (a "7900 XT" never takes the "7900 XTX" figure while the XT is listed), then a name
 * that contains the other, then the same position when both lists have the same length. A card left without a match stays at 0.
 * Returns one figure per host card (the host's own figure where it had one).
 */
export function engineVramForHostCards(host: { name: string; vramGb: number; unified?: boolean }[], engine: { name: string; vram_gb: number }[]): number[] {
  const norm = (value: string) => value.toLowerCase().replace(/[^a-z0-9]/g, '');
  const out = host.map(g => (g.unified || g.vramGb > 0 ? g.vramGb : 0));
  const need = host.map((g, i) => (g.unified || g.vramGb > 0 ? -1 : i)).filter(i => i >= 0);
  const used = new Set<number>();
  const take = (i: number, k: number) => { used.add(k); out[i] = Math.max(0, engine[k].vram_gb || 0); need.splice(need.indexOf(i), 1); };
  const rules: ((a: string, b: string) => boolean)[] = [(a, b) => a === b, (a, b) => !!a && !!b && (a.includes(b) || b.includes(a))];
  for (const same of rules) {
    for (const i of [...need]) {
      const k = engine.findIndex((e, n) => !used.has(n) && same(norm(e.name), norm(host[i].name)));
      if (k >= 0) take(i, k);
    }
  }
  if (engine.length === host.length) for (const i of [...need]) if (!used.has(i)) take(i, i);
  return out;
}

export function memoryReading(used:number,total:number){const safeTotal=Number.isFinite(total)?Math.max(0,total):0;const safeUsed=Number.isFinite(used)?Math.max(0,used):0;return {used:Math.min(safeUsed,safeTotal),total:safeTotal,free:Math.max(0,safeTotal-safeUsed),over:Math.max(0,safeUsed-safeTotal)};}
