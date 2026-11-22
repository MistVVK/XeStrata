/**
 * The card the first-run screen names as "your graphics card" (#6015, #6619): the card setup will run the engine on.
 * Small built-in graphics (a 2 GB Ryzen iGPU, 780M/890M, Intel UHD) always come after a discrete card, while a Big UMA
 * workstation APU (AMD Strix Halo 8060S / gfx1151 with >= 48 GB unified memory) competes on memory alongside discrete cards
 * so a 128 GB 8060S is not hidden behind a smaller discrete card (#6619). A list without any memory figure keeps the first card.
 */
export interface FirstRunCard {
    name: string;
    vramGb: number | null;
    arch?: string | null;
    uma?: boolean;
    integrated?: boolean;
    unified_gb?: number | null;
    usable?: boolean;
}

/**
 * Graphics built into the processor, by the name Windows and the drivers report: AMD's "Radeon(TM) Graphics", "Radeon 780M / 890M /
 * 8060S Graphics" and Intel's "UHD / Iris Xe / Arc(TM) Graphics" carry no model number of a card. A discrete card names one: "Radeon
 * RX 9070", "Radeon PRO W7900", "Radeon AI PRO R9700", "Arc(TM) A750 Graphics", "Arc B580". NVIDIA names are always discrete here.
 */
export function builtIntoProcessor(name: string): boolean {
    const n = String(name ?? '');
    if (/nvidia|geforce|quadro|tesla/i.test(n)) return false;
    if (/amd|radeon/i.test(n)) return /graphics\s*$/i.test(n) && !/\b(RX|PRO|FirePro|Instinct)\b/i.test(n);
    if (/intel/i.test(n)) return !/\bArc\b[^A-Za-z0-9]*(\(TM\)\s*)?[AB]\d{3}/i.test(n);
    return false;
}

/**
 * High-memory unified-memory APU (e.g. AMD Strix Halo 8060S / gfx1151 with 64-128 GB UMA): unlike a small 2-8 GB iGPU,
 * it is a first-class LLM accelerator that can hold large models a 16-32 GB discrete card cannot (#6619).
 */
export function isBigUma(card: Pick<FirstRunCard, 'name' | 'vramGb' | 'arch' | 'uma' | 'integrated' | 'unified_gb'>): boolean {
    const name = String(card?.name ?? '');
    const arch = String(card?.arch ?? '');
    const mem = Math.max(Number(card?.vramGb) || 0, Number(card?.unified_gb) || 0);
    const shared = !!(card?.uma || card?.integrated || card?.unified_gb || builtIntoProcessor(name));
    if (!shared) return false;
    if (mem >= 48) return true;
    return /gfx1151|strix\s*halo|\b80[456]0s\b/i.test(`${name} ${arch}`) && (mem === 0 || mem >= 32);
}

export function firstRunCard(gpus: FirstRunCard[]): FirstRunCard | undefined {
    if (!gpus.length) return undefined;
    const usable = gpus.filter(g => g.usable !== false);
    const pool = usable.length ? usable : gpus;
    if (pool.every(g => g.vramGb == null && g.unified_gb == null)) return pool[0];
    const rank = (g: FirstRunCard, i: number) => [
        builtIntoProcessor(g.name) && !isBigUma(g) ? 1 : 0,
        -Math.round(Math.max(Number(g.vramGb) || 0, Number(g.unified_gb) || 0)),
        i,
    ];
    let best = 0;
    for (let i = 1; i < pool.length; i++) {
        const a = rank(pool[i], i), b = rank(pool[best], best);
        if (a[0] < b[0] || (a[0] === b[0] && a[1] < b[1])) best = i;   // the lower number wins a tie (the loop keeps the earlier one)
    }
    return pool[best];
}
