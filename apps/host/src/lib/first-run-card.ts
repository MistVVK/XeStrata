/**
 * The card the first-run screen names as "your graphics card" (#6015): the card setup will run the engine on. Setup's order
 * (setup.py amd_rank): a discrete card before graphics built into the processor, then the most memory, then the lower number. On many
 * PCs the processor's graphics comes first and reports a memory carve-out (a Ryzen 9800X3D: "AMD Radeon(TM) Graphics, 2 GB" beside an
 * RX 9070 with 16 GB); on a Strix Halo the carve-out can be larger than a discrete card's memory. A list without any memory figure keeps
 * the first card.
 */
export interface FirstRunCard {
    name: string;
    vramGb: number | null;
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

export function firstRunCard(gpus: FirstRunCard[]): FirstRunCard | undefined {
    if (!gpus.length) return undefined;
    if (gpus.every(g => g.vramGb == null)) return gpus[0];
    const rank = (g: FirstRunCard, i: number) => [builtIntoProcessor(g.name) ? 1 : 0, -Math.round(g.vramGb ?? 0), i];
    let best = 0;
    for (let i = 1; i < gpus.length; i++) {
        const a = rank(gpus[i], i), b = rank(gpus[best], best);
        if (a[0] < b[0] || (a[0] === b[0] && a[1] < b[1])) best = i;   // the lower number wins a tie (the loop keeps the earlier one)
    }
    return gpus[best];
}
