/**
 * The card the first-run screen names as "your graphics card" (#6015). A processor's own graphics comes first in the list on many PCs and can
 * report a small memory carve-out (an AMD Ryzen 9800X3D's iGPU: "AMD Radeon(TM) Graphics, 2 GB" beside an RX 9070 with 16 GB), so the
 * first card with any memory is not the card the engine is for. The card with the most memory of its own is; a list without any memory
 * figure falls back to the first card. Order among equal cards is kept.
 */
export interface FirstRunCard {
    name: string;
    vramGb: number | null;
}

export function firstRunCard(gpus: FirstRunCard[]): FirstRunCard | undefined {
    let best: FirstRunCard | undefined;
    for (const g of gpus) {
        if ((g.vramGb ?? 0) < 1) continue;            // no memory of its own: not "a card"
        if (!best || (g.vramGb ?? 0) > (best.vramGb ?? 0)) best = g;
    }
    return best ?? gpus[0];
}
