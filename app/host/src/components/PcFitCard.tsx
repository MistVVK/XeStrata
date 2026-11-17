import { tr } from '../lib/i18n';
import type { Fit } from '../lib/native-fit';

const gb = (n: number) => `${Number.isInteger(n) ? n : n.toFixed(1)} GB`;
const about = (n: number) => `${Math.round(n)} GB`;

/** What this computer has free right now (real readings of the host; null where one is not available). */
export interface PcLive { vramTotalGb?: number; freeVramGb: number | null; freeRamGb: number | null; diskReadGbs: number | null }
/** A model that is loaded: what it takes of graphics memory and RAM, and how much of it is read from the SSD (the app's estimate: the engine does not report this split). */
export interface PcLoaded { id: string; label: string; fit: Fit }

function FreeMeter({ label, total, free }: { label: string; total: number; free: number }) {
  const used = total > 0 ? Math.max(0, Math.min(100, ((total - free) / total) * 100)) : 0;
  return <div className="pc-meter" data-low={free / (total || 1) < 0.1 || undefined}>
    <div className="pc-meter__top"><span>{tr(label)}</span><b>{tr("{0} free", [gb(free)])}</b><span className="muted">{tr("of {0}", [gb(total)])}</span></div>
    <div className="pc-meter__track" role="img" aria-label={tr("{0}: {1} free of {2}", [tr(label), gb(free), gb(total)])}><i style={{ transform: `scaleX(${used / 100})` }}/></div>
  </div>;
}

/**
 * "This computer": how much graphics memory and RAM is free right now. While a model is loaded that is all the card shows: the hardware line, the model's
 * name, "Loaded", what it takes and the SSD line are gone (owner, 11-01: the page already says what is loaded); only while the loaded model really reads
 * from the SSD, how fast it reads right now. Before anything is loaded the card says what the chosen version would take. That a version does not fit in
 * memory is said once, under the version picker ("may run slower"), not again here. (The n-gram table, which always sits on the SSD, is not counted.)
 */
export function PcFitCard({ gpuName, ramGb, vramGb, fit, inUse = null, live = null, loaded = [], selectedLoaded = false }: { gpuName?: string | null; ramGb: number | null; vramGb: number; fit: Fit | null; inUse?: { ramGb: number; vramGb: number } | null; live?: PcLive | null; loaded?: PcLoaded[]; selectedLoaded?: boolean }) {
  const showFit = !!fit?.known && !selectedLoaded;
  const isLoaded = loaded.length > 0;
  const streamed = Math.max(0, ...loaded.map(m => (m.fit.known ? m.fit.spillGb : 0)));
  return <section className="st-card pc-card" data-loaded={isLoaded ? true : undefined}>
    {isLoaded ? <h2 className="h4">{tr("This computer")}</h2> : <div><h2 className="h4">{tr("This computer")}</h2>
      <p className="small muted">{tr(gpuName ? tr("{0} · {1} graphics memory", [gpuName, gb(vramGb)]) : tr("No supported graphics card reported by this installation."))}{ramGb ? tr(" · {0} RAM", [gb(ramGb)]) : ''}</p></div>}
    {tr(live && (live.freeVramGb !== null || live.freeRamGb !== null) ? <div className="pc-free">
      <h3 className="pc-label">{tr("Free right now")}</h3>
      {tr(live.freeVramGb !== null && vramGb > 0 ? <FreeMeter label="Graphics memory" total={live.vramTotalGb ?? vramGb} free={live.freeVramGb}/> : null)}
      {tr(live.freeRamGb !== null && ramGb ? <FreeMeter label="RAM" total={ramGb} free={live.freeRamGb}/> : null)}
    </div> : null)}
    {tr(isLoaded && streamed >= 1 && live?.diskReadGbs && live.diskReadGbs >= 0.05 ? <p className="pc-note">{tr("Reading from your SSD now: {0}", [`${live.diskReadGbs.toFixed(1)} GB/s`])}</p> : null)}
    {tr(inUse ? <p className="xs muted">{tr("Models that stay loaded use about {0} GB of RAM and {1} GB of this card’s memory.", [inUse.ramGb, inUse.vramGb])}</p> : null)}
    {tr(showFit && !isLoaded && fit!.vramUseGb + fit!.ramUseGb > 0 ? <p className="pc-note">{tr("This version would take about {0} of graphics memory and {1} of RAM.", [about(fit!.vramUseGb), about(fit!.ramUseGb)])}</p> : null)}
  </section>;
}
