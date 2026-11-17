import { useEffect } from 'react';
import { api } from '../lib/client';
import { useStore } from '../lib/store';
import { PcFitCard, type PcLive } from './PcFitCard';

const tenth = (n: number | null | undefined) => typeof n === 'number' && Number.isFinite(n) ? Math.round(n * 10) / 10 : null;

/**
 * "This computer" with the host's own readings: while the Models page is open the host reads the card every few seconds (the same ask the
 * Monitor makes), and the free graphics memory and RAM come from those readings, not from a guess. It subscribes to the three numbers only,
 * so the Models page itself does not redraw on every reading.
 */
export function PcComputer(props: Omit<React.ComponentProps<typeof PcFitCard>, 'live'> & { vramAllGb?: number }) {
  const vramUsed = useStore(s => tenth(s.monitor?.hardware?.vramUsedGb));
  const ramUsed = useStore(s => tenth(s.monitor?.hardware?.ramUsedGb));
  const diskRead = useStore(s => tenth(s.monitor?.hardware?.diskReadGbs));
  useEffect(() => {
    const ask = () => { if (typeof document === 'undefined' || document.visibilityState !== 'hidden') void api.get('/api/monitor').catch(() => undefined); };
    ask();
    const timer = setInterval(ask, 10000);
    return () => clearInterval(timer);
  }, []);
  const vramTotal = props.vramAllGb ?? props.vramGb;
  const live: PcLive = {
    vramTotalGb: vramTotal,
    freeVramGb: vramUsed !== null && vramTotal > 0 ? Math.max(0, Math.round((vramTotal - vramUsed) * 10) / 10) : null,
    freeRamGb: ramUsed !== null && props.ramGb ? Math.max(0, Math.round((props.ramGb - ramUsed) * 10) / 10) : null,
    diskReadGbs: diskRead,
  };
  return <PcFitCard {...props} live={live}/>;
}
