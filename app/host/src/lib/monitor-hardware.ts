import type { GpuInfo, GpuReading } from './types';
export const hasReading=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v);
export const gpuId=(gpu:GpuInfo)=>gpu.id??`${gpu.vendor}:${gpu.index}`;

/** A total, mean, or maximum must cover every selected device. Partial totals are misleading. */
export function gpuMetric(devices:GpuInfo[],readings:GpuReading[]|undefined,key:keyof GpuReading,mode:'sum'|'mean'|'max'='sum'):number|null {
  const values=devices.map(device=>readings?.find(reading=>reading.id===gpuId(device))?.[key]);
  if(!values.length||!values.every(hasReading))return null;
  if(key==='memoryUsedGb'&&mode==='sum') {
    const shared=values.filter((_,i)=>devices[i].unified),dedicated=values.filter((_,i)=>!devices[i].unified);
    return dedicated.reduce((n,v)=>n+v,0)+(shared.length?Math.max(...shared):0);
  }
  return mode==='max'?Math.max(...values):values.reduce((n,v)=>n+v,0)/(mode==='mean'?values.length:1);
}

/**
 * The total next to the VRAM reading: the larger of what the host's probe says for the selected cards and what the engine reports for the same cards
 * (the Windows probe can report less than the card has, a Strix Halo with a 96 GB carve-out read as 64 GB), and none at all when the reading is still
 * above it: a figure above its own total is wrong whichever number is. Both are written to the debug console when they change, so a run on the
 * real machine can say which one is right.
 */
let lastTotals='';
export function gpuTotalShown(devices:GpuInfo[],readings:GpuReading[]|undefined,used:number|null):number {
  const host=gpuCapacity(devices);
  const engineValues=devices.map(device=>readings?.find(reading=>reading.id===gpuId(device))?.memoryTotalGb);
  const engine=engineValues.length&&engineValues.every(hasReading)
    ?devices.filter(d=>!d.unified).reduce((n,d)=>n+(readings?.find(r=>r.id===gpuId(d))?.memoryTotalGb as number),0)+Math.max(0,...devices.filter(d=>d.unified).map(d=>readings?.find(r=>r.id===gpuId(d))?.memoryTotalGb as number))
    :NaN;
  const total=Math.max(host,hasReading(engine)?engine:0);
  const signature=`${host}|${hasReading(engine)?engine:'-'}`;
  if(signature!==lastTotals){lastTotals=signature;try{console.debug('[monitor] GPU memory total, GB: host probe',host,'engine',hasReading(engine)?Math.round(engine*10)/10:'not reported','shown',total);}catch{/* no console */}}
  return hasReading(used)&&used>total+0.05?NaN:total;
}

/** The card's name without its maker: "NVIDIA GeForce RTX 5070" -> "RTX 5070". */
export const shortGpuName=(name:string)=>String(name??'').replace(/^(NVIDIA|AMD|Intel)(\(R\)|\(TM\))?\s+/i,'').replace(/^(GeForce|Radeon|Arc)(\(TM\))?\s+/i,'').trim()||String(name??'');

/**
 * The reading over the cards that DO report this value. A card that cannot report it (an integrated one without a temperature sensor) is left
 * out instead of blanking the figure; `partial` says some were left out, and `devices` names the cards the figure is from.
 */
export function gpuReported(devices:GpuInfo[],readings:GpuReading[]|undefined,key:keyof GpuReading,mode:'sum'|'mean'|'max'='sum'):{value:number|null;devices:GpuInfo[];partial:boolean} {
  const reporting=devices.filter(device=>hasReading(readings?.find(reading=>reading.id===gpuId(device))?.[key]));
  return {value:reporting.length?gpuMetric(reporting,readings,key,mode):null,devices:reporting,partial:reporting.length>0&&reporting.length<devices.length};
}

export function gpuCapacity(devices:GpuInfo[]) {
  const shared=devices.filter(g=>g.unified).map(g=>g.vramGb);
  return devices.filter(g=>!g.unified).reduce((n,g)=>n+g.vramGb,0)+(shared.length?Math.max(...shared):0);
}

/**
 * The cards "All GPUs" covers by default. A small integrated chip (an AMD or Intel adapter on the Windows/Vulkan list with a carve-out under
 * 2 GB) beside a real card is left out: Strata cannot run a model on it (the native engine report marks it unusable), and counting it adds a
 * second "GPU" and half a gigabyte to every total. It stays in the picker. Alone (a Strix Halo, an iGPU-only PC) it stays the default.
 */
export function defaultGpuSet(devices:GpuInfo[]):GpuInfo[] {
  const real=(g:GpuInfo)=>g.vramGb>=4||g.vendor==='nvidia'||g.backend==='cuda';
  const small=(g:GpuInfo)=>g.backend==='vulkan'&&(g.vendor==='amd'||g.vendor==='intel')&&g.vramGb<2&&!g.unified;
  if(!devices.some(real))return devices;
  const kept=devices.filter(g=>!small(g));
  return kept.length?kept:devices;
}
