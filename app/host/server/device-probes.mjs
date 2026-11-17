/** Physical topology and GPU sensors; unavailable values stay null. */
import { readFile, readdir, realpath } from 'node:fs/promises';
import { probe, round } from './util.mjs';
import { windowsGpuReading } from './windows-gpu.mjs';

/** Unified GPUs share one system memory pool; dedicated GPUs each have their own. */
export function gpuMemoryTotals(devices, readings) {
  const id = gpu => gpu.id ?? `${gpu.vendor}:${gpu.index}`;
  const values = devices.map(gpu => readings.find(r => r.id === id(gpu))?.memoryUsedGb);
  const total = devices.filter(g => !g.unified).reduce((n,g) => n + g.vramGb,0)
    + Math.max(0,...devices.filter(g => g.unified).map(g => g.vramGb));
  if (!values.length || values.some(v => typeof v !== 'number' || !Number.isFinite(v))) return {used:null,total};
  const used = values.filter((_,i) => !devices[i].unified).reduce((n,v) => n+v,0)
    + Math.max(0,...values.filter((_,i) => devices[i].unified));
  return {used,total};
}

export function parseCpuTopology(raw, model, arch, ghz = 0, names = new Map()) {
  const sockets = new Map();
  for (const line of raw.split('\n')) {
    if (!line.trim() || line.startsWith('#')) continue;
    const [cpu, core, socket] = line.split(',').map(v=>v.trim());
    if (!/^\d+$/.test(cpu) || !/^\d+$/.test(core) || !/^\d+$/.test(socket)) continue;
    const entry = sockets.get(socket) ?? { cores:new Set(), threads:new Set() };
    entry.cores.add(core); entry.threads.add(cpu); sockets.set(socket,entry);
  }
  return [...sockets].map(([socket,s])=>({id:`socket-${socket}`,socket,model:names.get(socket)??model,cores:s.cores.size,threads:s.threads.size,logicalIds:[...s.threads].map(Number),arch,ghz}));
}
export function parseWindowsProcessors(raw, arch) {
  try {
    const parsed = JSON.parse(raw), rows = Array.isArray(parsed)?parsed:[parsed];
    return rows.filter(r=>r?.DeviceID&&Number(r.NumberOfCores)>0&&Number(r.NumberOfLogicalProcessors)>0).map(r=>({id:String(r.DeviceID),socket:String(r.SocketDesignation??r.DeviceID),model:String(r.Name??'Unknown processor').trim(),cores:Number(r.NumberOfCores),threads:Number(r.NumberOfLogicalProcessors),arch,ghz:round(Number(r.MaxClockSpeed??0)/1000,2)}));
  } catch { return []; }
}
const number = value => value!=null&&String(value).trim()!==''&&Number.isFinite(Number(value))?Number(value):null;
export function parseNvidiaDevices(raw) {
  return raw.split('\n').flatMap(line=>{
    const [index,uuid,bus,name,memory,driver]=line.split(',').map(s=>s.trim());
    if (!uuid?.startsWith('GPU-') || !name) return [];
    return [{id:uuid,index:Number(index),sensorIndex:Number(index),busId:bus?.toLowerCase(),vendor:'nvidia',name,vramGb:round((number(memory)??0)/1024,1),unified:false,backend:'cuda',driver}];
  });
}
/** A percentage reading; "[N/A]" and out-of-range values are no reading. */
const percentOf = v => { const n=number(v); return n!==null&&n>=0&&n<=100?n:null; };
export function parseNvidiaReadings(raw) {
  return raw.split('\n').flatMap(line=>{
    const [id,temp,load,memory,power,limit,gen,maxGen,width,fan,clock]=line.split(',').map(s=>s.trim());
    if (!id?.startsWith('GPU-')) return [];
    const t=number(temp),u=number(load),m=number(memory);
    const positive = v => { const n=number(v); return n!==null&&n>=0?n:null; };
    return [{id,temperatureC:t!==null&&t>=0&&t<=150?t:null,utilizationPct:u!==null&&u>=0&&u<=100?u:null,memoryUsedGb:m!==null&&m>=0?round(m/1024,2):null,powerW:positive(power),powerLimitW:positive(limit),pcieGen:positive(gen),pcieGenMax:positive(maxGen),pcieWidth:positive(width),pcieRxMbs:null,fanPct:percentOf(fan),clockMhz:positive(clock)}];
  });
}
/** A GP100 reports just under 200 GiB/s now and then, even idle (see serve/telemetry.py). No PCIe link carries that: 6.0 x16 is under 120 GiB/s. */
const PCIE_MBS_MAX=150*1024;
export function parsePcieThroughput(raw) {
  const values=new Map();
  for(const line of raw.split('\n')) {
    if(line.trim().startsWith('#'))continue;
    const fields=line.trim().split(/\s+/),index=number(fields[0]),rx=number(fields[1]);
    if(fields.length===3&&index!==null&&Number.isInteger(index)&&index>=0&&rx!==null&&rx>=0&&rx<=PCIE_MBS_MAX)values.set(index,rx);
  }
  return values;
}
async function read(path) { try {return (await readFile(path,'utf8')).trim();} catch {return '';} }
async function list(path) {try{return await readdir(path);}catch{return [];}}
/** A server's remote-console chip (ASPEED), a virtual display or a framebuffer is not a graphics card: no maker we know and no memory of its own. */
export function isComputeAdapter(vendor,vramBytes) { return vendor!=='unknown'||vramBytes>0; }
/**
 * The temperature of an amdgpu card from its hwmon inputs [{label, milli}]: the edge sensor (what nvtop and rocm-smi show), else the
 * junction, else the first one that has a plausible value. Cards such as the Instinct line have no edge sensor. 0 is no reading.
 */
export function pickGpuTemperature(inputs) {
  const ok=inputs.filter(i=>i.milli!==null&&i.milli>0&&i.milli<=150000);
  const first=re=>ok.find(i=>re.test(i.label??''));
  const chosen=first(/edge|^gpu/i)??ok.find(i=>!i.label)??first(/junction|hotspot/i)??ok.find(i=>!/mem/i.test(i.label??''));
  return chosen?chosen.milli/1000:null;
}
export async function detectDrmGpus() {
  const result=[];
  for(const card of (await list('/sys/class/drm')).filter(n=>/^card\d+$/.test(n))) {
    const device=`/sys/class/drm/${card}/device`;
    let resolved;try{resolved=await realpath(device);}catch{continue;}
    const busId=resolved.split('/').at(-1),vendorId=await read(`${device}/vendor`);
    const vendor={'0x1002':'amd','0x8086':'intel','0x10de':'nvidia'}[vendorId]??'unknown';
    const name=await probe('lspci',['-s',busId]);
    const bytes=number(await read(`${device}/mem_info_vram_total`));
    if(!isComputeAdapter(vendor,bytes))continue;
    result.push({id:`drm:${busId}`,index:result.length,busId,vendor,name:name.replace(/^[^ ]+\s+[^:]+:\s*/,'')||`${vendor.toUpperCase()} GPU`,vramGb:bytes===null?0:round(bytes/1024**3,1),unified:vendor==='intel'&&!(bytes>0),backend:'vulkan',sensorPath:device});
  }
  return result;
}
/** `transfer: false` skips the PCIe throughput sample (a second nvidia-smi); it only means something while a model runs. */
export async function readGpuTelemetry(gpus, { transfer: wantTransfer = true, run = probe } = {}) {
  const now=new Date().toISOString();
  const hasNvidia=gpus.some(g=>g.vendor==='nvidia');
  let raw=hasNvidia?await run('nvidia-smi',['--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used,power.draw,power.limit,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,fan.speed,clocks.current.graphics','--format=csv,noheader,nounits'],2000):'';
  // Older drivers can reject a field; retain their basic sensors in that case.
  if(hasNvidia&&!raw)raw=await run('nvidia-smi',['--query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits'],2000);
  const nvidia=new Map(parseNvidiaReadings(raw).map(r=>[r.id,r]));
  const transfer=hasNvidia&&wantTransfer?parsePcieThroughput(await run('nvidia-smi',['dmon','-s','t','-c','1'],1500)):new Map();
  return Promise.all(gpus.map(async gpu=>{
    let reading=nvidia.get(gpu.id)??{id:gpu.id??`${gpu.vendor}:${gpu.index}`,temperatureC:null,utilizationPct:null,memoryUsedGb:null,powerW:null,powerLimitW:null,pcieGen:null,pcieGenMax:null,pcieWidth:null,pcieRxMbs:null};
    if(gpu.vendor==='nvidia')reading={...reading,pcieRxMbs:transfer.get(gpu.sensorIndex??gpu.index)??null};
    if(gpu.luid&&!nvidia.has(gpu.id)) {
      const counters=windowsGpuReading(gpu.luid);
      if(counters)reading={...reading,utilizationPct:counters.utilizationPct,memoryUsedGb:round(counters.memoryUsedGb,2),temperatureC:counters.temperatureC??null};
    }
    if(gpu.sensorPath&&!nvidia.has(gpu.id)) {
      let temperature=null,power=null,powerLimit=null,fanPwm=null,clockHz=null;
      for(const hwmon of await list(`${gpu.sensorPath}/hwmon`)) {
        const folder=`${gpu.sensorPath}/hwmon/${hwmon}`,inputs=(await list(folder)).filter(n=>/^temp\d+_input$/.test(n));
        power ??= number(await read(`${folder}/power1_average`)) ?? number(await read(`${folder}/power1_input`));
        powerLimit ??= number(await read(`${folder}/power1_cap`));
        fanPwm ??= number(await read(`${folder}/pwm1`));          // 0-255, the fan's duty as the amdgpu driver reports it
        clockHz ??= number(await read(`${folder}/freq1_input`));  // the graphics clock in Hz
        const sensors=[];
        for(const input of inputs)sensors.push({label:await read(`${folder}/${input.replace('_input','_label')}`),milli:number(await read(`${folder}/${input}`))});
        temperature=pickGpuTemperature(sensors);
        if(temperature!==null)break;
      }
      const utilization=number(await read(`${gpu.sensorPath}/gpu_busy_percent`)),bytes=number(await read(`${gpu.sensorPath}/mem_info_vram_used`));
      const gen=path=>read(path).then(raw=>({2.5:1,5:2,8:3,16:4,32:5,64:6}[Number.parseFloat(raw)]??null));
      const [pcieGen,pcieGenMax,width]=await Promise.all([gen(`${gpu.sensorPath}/current_link_speed`),gen(`${gpu.sensorPath}/max_link_speed`),read(`${gpu.sensorPath}/current_link_width`)]);
      reading={...reading,fanPct:fanPwm!==null&&fanPwm>=0&&fanPwm<=255?Math.round(fanPwm/255*100):null,clockMhz:clockHz!==null&&clockHz>0?Math.round(clockHz/1e6):null,temperatureC:temperature,utilizationPct:utilization!==null&&utilization>=0&&utilization<=100?utilization:null,memoryUsedGb:bytes!==null&&bytes>=0?round(bytes/1024**3,2):null,powerW:power!==null&&power>=0?power/1e6:null,powerLimitW:powerLimit!==null&&powerLimit>=0?powerLimit/1e6:null,pcieGen,pcieGenMax,pcieWidth:number(width)};
    }
    return {...reading,sampledAt:now,source:Object.entries(reading).some(([key,value])=>key!=='id'&&typeof value==='number')?'sensor':'none'};
  }));
}
