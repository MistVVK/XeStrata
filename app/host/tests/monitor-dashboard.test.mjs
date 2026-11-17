import test from 'node:test';
import assert from 'node:assert/strict';
import { parseCpuTopology, parseNvidiaReadings, parsePcieThroughput, gpuMemoryTotals, pickGpuTemperature, isComputeAdapter } from '../server/device-probes.mjs';
import { cpuUtilization, diskRate, parseLinuxDiskCounters, parseMacDiskCounters } from '../server/system-telemetry.mjs';
import { parseEngineMetrics } from '../server/supervisor.mjs';
import { originalDashboard, mergeOwnReading } from '../server/monitor-dashboard.mjs';
import { gpuMetric, gpuCapacity, gpuReported, shortGpuName, defaultGpuSet, gpuId } from '../src/lib/monitor-hardware.ts';
import { referencePaths } from '../src/lib/monitor-chart.ts';

const gpu=(id,index=0,unified=false)=>({id,index,unified,vramGb:24,name:id,vendor:'nvidia',backend:'cuda'});
const reading=(id,utilizationPct,memoryUsedGb,temperatureC,powerW)=>({id,utilizationPct,memoryUsedGb,temperatureC,powerW});

test('two GPUs use stable IDs, mean load, total memory/power, and hottest temperature',()=>{
  const devices=[gpu('a'),gpu('b',1)],readings=[reading('b',80,10,72,120),reading('a',20,4,48,60)];
  assert.equal(gpuMetric(devices,readings,'utilizationPct','mean'),50);
  assert.equal(gpuMetric(devices,readings,'memoryUsedGb'),14);
  assert.equal(gpuMetric(devices,readings,'temperatureC','max'),72);
  assert.equal(gpuMetric(devices,readings,'powerW'),180);
  assert.equal(gpuMetric([devices[1]],readings,'utilizationPct','mean'),80);
  assert.equal(gpuCapacity(devices),48);
});
test('missing sensors remain unavailable, including partial totals',()=>{
  const devices=[gpu('a'),gpu('b',1)];
  assert.equal(gpuMetric(devices,[reading('a',20,4,48,60)],'memoryUsedGb'),null);
  assert.equal(gpuMetric(devices,[reading('a',20,4,48,60),reading('b',null,10,72,null)],'utilizationPct','mean'),null);
  assert.equal(gpuMetric([],[],'powerW'),null);
});
test('unified memory capacity and readings do not count the same pool twice',()=>{
  const devices=[gpu('a',0,true),gpu('b',1,true)];
  assert.equal(gpuCapacity(devices),24);
  assert.equal(gpuMetric(devices,[reading('a',20,4,48,60),reading('b',80,4,72,120)],'memoryUsedGb'),4);
  assert.deepEqual(gpuMemoryTotals(devices,[reading('a',20,4,48,60),reading('b',80,4,72,120)]),{used:4,total:24});
  assert.deepEqual(gpuMemoryTotals([...devices,gpu('c',2)],[reading('a',20,4,48,60),reading('b',80,4,72,120),reading('c',20,10,50,100)]),{used:14,total:48});
  assert.deepEqual(gpuMemoryTotals(devices,[reading('a',20,4,48,60)]),{used:null,total:24});
});
test('CPU topology keeps actual logical CPU IDs per socket',()=>{
  const processors=parseCpuTopology('# CPU,Core,Socket\n0,0,0\n1,1,0\n2,0,1\n3,1,1','Xeon','x64');
  assert.deepEqual(processors.map(p=>p.logicalIds),[[0,1],[2,3]]);
  assert.deepEqual(processors.map(p=>p.cores),[2,2]);
});
test('per socket CPU utilization is a counter delta rather than a model or core count estimate',()=>{
  const make=values=>values.map(([busy,idle])=>({times:{user:busy,nice:0,sys:0,idle,irq:0}}));
  const before=make([[0,0],[0,0],[0,0],[0,0]]),after=make([[25,75],[25,75],[90,10],[90,10]]);
  assert.equal(cpuUtilization(before,after,[0,1]),25);
  assert.equal(cpuUtilization(before,after,[2,3]),90);
  assert.equal(cpuUtilization(null,after,[0]),null);
  assert.equal(cpuUtilization(before,after,[]),null);
  assert.equal(cpuUtilization(before,after,[7]),null);
});
test('disk counters exclude partitions to avoid double counting',()=>{
  const row=(name,read,write)=>`8 0 ${name} 0 0 ${read} 0 0 0 ${write} 0 0 0 0`;
  assert.deepEqual(parseLinuxDiskCounters([row('sda',10,20),row('sda1',10,20),row('sdb',5,10)].join('\n'),['sda','sdb']),{read:15*512,write:30*512});
  assert.deepEqual(parseMacDiskCounters('"Bytes (Read)"=100, "Bytes (Write)"=200\n"Bytes (Read)"=50, "Bytes (Write)"=20'),{read:150,write:220});
  assert.deepEqual(diskRate({time:1000,counters:{read:0,write:0}},{read:2*1024**2,write:1024**2},3000),{readMbs:1,writeMbs:.5});
  assert.equal(diskRate(null,{read:4,write:2},3000).readMbs,null);
  assert.equal(diskRate({time:1000,counters:{read:10,write:10}},{read:0,write:0},3000).readMbs,null);
});
test('NVIDIA power and PCIe link readings survive unavailable driver fields',()=>{
  const [r]=parseNvidiaReadings('GPU-1, 60, 70, 4096, 123.5, 250, 3, 4, 16');
  assert.equal(r.memoryUsedGb,4);assert.equal(r.powerW,123.5);assert.equal(r.pcieGenMax,4);assert.equal(r.pcieWidth,16);
  const [missing]=parseNvidiaReadings('GPU-1, N/A, 0, 0, [Not Supported], N/A, N/A, N/A, N/A');
  assert.equal(missing.temperatureC,null);assert.equal(missing.powerW,null);assert.equal(missing.utilizationPct,0);
  const transfer=parsePcieThroughput('# gpu rxpci txpci\n# Idx MB/s MB/s\n0 12.5 8\n1 200 0\n2 - -');
  assert.equal(transfer.get(0),12.5);assert.equal(transfer.get(1),200);assert.equal(transfer.has(2),false);
  const wrapped=parsePcieThroughput('# Idx MB/s MB/s\n0 59392 0\n1 209673 0\n2 5 209710');
  assert.equal(wrapped.get(0),59392);assert.equal(wrapped.has(1),false);assert.equal(wrapped.get(2),5);
});
test('chart paths reuse the exact original 100×32 geometry',()=>{
  const {paths}=referencePaths([0,5,10]);
  assert.equal(paths[0].line,'M0.00,30.00L50.00,17.00L100.00,4.00');
  assert.equal(paths[0].area,'M0.00,30.00L50.00,17.00L100.00,4.00L100.00,32L0.00,32Z');
});
test('charts leave gaps for missing sensors and do not draw a made up zero sample',()=>{
  assert.equal(referencePaths([null,null]).paths.length,0);
  assert.equal(referencePaths([1]).paths.length,0);
  const {paths}=referencePaths([1,2,null,3,4]);
  assert.equal(paths.length,2);
  assert.ok(paths[0].line.endsWith('25.00,17.00'));
  assert.ok(paths[1].line.startsWith('M75.00'));
});
test('both supported engine metric formats keep their actual telemetry',()=>{
  const prom=parseEngineMetrics('llamacpp:prompt_tokens_total 4\nllamacpp:predicted_tokens_seconds 42');
  assert.equal(prom.outputTps,42);
  const raw={engine:{max_context:32768},live:{state:'generating',tok_s:44},hardware:{},requests:[]};
  assert.equal(parseEngineMetrics(JSON.stringify(raw)).dashboard.live.tok_s,44);
  assert.throws(()=>parseEngineMetrics('<html>error</html>'));
});
test('original requests, cache, progress, hardware and MCP facts are translated without losing information',()=>{
  const raw={engine:{max_context:32768,expert_slots:48,expert_cache_mib:8192},live:{state:'generating',generated:120,max_tokens:600,prompt_tokens:200,tok_s:44},hardware:{gpu_mem_used:4*1024**3,gpu_mem_total:24*1024**3},history:{tok_s:[42,44]},requests:[{time:1,finish:'stop',prompt_tokens:200,reused:100,output_tokens:120,decode_tok_s:44,hit_rate:.85,pcie_share:.1,duration_s:3}],totals:{requests:20,prompt_tokens:4000,output_tokens:2400,reused:2000,since:1},conversation_cache:{enabled:true,budget_mib:8192,parked:2,slots:4,bytes:2*1024**3,parks:3,restores:1,requests:20,requests_reused:10},mcp:{tools:1,servers:[{name:'files',status:'ready',transport:'stdio',tools:[{tool:'read_file',description:'Read a file'}]}]}};
  const mapped=originalDashboard(raw,{gpus:[gpu('a')]},{context:{maxTokens:32768},hardware:{},history:[],totals:{},telemetry:{}});
  assert.equal(mapped.requests[0].cachedTokens,100);assert.equal(mapped.requests[0].hitRate,.85);
  assert.equal(mapped.conversationCache.memoryGb,2);assert.equal(mapped.totals.cachedTokens,2000);
  assert.equal(mapped.expertCache.memoryGb,8);assert.equal(mapped.hardware.vramUsedGb,4);
  assert.equal(mapped.generation.total,600);assert.equal(mapped.history[0].gpuTempC,null);
  assert.equal(mapped.mcp.servers[0].state,'ready');assert.equal(mapped.mcp.servers[0].tools[0].name,'read_file');
});

test('All GPUs: a card with no temperature sensor is left out and the card that reports is named (#3727 follow-up)',()=>{
  const devices=[{...gpu('nv',0),name:'NVIDIA GeForce RTX 5070'},{...gpu('igpu',1),name:'AMD Radeon(TM) Graphics',vramGb:0.4}];
  const readings=[reading('nv',3,0.8,34,7),reading('igpu',0,0.001,null,null)];
  const temp=gpuReported(devices,readings,'temperatureC','max');
  assert.equal(temp.value,34); assert.equal(temp.partial,true); assert.deepEqual(temp.devices.map(d=>shortGpuName(d.name)),['RTX 5070']);
  const load=gpuReported(devices,readings,'utilizationPct','mean');
  assert.equal(load.partial,false); assert.equal(load.value,1.5);
  assert.equal(gpuReported(devices,[reading('nv',3,0.8,null,null),reading('igpu',0,0,null,null)],'temperatureC','max').value,null,'no card reports: still unavailable');
  assert.equal(shortGpuName('AMD Radeon RX 9070 XT'),'RX 9070 XT');
  assert.equal(shortGpuName('Intel(R) Arc(TM) A750 Graphics'),'A750 Graphics');
});

test('an RTX 5070 beside the unsupported Radeon iGPU: All GPUs is the RTX alone, the iGPU stays pickable',()=>{
  const rtx={id:'GPU-5070',index:0,vendor:'nvidia',name:'NVIDIA GeForce RTX 5070',vramGb:11.9,unified:false,backend:'cuda'};
  const igpu={id:'PCI\VEN_1002&DEV_164E',index:1,vendor:'amd',name:'AMD Radeon(TM) Graphics',vramGb:0.5,unified:false,backend:'vulkan'};
  const set=defaultGpuSet([rtx,igpu]);
  assert.deepEqual(set.map(g=>g.id),['GPU-5070']);
  assert.equal(gpuCapacity(set),11.9);
  // the native engine report numbers each make from 0, so both cards say index 0: the ids stay different
  assert.notEqual(gpuId({...rtx,id:undefined}),gpuId({...igpu,id:undefined,index:0}));
  // alone, or a real AMD card beside it, nothing is dropped
  assert.deepEqual(defaultGpuSet([igpu]).map(g=>g.id),[igpu.id]);
  assert.equal(defaultGpuSet([{...igpu,vramGb:96,unified:true},rtx]).length,2);
});

// ---- 0.2.8: Monitor readings for AMD cards (#3727) ----
const amd=(id,index)=>({id,index,unified:false,vramGb:16,name:'AMD Radeon RX 9070 XT',vendor:'amd',backend:'vulkan'});
const ownReading=(id,extra={})=>({id,source:'sensor',sampledAt:'t0',temperatureC:null,utilizationPct:37,memoryUsedGb:5.5,powerW:null,powerLimitW:null,pcieGen:null,pcieGenMax:null,pcieWidth:null,pcieRxMbs:null,...extra});
const shell=(own)=>({context:{maxTokens:32768},hardware:{gpus:own,gpuLoad:null},history:[],totals:{},telemetry:{}});
const rawNoGpu={engine:{max_context:32768},live:{state:'generating'},hardware:{cpu:12},history:{}};

test('Windows AMD, one card, engine running: the engine has no GPU reader, the host load and memory stay (#3727)',()=>{
  const mapped=originalDashboard(rawNoGpu,{gpus:[amd('dx:1',0)]},shell([ownReading('dx:1')]));
  assert.equal(mapped.hardware.gpus.length,1);
  assert.equal(mapped.hardware.gpus[0].utilizationPct,37);
  assert.equal(mapped.hardware.gpus[0].memoryUsedGb,5.5);
  assert.equal(mapped.hardware.gpus[0].source,'sensor');
  assert.equal(mapped.hardware.gpus[0].temperatureC,null,'a value nobody read stays unavailable');
});
test('engine numbers win, the host fills only the gaps (fan and clock come from the host)',()=>{
  const raw={...rawNoGpu,hardware:{gpu_util:80,gpu_mem_used:4*1024**3,gpu_temp:61,gpu_power:200}};
  const mapped=originalDashboard(raw,{gpus:[amd('a',0)]},shell([ownReading('a',{utilizationPct:10,temperatureC:50,fanPct:35,clockMhz:2400,powerLimitW:300})]));
  const g=mapped.hardware.gpus[0];
  assert.equal(g.utilizationPct,80); assert.equal(g.temperatureC,61); assert.equal(g.powerW,200);
  assert.equal(g.powerLimitW,300); assert.equal(g.fanPct,35); assert.equal(g.clockMhz,2400); assert.equal(g.source,'engine');
});
test('a card the host has no reading for keeps the engine reading unchanged',()=>{
  const e={id:'x',source:'engine',temperatureC:null,utilizationPct:5};
  assert.equal(mergeOwnReading(e,undefined),e);
});
test('the headline aggregates fall back to the host when the engine reports none',()=>{
  const fb=shell([ownReading('a')]); fb.hardware.gpuTempC=55; fb.hardware.vramUsedGb=5.5;
  const mapped=originalDashboard(rawNoGpu,{gpus:[amd('a',0)]},fb);
  assert.equal(mapped.hardware.gpuTempC,55); assert.equal(mapped.hardware.vramUsedGb,5.5);
});
test('amdgpu temperature: edge first (what nvtop shows), junction when there is no edge, 0 is no reading',()=>{
  assert.equal(pickGpuTemperature([{label:'edge',milli:36000},{label:'junction',milli:38000},{label:'mem',milli:56000}]),36);
  assert.equal(pickGpuTemperature([{label:'mem',milli:56000},{label:'junction',milli:48000}]),48);
  assert.equal(pickGpuTemperature([{label:'',milli:41000}]),41);
  assert.equal(pickGpuTemperature([{label:'edge',milli:0},{label:'junction',milli:44000}]),44);
  assert.equal(pickGpuTemperature([{label:'edge',milli:0}]),null);
  assert.equal(pickGpuTemperature([{label:'mem',milli:56000}]),null,'a memory sensor alone is not the GPU temperature');
  assert.equal(pickGpuTemperature([]),null);
});
test('a remote-console chip with no memory is not a GPU; an unnamed card with memory and an Intel iGPU are',()=>{
  assert.equal(isComputeAdapter('unknown',null),false);
  assert.equal(isComputeAdapter('unknown',0),false);
  assert.equal(isComputeAdapter('unknown',8*1024**3),true);
  assert.equal(isComputeAdapter('intel',null),true);
  assert.equal(isComputeAdapter('amd',34208743424),true);
});
