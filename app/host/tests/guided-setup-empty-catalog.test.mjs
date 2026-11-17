import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,readFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {createRequire} from 'node:module';
import {build} from 'esbuild';

/* The installed app crashed on its first screen ("Cannot read properties of undefined (reading 'family')") when the
   catalog it was given had no `recommendation` and no families. Render the real GuidedSetup and NativeModels components
   (react-dom/server) with the state libraries replaced by fixed values, for every shape of an empty or failed catalog. */
const root=resolve(import.meta.dirname,'..');
/** Every export of the real module becomes a harmless no-op, except the few that carry the fixed state. */
function stubFor(file,overrides){
 const source=readFileSync(resolve(root,'src/lib',file),'utf8');
 const names=new Set([...source.matchAll(/export\s+(?:async\s+)?(?:function\*?|const|let|var|class)\s+([A-Za-z_$][\w$]*)/g)].map(m=>m[1]));
 for(const m of source.matchAll(/export\s*\{([^}]*)\}/g))for(const part of m[1].split(','))names.add(part.split(/\s+as\s+/).pop().trim());
 names.delete('');
 const lines=[...names].filter(n=>!(n in overrides)).map(n=>`export const ${n}=(...a)=>Promise.resolve({});`);
 for(const [n,v] of Object.entries(overrides))lines.push(`export const ${n}=${v};`);
 return lines.join(String.fromCharCode(10));
}
const stubs={
 native:stubFor('native.ts',{useNative:'()=>globalThis.__native',nativeState:'()=>globalThis.__native'}),
 store:stubFor('store.ts',{useStore:'sel=>sel(globalThis.__store)'}),
 client:stubFor('client.ts',{}),
 router:stubFor('router.ts',{}),
 'kept-state':stubFor('kept-state.ts',{useKept:'(key,initial)=>[globalThis.__kept?.[key]??initial,()=>{}]'}),
};

let dir,mod;
async function load(){
 if(mod)return mod;
 dir=mkdtempSync(join(tmpdir(),'strata-gs-'));
 const stubPlugin={name:'stubs',setup(b){
  b.onResolve({filter:/\/lib\/(native|store|client|router|kept-state)$/},args=>({path:args.path.split('/').pop(),namespace:'stub'}));
  b.onLoad({filter:/.*/,namespace:'stub'},args=>({contents:stubs[args.path],loader:'js',resolveDir:root}));
 }};
 const entry=join(dir,'entry.tsx');
 const out=join(dir,'out.cjs');
 await build({
  stdin:{contents:`export {GuidedSetup} from ${JSON.stringify(resolve(root,'src/components/GuidedSetup.tsx'))};
   export {NativeModels} from ${JSON.stringify(resolve(root,'src/components/NativeModels.tsx'))};
   export {renderToString} from 'react-dom/server'; export {createElement} from 'react';`,resolveDir:root,sourcefile:'entry.tsx',loader:'tsx'},
  bundle:true,format:'cjs',platform:'node',outfile:out,jsx:'automatic',logLevel:'silent',plugins:[stubPlugin],
  loader:{'.svg':'dataurl','.png':'dataurl','.css':'empty'},
 });
 globalThis.window??=globalThis;globalThis.matchMedia??=()=>({matches:false,addEventListener(){},removeEventListener(){}});
 mod=createRequire(import.meta.url)(out);
 return mod;
}
test.after(()=>{if(dir)rmSync(dir,{recursive:true,force:true});});

const store=(complete=false)=>({setup:{complete,choices:{},app:{modelsDir:''}},hardware:null});
const base={available:true,repo:'',summary:'',error:null,installed:{models:[],data_folder:'',engine:{version:null}},hardware:null,instances:[],jobs:[],defaultId:null,fields:[],capabilities:{}};
const catalogs={
 'no catalog at all':null,
 'the packaged host before it had a catalog':{families:[],sizes:[]},
 'no recommendation':{families:[],contexts:[32768]},
 'recommendation with nothing in it':{families:[],contexts:[],recommendation:{family:null,model:null}},
 'no families key':{contexts:[]},
 'a family without sizes':{families:[{family:'qwen',title:'Qwen3.8-Flash-Next',sizes:[]}],contexts:[32768]},
};

for(const [name,catalog] of Object.entries(catalogs)){
 for(const component of ['GuidedSetup','NativeModels']){
  test(`${component} renders with ${name}`,async()=>{
   const m=await load();
   globalThis.__native={...base,catalog};globalThis.__store=store();
   let html;
   assert.doesNotThrow(()=>{html=m.renderToString(m.createElement(m[component]));});
   assert.equal(typeof html,'string');
  });
 }
}

test('GuidedSetup lists the real catalog when it has one',async()=>{
 const m=await load();
 globalThis.__native={...base,catalog:{families:[{family:'qwen',title:'Qwen3.8-Flash-Next',about:'',images:true,experimental:false,sizes:[
   {id:'iq3_xxs',model:'IQ3_XXS',download_gb:75.8,ram_gb:60,about:'',on_this_pc:'fits',installed:false}]}],contexts:[32768],
   recommendation:{family:'qwen',model:'IQ3_XXS',context:32768},context_note:''}};
 globalThis.__store={...store(),setup:{complete:false,choices:{onboarding:{step:2,manual:false}},app:{modelsDir:''}}};
 const html=m.renderToString(m.createElement(m.GuidedSetup));
 assert.match(html,/IQ3_XXS/);
});

const failedJob=(id,model,family)=>({id,kind:'install',model,family,state:'error',
 startedAt:'2026-01-01T00:00:00.000Z',summary:'Previous preparation failed',
 error:'Data_dir C:\\Users\\Example\\.strata\\models: a new data folder must be named Strata',result:null});
async function renderModelJobs(jobs,kept={}){
 const m=await load();
 const size=(id,model)=>({id,model,download_gb:50,ram_needed_gb:48,experts_gb:30,about:'',on_this_pc:'fits',installed:false});
 globalThis.__native={...base,jobs,catalog:{contexts:[32768,65536],families:[
  {family:'qwen',title:'Qwen3.8-Flash-Next',about:'',images:false,experimental:false,sizes:[size('iq3_s','IQ3_S')]},
  {family:'swift',title:'Qwen3.8-Flash-Next (Swift)',about:'',images:false,experimental:false,sizes:[size('swift-iq3_s','IQ3_S')]},
 ]}};
 globalThis.__store=store();globalThis.__kept=kept;
 try{return m.renderToString(m.createElement(m.NativeModels));}finally{globalThis.__kept={};}
}

test('NativeModels does not attach another family’s failed preparation to the selected version',async()=>{
 const job=failedJob('old-swift','IQ3_S','swift'),jobs=[job];
 assert.doesNotMatch(await renderModelJobs(jobs),/Data_dir/);
 assert.equal(jobs[0],job,'the job is retained for diagnostics');
});

test('NativeModels ignores a remembered job after selecting a different model',async()=>{
 assert.doesNotMatch(await renderModelJobs([failedJob('old-swift','swift-iq3_s','swift')],{'models.job':'old-swift'}),/Data_dir/);
});

test('NativeModels still shows the selected canonical model’s real failure',async()=>{
 assert.match(await renderModelJobs([failedJob('current','iq3_s')]),/Data_dir/);
});

test('NativeModels identifies aliases by source family and keeps matching remembered errors',async()=>{
 assert.match(await renderModelJobs([failedJob('current','IQ3_S','qwen')],{'models.job':'current'}),/Data_dir/);
 assert.doesNotMatch(await renderModelJobs([failedJob('ambiguous','IQ3_S')]),/Data_dir/);
});

test('NativeModels falls back to the latest job for the selected version, not an unrelated remembered job',async()=>{
 const own=failedJob('own','iq3_s'),other={...failedJob('other','swift-iq3_s','swift'),error:'Other model error'};
 const html=await renderModelJobs([own,other],{'models.job':'other'});
 assert.match(html,/Data_dir/);assert.doesNotMatch(html,/Other model error/);
});

/* The first-run model step starts on a version that is already on this computer, else on the catalog's recommendation, never on the first
   size in the list (it used to be filled once at mount, before the catalog had arrived: Q2_0, a 66 GB download, beside an installed IQ3_S). */
const sizeRow=(id,model)=>({id,model,download_gb:50,ram_needed_gb:48,experts_gb:30,about:'',on_this_pc:'fits',installed:false});
async function selectedStart(readyIds,recommendation='IQ3_S',onboarding={step:2,manual:false}){
 const m=await load();
 globalThis.__native={...base,installed:{...base.installed,models:readyIds.map(id=>({model:id,ready:true,context:16384}))},catalog:{contexts:[32768],
  families:[{family:'qwen',title:'Qwen3.8-Flash-Next',about:'',images:false,experimental:false,sizes:[sizeRow('q2_0','Q2_0'),sizeRow('iq3_s','IQ3_S')]},
  {family:'swift',title:'Qwen3.8-Flash-Next (Swift)',about:'',images:false,experimental:false,sizes:[sizeRow('swift-iq2_xs','IQ2_XS')]}],
  recommendation:{family:'qwen',model:recommendation,context:32768}}};
 globalThis.__store={...store(),setup:{complete:false,choices:{onboarding},app:{modelsDir:''}}};
 const html=m.renderToString(m.createElement(m.GuidedSetup));
 return [...html.matchAll(/<button[^>]*role="radio"[^>]*aria-checked="true"[^>]*>(.*?)<\/button>/gs)].map(x=>x[1].replace(/<[^>]*>/g,' ').replace(/\s+/g,' ').trim());
}
test('GuidedSetup starts on the recommended version when nothing is installed, not on the first size',async()=>{
 const chosen=await selectedStart([]);
 assert.equal(chosen.length,1);assert.match(chosen[0],/IQ3_S/);assert.doesNotMatch(chosen[0],/Q2_0/);
});
test('GuidedSetup starts on a version that is installed and ready, even when the recommendation is another',async()=>{
 const chosen=await selectedStart(['swift-iq2_xs']);
 assert.equal(chosen.length,1);assert.match(chosen[0],/IQ2_XS/);
});
test('GuidedSetup prefers the recommended version when it is among the installed ones',async()=>{
 const chosen=await selectedStart(['q2_0','iq3_s']);
 assert.equal(chosen.length,1);assert.match(chosen[0],/IQ3_S/);
});
test('GuidedSetup keeps a version the person already chose over the computed start',async()=>{
 const chosen=await selectedStart(['iq3_s'],'IQ3_S',{step:2,manual:false,family:'qwen3.8',size:'swift-iq2_xs'});
 assert.equal(chosen.length,1);assert.match(chosen[0],/IQ2_XS/);
});
