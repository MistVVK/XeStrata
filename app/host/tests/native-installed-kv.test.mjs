/** Saved settings projected by the packaged native Control boundary, without starting an engine. */
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {EventEmitter} from 'node:events';
import {mkdtempSync,mkdirSync,writeFileSync,readFileSync,readdirSync,rmSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createNativeControl} from '../server/native-control.mjs';
import {nativeFixture} from './helpers/native-fixture.mjs';

async function installedKv(config,{family='swift',listed=true}={}){
 const root=mkdtempSync(join(tmpdir(),'strata-installed-kv-')),install=join(root,'install'),data=join(root,'host'),calls=[];
 for(const dir of [install,data])mkdirSync(dir,{recursive:true});
 const path=join(install,'strata-swift-iq2_xs.json'),original=JSON.stringify(config);
 writeFileSync(path,original);
 writeFileSync(join(data,'native-host.json'),JSON.stringify({installDir:install}));
 const control={control:{dir:join(root,'control'),python:join(root,'control','python')},version:'fixture',
  async installedModels({installDir}){assert.equal(installDir,install);return {install_dir:install,models:[{id:'swift-iq2_xs',name:'Swift IQ2_XS',source_config:path,context_tokens:65536,backend:'cuda',component_status:'present'}]};},
  async engine(){return {usable:true,backend:'cuda',engine:{version:'0.1.57',platform:'windows-x86_64',backend:'cuda',root:join(root,'engine')}};}};
 const spawnProcess=()=>{
  const child=new EventEmitter();child.stdout=new EventEmitter();child.stderr=new EventEmitter();child.exitCode=null;
  child.kill=()=>{child.exitCode=0;queueMicrotask(()=>child.emit('exit',0));return true;};
  child.stdin={write(line){const m=JSON.parse(line);if(!m.id)return;calls.push(m);
   const result=m.method==='initialize'?{}:m.method==='tools/list'?{tools:['strata_models','strata_status'].map(name=>({name}))}:
    {structuredContent:m.params.name==='strata_models'?{families:listed?[{family,sizes:[{id:'swift-iq2_xs',model:'IQ2_XS'}]}]:[]}:{hardware:{ram_gb:64,gpus:[]}}};
   queueMicrotask(()=>child.stdout.emit('data',Buffer.from(JSON.stringify({id:m.id,result})+'\n')));
  }};return child;
 };
 const native=createNativeControl({directory:data,control,spawnProcess,modelsDirectory:()=>join(root,'models'),hardwareProbe:()=>[],fetcher:()=>{throw new Error('Metadata must not contact an engine.');}});
 try{
  const state=await native.refresh(true);
  assert.equal(state.available,true,state.error);
  assert.equal(state.packaged,true);
  assert.deepEqual(state.instances,[]);assert.deepEqual(state.jobs,[]);
  assert.deepEqual(calls.filter(c=>c.method==='tools/call').map(c=>c.params.name),['strata_status','strata_models']);
  assert.equal(readFileSync(path,'utf8'),original,'metadata reads preserve the saved config');
  assert.deepEqual(readdirSync(install),['strata-swift-iq2_xs.json'],'no config backup or rewrite');
  return JSON.parse(JSON.stringify(state.installed.models[0]));
 }finally{await native.close();rmSync(root,{recursive:true,force:true});}
}

for(const kv of ['int8','q4_0','k8v4','fp16'])test(`packaged metadata restores saved ${kv} from its run config`,async()=>{
 const model=await installedKv({args:['--kv',kv]});
 assert.equal(model.kv,kv);assert.equal(model.ready,true);assert.equal(model.context,65536);
});

test('user_args overrides generated KV; the last repeated flag wins in either layer',async()=>{
 for(const config of [
  {args:['--kv','int8'],user_args:['--kv','k8v4']},
  {args:['--kv','int8','--kv','k8v4']},
  {args:['--kv','fp16'],user_args:['--kv','q4_0','--kv','k8v4']},
  {args:['--kv','q4_0'],args_written:['--kv','int8'],user_args:['--kv','k8v4']},
  {args:[],user_args:['--kv','k8v4']},
 ])assert.equal((await installedKv(config)).kv,'k8v4');
});

test('removing a written KV flag clears its user override as it does in a Host load',async()=>{
 const model=await installedKv({args:[],args_written:['--kv','int8'],user_args:['--kv','k8v4']});
 assert.equal(Object.hasOwn(model,'kv'),false);
});

test('absent, malformed or unknown saved KV stays absent without aliases or a default',async()=>{
 for(const config of [
  {},{kv:'k8v4',args:[]},{cache_type_k:'q8_0',cache_type_v:'q4_0',args:[]},
  {args:['--kv']},{args:['--kv','']},{args:['--kv',null]},{args:['--kv',{}]},
  {args:['--kv',['k8v4']]},{args:[],user_args:['--kv',['k8v4']]},
  {args:['--kv','q8_0']},{args:['--kv','K8V4']},{args:['--kv','future-format']},
  {args:['--kv','int8','--kv','unknown']},{args:['--kv','int8','--kv']},
  {args:['--kv','int8'],user_args:['--kv','unknown']},
  {args:['--kv','int8'],user_args:['--kv']},
 ])assert.equal(Object.hasOwn(await installedKv(config),'kv'),false,JSON.stringify(config));
});

test('ordinary KV metadata requires a matching Qwen3.8 catalog family',async()=>{
 for(const family of ['qwen','swift','coder','unsloth','gyro'])assert.equal((await installedKv({args:['--kv','k8v4']},{family})).kv,'k8v4');
 for(const family of ['qwen36','deepseek41','nvfp4','unknown'])assert.equal(Object.hasOwn(await installedKv({args:['--kv','k8v4']},{family}),'kv'),false);
 assert.equal(Object.hasOwn(await installedKv({args:['--kv','k8v4']},{listed:false}),'kv'),false);
 for(const arch of ['qwen35moe','deepseek41'])assert.equal(Object.hasOwn(await installedKv({arch,args:['--kv','k8v4']}),'kv'),false);
});

test('source-checkout metadata keeps the installer-provided KV value',async()=>{
 const f=nativeFixture();try{
  const path=join(f.repo,'strata-iq2_xs.json'),config={args:['--kv','k8v4']},original=JSON.stringify(config);
  writeFileSync(path,original);
  assert.equal((await f.control.refresh(true)).installed.models[0].kv,'int8');
  assert.equal(readFileSync(path,'utf8'),original);
 }finally{await f.cleanup();}
});
