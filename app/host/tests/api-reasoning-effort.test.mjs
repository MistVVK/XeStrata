/** Behavioral checks use only disposable files and a loopback mock API. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,readFileSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createServer} from 'node:net';
import {createModelProviders} from '../server/model-providers.mjs';
import {createMockModelApi} from './mock-model-api.mjs';
import {createVisionProbe} from '../server/vision-probe.mjs';
import {ROOMY_HOST} from './helpers/hardware.mjs';
const choices={Off:'none',Low:'low',Medium:'medium',High:'high',Max:'high'};
const history=[{role:'user',content:'Earlier question'},{role:'assistant',content:'Earlier answer'}];
async function fixture(t,options={}){
 const mock=await createMockModelApi(),directory=mkdtempSync(join(tmpdir(),'strata-api-effort-'));
 t.after(async()=>{await mock.close();rmSync(directory,{recursive:true,force:true});});
 const probe=createVisionProbe(),service=createModelProviders({directory,knownThinks:(c,id)=>probe.knownThinks(c,id)});
 const connection=service.save({kind:'custom',name:'Fixture',baseUrl:mock.baseUrl,protocol:'openai-chat',apiKey:'fixture-key',models:[{id:'mock-reasoning',reasoning:true}],...options}).connections[0];
 return {mock,service,connection,probe,selection:{connectionId:connection.id,modelId:connection.models[0].id}};
}
test('saved reasoning models receive all official composer choices with the remaining request unchanged',async t=>{
 const f=await fixture(t);
 for(const [thinking,effort] of Object.entries(choices)){
  const frames=[];await f.service.stream({selection:f.selection,message:'Next question',history,systemPrompt:'Fixture instructions',thinking,thinkingBudget:1234,onFrame:f=>frames.push(f)});
  const request=f.mock.requests.at(-1);
  assert.equal(request.path,'/v1/chat/completions');assert.equal(request.headers.authorization,'Bearer fixture-key');
  assert.deepEqual(request.body,{model:'mock-reasoning',messages:[{role:'system',content:'Fixture instructions'},...history,{role:'user',content:'Next question'}],stream:true,reasoning_effort:effort});
  assert.equal(frames.at(-1).type,'done');
 }
});
test('invalid or omitted efforts keep the provider default',async t=>{
 const f=await fixture(t);
 for(const thinking of [undefined,null,false,1,{},'','Default','Provider Default','ultra','none','__proto__']){
  await f.service.stream({selection:f.selection,message:'Test',thinking,onFrame:()=>{}});
  assert.equal(Object.hasOwn(f.mock.requests.at(-1).body,'reasoning_effort'),false);
 }
});

test('direct OpenRouter chat uses the same effort format and Max mapping as agent chat',async t=>{
 const f=await fixture(t,{kind:'provider',provider:'openrouter'});
 for(const [thinking,effort] of Object.entries({...choices,Max:'xhigh'})){
  await f.service.stream({selection:f.selection,message:'Test',thinking,onFrame:()=>{}});
  const payload=f.mock.requests.at(-1).body;
  assert.deepEqual(payload.reasoning,{effort});
  assert.equal(Object.hasOwn(payload,'reasoning_effort'),false);
 }
});
test('saved false and unknown support cannot be overridden by request capability fields',async t=>{
 for(const reasoning of [false,null]){
  const f=await fixture(t,{models:[{id:'mock-chat',reasoning}]});
  await f.service.stream({selection:{...f.selection,reasoning:true,thinks:true},message:'Test',thinking:'High',reasoning:true,thinks:true,onFrame:()=>{}});
  assert.equal(Object.hasOwn(f.mock.requests.at(-1).body,'reasoning_effort'),false);
 }
});
test('host-owned probe support is used only when the saved answer is unknown',async t=>{
 for(const reasoning of [null,false]){
  const f=await fixture(t,{models:[{id:'mock-reasoning',reasoning}]});
  await f.probe.check(f.service.resolve(f.selection).connection,'mock-reasoning');assert.equal(f.probe.knownThinks(f.connection,'mock-reasoning'),true);
  await f.service.stream({selection:f.selection,message:'Test',thinking:'Low',onFrame:()=>{}});
  assert.equal(f.mock.requests.at(-1).body.reasoning_effort,reasoning===false?undefined:'low');
 }
});
test('Responses and Anthropic retain their existing payloads even for reasoning models',async t=>{
 for(const protocol of ['openai-responses','anthropic']){
  const f=await fixture(t,{protocol});await f.service.stream({selection:f.selection,message:'Test',onFrame:()=>{}});const before=f.mock.requests.at(-1).body;
  await f.service.stream({selection:f.selection,message:'Test',thinking:'Max',thinkingBudget:1234,onFrame:()=>{}});assert.deepEqual(f.mock.requests.at(-1).body,before);
 }
});
async function freePort(){const s=createServer();await new Promise(r=>s.listen(0,'127.0.0.1',r));const p=s.address().port;await new Promise(r=>s.close(r));return p;}
test('host forwards incoming composer choices using saved and probed support without trusting client capabilities',{timeout:30000},async t=>{
 const mock=await createMockModelApi(),home=mkdtempSync(join(tmpdir(),'strata-api-effort-host-'));
 let child,exited;
 t.after(async()=>{if(child&&child.exitCode===null){child.kill();await exited;}await mock.close();rmSync(home,{recursive:true,force:true,maxRetries:3});});
 mkdirSync(join(home,'data'));writeFileSync(join(home,'spy.log'),'');
 // No picker/discovery call precedes the first chat: this also exercises cold probe routing.
 const models=[{id:'mock-chat',reasoning:true},{id:'mock-reasoning',reasoning:null},{id:'plain',reasoning:false},{id:'unknown',reasoning:null}];
 writeFileSync(join(home,'data','model-providers.json'),JSON.stringify([{id:'fixture',kind:'custom',name:'Fixture',baseUrl:mock.baseUrl,protocol:'openai-chat',apiKey:'fixture-key',models}]));
 const port=await freePort(),base=`http://127.0.0.1:${port}`;
 const env={...process.env,STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models'),STRATA_WORKSPACE:join(home,'work'),STRATA_SETTINGS_FILE:join(home,'settings.json'),STRATA_COMPONENTS_DIR:join(home,'components'),STRATA_CONTROL_DIR:join(home,'no-control'),STRATA_HARNESS_DIR:join(home,'no-harness'),STRATA_NATIVE_REPO:join(home,'no-repo'),STRATA_PORT:String(port),STRATA_SHARE_PORT:String(await freePort()),STRATA_TUNNEL_PORT:String(await freePort()),STRATA_BIND:'127.0.0.1',STRATA_REAL_ENGINE:'0',STRATA_TEST_HARDWARE:ROOMY_HOST,STRATA_AGENT:'off',STRATA_NO_ACQUIRE:'1',ACQUIRE_SPY_LOG:join(home,'spy.log')};
 child=spawn(process.execPath,['--import','./tests/helpers/acquire-spy.mjs','server/index.mjs'],{cwd:join(import.meta.dirname,'..'),windowsHide:true,stdio:'ignore',env});
 exited=once(child,'exit');
 let up=false;for(let n=0;n<100&&!up;n++){try{up=(await fetch(base+'/api/health')).ok;}catch{}if(!up)await new Promise(r=>setTimeout(r,100));}assert.ok(up,'fixture host started');
 const before=readFileSync(join(home,'data','model-providers.json'),'utf8');
 for(const model of models)for(const [thinking,effort] of Object.entries(choices)){
  const res=await fetch(base+'/api/chat/stream',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({message:'Next question',history,thinking,thinkingBudget:1234,agent:false,useTools:false,mode:'bypass',remoteModel:{connectionId:'fixture',modelId:model.id,reasoning:true,thinks:true}})});
  assert.equal(res.status,200);const frames=(await res.text()).trim().split('\n').map(JSON.parse);
  assert.equal(frames.some(f=>f.type==='error'),false,JSON.stringify(frames));assert.equal(frames.at(-1).type,'done');
  const request=mock.requests.filter(r=>r.path==='/v1/chat/completions').at(-1);
  assert.equal(request.body.model,model.id);assert.equal(request.headers.authorization,'Bearer fixture-key');assert.deepEqual(request.body.messages.slice(-3),[...history,{role:'user',content:'Next question'}]);
  assert.equal(request.body.reasoning_effort,['plain','unknown'].includes(model.id)?undefined:effort);assert.equal(Object.hasOwn(request.body,'reasoning_budget_tokens'),false);
 }
 assert.equal(readFileSync(join(home,'data','model-providers.json'),'utf8'),before);
 assert.equal(readFileSync(join(home,'spy.log'),'utf8'),'','no external requests or software acquisition');
});
