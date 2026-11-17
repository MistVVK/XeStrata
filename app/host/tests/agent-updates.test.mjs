import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/agents.ts', import.meta.url),'utf8'), {loader:'ts',format:'esm'});
const { updateAgent, runningAgents, settleAgents } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('partial agent updates retain identity and keep elapsed time moving',()=>{
  let agents=updateAgent([],{id:'a',name:'Review',status:'working',tokens:20,durationMs:1000},5000);
  agents=updateAgent(agents,{id:'a',tokens:30},7000);
  assert.equal(agents.length,1);
  assert.equal(agents[0].name,'Review');
  assert.equal(agents[0].durationMs,3000);
  agents=updateAgent(agents,{id:'a',status:'completed'},8000);
  assert.equal(agents[0].durationMs,4000);
  assert.equal(runningAgents(agents),0);
});
test('queued and finished agents do not inflate the running count',()=>{
  let agents=[];
  for(const [id,status] of [['a','working'],['b','queued'],['c','failed'],['d','completed']])agents=updateAgent(agents,{id,status},5000);
  assert.equal(runningAgents(agents),1);
  assert.equal(runningAgents(updateAgent(agents,{id:'b',status:'working'},6000)),2);
});
test('late and malformed frames cannot overwrite newer agent readings',()=>{
  const agents=updateAgent([],{id:'a',status:'completed',durationMs:5000,tokens:42,updatedAt:6000},6000);
  assert.equal(updateAgent(agents,{id:'a',status:'working',updatedAt:4000},7000),agents);
  assert.equal(updateAgent(agents,{name:'Missing identity'},7000),agents);
  const next=updateAgent(agents,{id:'a',durationMs:-1,tokens:NaN,status:'unknown'},7000);
  assert.equal(next[0].durationMs,5000);
  assert.equal(next[0].tokens,42);
  assert.equal(next[0].status,'completed');
});

test('a saved helper row that says working is stopped after a reload, and a live frame brings it back (#4218)',()=>{
  const saved=[{id:'a',name:'x',description:'',status:'working',durationMs:5,tokens:1,updatedAt:1000},{id:'b',name:'y',description:'',status:'completed',durationMs:5,tokens:1,updatedAt:1000}];
  const settled=settleAgents(saved);
  assert.deepEqual(settled.map(a=>a.status),['cancelled','completed']);
  assert.equal(runningAgents(settled),0);
  assert.equal(saved[0].status,'working','the saved list is not changed');
  assert.equal(settleAgents(undefined),undefined);
  assert.equal(updateAgent(settled,{id:'a',status:'working',updatedAt:2000})[0].status,'working');
});
