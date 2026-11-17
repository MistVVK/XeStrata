import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {nativeGateway} from '../server/native-gateway.mjs';
const listen=server=>new Promise(r=>server.listen(0,'127.0.0.1',()=>r(server.address().port)));
const close=server=>new Promise(r=>{server.closeAllConnections();server.close(r);});
test('native gateway preserves all three protocols and routes exact IDs, usage, errors and cancellation',{timeout:10000},async()=>{
 const received=[],records=[];let cancelled=false;
 const engine=http.createServer(async(req,res)=>{let text='';for await(const chunk of req)text+=chunk;const body=JSON.parse(text);received.push({path:req.url,body,key:req.headers.authorization});
  if(body.input==='fail'){res.writeHead(422,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:'Invalid native input'}}));return;}
  if(body.stream){res.writeHead(200,{'content-type':'text/event-stream'});res.write('event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hello"}\n\n');if(body.input==='slow'){res.on('close',()=>{cancelled=true;});return;}res.end('event: response.completed\ndata: {"response":{"usage":{"input_tokens":12,"output_tokens":3}}}\n\n');return;}
  res.writeHead(200,{'content-type':'application/json'});res.end(JSON.stringify(req.url==='/v1/messages'?{type:'message',content:[{type:'text',text:'Hello'}],usage:{input_tokens:12,output_tokens:3}}:{object:'response',output:[{type:'function_call',name:'lookup',arguments:'{}'}],usage:{input_tokens:12,output_tokens:3}}));
 });const port=await listen(engine);
 const instances=[{id:'first',apiModelId:'native-a',model:'iq2_xs',port,key:'private-a',gpus:[0]},{id:'second',apiModelId:'native-b',model:'iq3_s',port,key:'private-b',gpus:[1]}];
 const serving={open:value=>{const record={...value,id:String(records.length)};records.push(record);return record;},mark:(id,value)=>Object.assign(records[Number(id)],value),finish:(id,value)=>Object.assign(records[Number(id)],value)};
 const control={resolveInstance:id=>{const found=instances.find(i=>i.apiModelId===id);if(!found)throw new Error('Choose an exact loaded ID');return found;}};
 const gateway=http.createServer(async(req,res)=>{let text='';for await(const chunk of req)text+=chunk;await nativeGateway({req,res,path:req.url,body:JSON.parse(text),control,serving});});const publicPort=await listen(gateway);
 const call=async(path,body)=>fetch(`http://127.0.0.1:${publicPort}${path}`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
 try{
  const tools=[{type:'function',name:'lookup',parameters:{type:'object'}}];const input=[{type:'message',role:'user',content:[{type:'input_text',text:'Hello'}]},{type:'function_call_output',call_id:'c1',output:'Found'}];
  let result=await call('/v1/responses',{model:'native-b',input,tools,strata_mcp:true,reasoning:{effort:'high'}});assert.equal(result.status,200);assert.equal((await result.json()).output[0].name,'lookup');assert.deepEqual(received[0].body.input,input);assert.deepEqual(received[0].body.tools,tools);assert.equal(received[0].body.model,'native-b');assert.equal(received[0].body.strata_mcp,false);assert.equal(received[0].key,'Bearer private-b');assert.equal(records[0].tokensIn,12);assert.equal(records[0].tokensOut,3);
  result=await call('/v1/messages',{model:'native-a',messages:[{role:'user',content:'Hello'}],max_tokens:50});assert.equal((await result.json()).type,'message');assert.equal(received[1].path,'/v1/messages');assert.equal(received[1].key,'Bearer private-a');
  result=await call('/v1/chat/completions',{model:'native-a',messages:[{role:'user',content:'Hello'}]});assert.equal(result.status,200);await result.text();assert.equal(received[2].path,'/v1/chat/completions');
  result=await call('/v1/responses',{model:'native-b',stream:true,input:'Hello'});assert.match(await result.text(),/event: response.completed/);assert.equal(records[3].tokensOut,3);
  result=await call('/v1/responses',{model:'missing',input:'Hello'});assert.equal(result.status,400);assert.equal((await result.json()).error.param,'model');
  result=await call('/v1/responses',{model:'native-b',input:'fail'});assert.equal(result.status,422);assert.equal((await result.json()).error.message,'Invalid native input');
  result=await call('/v1/responses',{model:'native-a',input:'slow',stream:true});const read=result.body.getReader();await read.read();await read.cancel();for(let n=0;n<30&&!cancelled;n++)await new Promise(r=>setTimeout(r,20));assert.equal(cancelled,true);
 }finally{await close(gateway);await close(engine);}
});
test('a client that hangs up right after data: [DONE] has its answer: the request is done, not "Request cancelled." (#4690)',{timeout:30000},async()=>{
 const net=await import('node:net');const records=[];let closeDelay=0;
 const engine=http.createServer(async(req,res)=>{for await(const _ of req);res.writeHead(200,{'content-type':'text/event-stream',connection:'close'});
  res.write('data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n');res.write('data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3}}\n\n');res.write('data: [DONE]\n\n');
  await new Promise(r=>setTimeout(r,closeDelay));res.end();});
 const enginePort=await listen(engine);
 const serving={open:value=>{const record={id:records.length,state:'queued',tokensIn:null,tokensOut:null,...value};records.push(record);return record;},mark:(id,value)=>Object.assign(records[id],value),finish:(id,{tokensOut,error})=>Object.assign(records[id],{state:error?'error':'done',tokensOut,error})};
 const control={resolveInstance:()=>({id:'x',apiModelId:'x',port:enginePort,key:'k',gpus:[0]})};
 const gateway=http.createServer(async(req,res)=>{let text='';for await(const chunk of req)text+=chunk;await nativeGateway({req,res,path:req.url,body:JSON.parse(text),control,serving});});
 const gatewayPort=await listen(gateway);
 const call=()=>new Promise(resolve=>{const body=JSON.stringify({model:'x',stream:true,messages:[]});
  const socket=net.connect(gatewayPort,'127.0.0.1',()=>socket.write(`POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`));
  let text='';socket.on('data',data=>{text+=data;if(text.includes('data: [DONE]'))socket.destroy();});socket.on('close',()=>resolve(text));});
 try{
  for(const delay of [0,5,30])for(let run=0;run<5;run++){closeDelay=delay;const before=records.length;assert.match(await call(),/\[DONE\]/);
   for(let n=0;n<100&&!['done','error'].includes(records[before]?.state);n++)await new Promise(r=>setTimeout(r,20));
   const last=records[before];assert.equal(last.state,'done',`delay ${delay} ms: ${last.error}`);assert.equal(last.tokensIn,12);assert.equal(last.tokensOut,3);}
 }finally{await close(gateway);await close(engine);}
});

test('an Anthropic stream: the prompt count (message_start) and the output count (message_delta) both reach the request log (#4678)',{timeout:10000},async()=>{
 const records=[];
 const engine=http.createServer(async(req,res)=>{for await(const _ of req);res.writeHead(200,{'content-type':'text/event-stream'});
  res.write('event: message_start\ndata: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":29757,"output_tokens":0}}}\n\n');
  res.write('event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":561}}\n\n');res.end();});
 const enginePort=await listen(engine);
 const serving={open:value=>{const record={id:records.length,state:'queued',tokensIn:null,tokensOut:null,...value};records.push(record);return record;},mark:(id,value)=>Object.assign(records[id],value),finish:(id,{tokensOut})=>Object.assign(records[id],{state:'done',tokensOut:tokensOut??records[id].tokensOut})};
 const control={resolveInstance:()=>({id:'x',apiModelId:'x',port:enginePort,key:'k',gpus:[0]})};
 const gateway=http.createServer(async(req,res)=>{let text='';for await(const chunk of req)text+=chunk;await nativeGateway({req,res,path:req.url,body:JSON.parse(text),control,serving});});
 const gatewayPort=await listen(gateway);
 try{
  const r=await fetch(`http://127.0.0.1:${gatewayPort}/v1/messages`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({model:'x',stream:true,max_tokens:50,messages:[]})});await r.text();
  assert.equal(records[0].tokensIn,29757);assert.equal(records[0].tokensOut,561);assert.equal(records[0].usageSource,'reported');
 }finally{await close(gateway);await close(engine);}
});

test('an Anthropic client that sends a model name of its own gets the one loaded model (#5312); OpenAI and Responses stay exact',{timeout:10000},async()=>{
 const received=[];const engine=http.createServer(async(req,res)=>{let text='';for await(const chunk of req)text+=chunk;received.push(JSON.parse(text));res.writeHead(200,{'content-type':'application/json'});res.end(JSON.stringify({type:'message',content:[{type:'text',text:'Hi'}],usage:{input_tokens:1,output_tokens:1}}));});
 const port=await listen(engine);const records=[];
 const serving={open:value=>{const r={...value,id:String(records.length)};records.push(r);return r;},mark:(id,value)=>Object.assign(records[Number(id)],value),finish:(id,value)=>Object.assign(records[Number(id)],value)};
 const one={id:'only',apiModelId:'iq3_s',port,key:'k',gpus:[0]},two={id:'other',apiModelId:'iq2_xs',port,key:'k',gpus:[1]};
 const make=instances=>({resolveInstance:id=>{const found=instances.find(i=>i.apiModelId===id)??(!id&&instances.length===1?instances[0]:undefined);if(!found)throw new Error('Choose a loaded model ID.');return found;}});
 const serve=control=>http.createServer(async(req,res)=>{let text='';for await(const chunk of req)text+=chunk;await nativeGateway({req,res,path:req.url,body:JSON.parse(text),control,serving});});
 const single=serve(make([one])),pair=serve(make([one,two]));const p1=await listen(single),p2=await listen(pair);
 const call=(p,path,body)=>fetch(`http://127.0.0.1:${p}${path}`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
 try{
  let r=await call(p1,'/v1/messages',{model:'claude-5-5-sonnet',messages:[{role:'user',content:'Hi'}],max_tokens:5});assert.equal(r.status,200);await r.text();assert.equal(received.at(-1).model,'iq3_s');
  r=await call(p2,'/v1/messages',{model:'claude-5-5-sonnet',messages:[],max_tokens:5});assert.equal(r.status,400);assert.equal((await r.json()).error.param,'model');
  r=await call(p1,'/v1/chat/completions',{model:'claude-5-5-sonnet',messages:[]});assert.equal(r.status,400);
  r=await call(p1,'/v1/responses',{model:'claude-5-5-sonnet',input:'Hi'});assert.equal(r.status,400);
 }finally{await close(single);await close(pair);await close(engine);}
});
