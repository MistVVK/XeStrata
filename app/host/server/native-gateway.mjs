/** Preserve native protocol events and tool payloads; the gateway owns identity and cancellation. */
import { once } from 'node:events';
/** FIX-APP A4: more than one model is loaded and the request named none of them: say which names work (the same IDs /v1/models lists). The engine's own sentence stays when there is nothing to list. */
export function unknownModelMessage(error,asked,control){
 let ids=[];try{ids=(control.state?.().instances??[]).filter(i=>i.status==='ready').map(i=>i.apiModelId).filter(Boolean);}catch{ids=[];}
 if(ids.length<2||typeof asked!=='string'||!asked||error.message!=='Choose a loaded model ID.')return error.message;
 return `The model "${asked.slice(0,80)}" is not loaded. Loaded models: ${ids.join(', ')}. Use one of these names as "model".`;
}
/* #5661: mcpAllowed(model) says whether this request may use the model's MCP tools (serving.mjs mcpToolsAllowed); without it they stay off. */
export async function nativeGateway({req,res,path,body,control,serving,fetcher=fetch,mcpAllowed=()=>false}){
 let instance;try{instance=control.resolveInstance(body.model);}catch(e){/* #5312: an Anthropic client sends a model name of its own (Claude Desktop: claude-5-5-sonnet) and cannot change it: with exactly one model loaded the route answers with it.
   FIX-APP A4 (0.2.16 test round): the same for chat completions and responses, as `strata serve` has always done ("Other names are still served"): an app that sends "gpt-4o" or "x" to the one
   loaded model got HTTP 400 here but works against the Terminal's server. With two or more loaded, an unknown name stays a 400, and it names the loaded IDs. */
  if(body.model){try{instance=control.resolveInstance(undefined);}catch{}}
  if(!instance){res.writeHead(400,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:unknownModelMessage(e,body.model,control),type:'invalid_request_error',param:'model',code:null}}));return;}}
 const abort=new AbortController(),started=Date.now();let sawDone=false;/* #4690: a client that hangs up after `data: [DONE]` (the OpenAI SDK does) has the whole answer: that is not a cancel. */res.on('close',()=>{if(!res.writableEnded&&!sawDone)abort.abort();});
 const entry=serving.open({client:req.headers['user-agent']?.slice(0,100)??'API client',endpoint:path.slice(3),model:instance.apiModelId,instanceId:instance.id,gpuIds:instance.gpus.map(String),cancel:()=>abort.abort()});
 let tokensIn=null,tokensOut=null,tps=null,promptTps=null;
 const observe=value=>{/* Anthropic's stream carries the prompt count in message_start (message.usage) and only the output count in message_delta (usage): both are read (#4678). */for(const usage of [value.message?.usage,value.usage??value.response?.usage])if(usage){tokensIn=usage.prompt_tokens??usage.input_tokens??tokensIn;tokensOut=usage.completion_tokens??usage.output_tokens??tokensOut;}if(value.timings){tps=value.timings.predicted_per_second??tps;promptTps=value.timings.prompt_per_second??promptTps;}};
 try{
  const response=await fetcher(`http://127.0.0.1:${instance.port}${path}`,{method:'POST',headers:{'content-type':'application/json',authorization:`Bearer ${instance.key}`},signal:abort.signal,body:JSON.stringify({...body,model:instance.apiModelId,strata_mcp:body.strata_mcp===true&&mcpAllowed(instance.apiModelId)===true})});
  if(!response.ok){const content=await response.text();serving.finish(entry.id,{durationMs:Date.now()-started,error:`Engine returned ${response.status}`});res.writeHead(response.status,{'content-type':response.headers.get('content-type')??'application/json'});res.end(content);return;}
  serving.mark(entry.id,{state:'generating'});
  res.writeHead(response.status,{'content-type':response.headers.get('content-type')??'application/json','cache-control':'no-cache','x-strata-request-id':entry.id,'x-accel-buffering':'no'});
  if(body.stream===true){let buffer='';const decoder=new TextDecoder();for await(const chunk of response.body){buffer+=decoder.decode(chunk,{stream:true});let end;while((end=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,end);buffer=buffer.slice(end+1);if(line.startsWith('data:')){if(line.slice(5).trim()==='[DONE]')sawDone=true;else try{observe(JSON.parse(line.slice(5)));}catch{}}}if(!res.destroyed&&!res.write(chunk))await once(res,'drain',{signal:abort.signal});}}
  else{const text=await response.text();try{observe(JSON.parse(text));}catch{}res.write(text);}
  serving.mark(entry.id,{tokensIn,tokensOut,tps,promptTps,usageSource:tokensIn===null||tokensOut===null?'unavailable':'reported'});serving.finish(entry.id,{tokensOut,durationMs:Date.now()-started});res.end();
 }catch(e){serving.finish(entry.id,{tokensOut,durationMs:Date.now()-started,error:abort.signal.aborted?'Request cancelled.':e.message});if(!res.headersSent){res.writeHead(502,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:e.message,type:'server_error'}}));}else res.end();}
}
