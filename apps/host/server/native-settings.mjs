import { validMcpName, MCP_NAME_HELP } from './mcp-names.mjs';
import { SPEED_SWITCH_TABLE, DRAFTING_FAMILIES as GENERATED_DRAFTING_FAMILIES, DFLASH2_MODEL_PATTERN } from './policy-generated.mjs';
/** Editable native keys. Unknown config keys and program paths stay intact. */
/* The Drafting hint, word for word, with the size of the drafter this model would really fetch (the tuned one is 0.34 GB, 0.52 on an Arc card; the stock one 1.2). */
export function draftingHint(gb=1.2){const size=Number.isFinite(Number(gb))&&Number(gb)>0?Number(gb):1.2;return `Which drafter guesses words ahead to make answers faster, one at a time. Auto (recommended) is DFlash2 for Qwen3.8-Flash-Next where the engine has it, the MTP otherwise; on two or more identical graphics cards that the all-cards split engine takes, it is that engine with the MTP draft layer. DFlash2 downloads a ${size} GB drafter once, the MTP its draft layer once. With Auto, several requests at once use the MTP. If you choose DFlash2 yourself, it runs with the number of requests you set. If DFlash2 cannot run, the MTP is used alone and Details says why. Off turns the drafter off.`;}
export const NATIVE_FIELDS=[
 {id:'reasoning_budget_tokens',label:'Thinking token budget',group:'Responses',type:'number',min:0,max:1048576,hint:'Leave blank for the model default.'},
 {id:'fit_max_tokens',label:'Fit output to remaining context',group:'Responses',type:'boolean',hint:'Shortens the requested output cap when the prompt leaves less room.'},
 {id:'anthropic_thinking',label:'Anthropic thinking',group:'Responses',type:'enum',choices:['model','on_request'],hint:'Follow the model default or think only when requested.'},
 {id:'effort_position',label:'Reasoning effort position',group:'Responses',type:'enum',choices:['start','end']},
 {id:'aliases',label:'API model aliases',group:'Serving',type:'array',hint:'Comma-separated names clients may use for this model.'},
 {id:'idle_unload_s',label:'Unload after idle seconds',group:'Serving',type:'number',min:0,max:86400,hint:'Zero keeps the model loaded.'},
 {id:'lazy_load',label:'Load on first request',group:'Serving',type:'boolean'},
 {id:'mtp',label:'Draft layer (MTP)',group:'Serving',type:'boolean',hint:'The draft layer guesses several words ahead, which makes answers faster, and costs about 1 GB of graphics memory and some RAM. Turn it off if the model stops while it starts, or the card is small: the expert cache gets that memory back and answers are slower. Default is on.'},
 {id:'drafting',label:'Drafting',group:'Serving',type:'enum',choices:['auto','mtp','dflash2','off'],hint:draftingHint(1.2)},
 {id:'batchMtp',label:'MTP per request',group:'Serving',type:'boolean',hint:'Each concurrent request uses the draft layer. Left alone, Strata turns it on for one graphics card with 24 GB or more and several requests at once, and off otherwise; choose on or off here to decide yourself.'},
 {id:'parallel',label:'Concurrent requests',group:'Serving',type:'number',min:1,max:8,hint:'More slots need more memory and can slow each response.'},
 {id:'engine_silence_s',label:'Engine stall timeout (seconds)',group:'Recovery',type:'number',min:10,max:3600},
 {id:'api_monitor',label:'Keep API request metadata',group:'Recovery',type:'boolean'},
 {id:'layer_split',label:'Layer split',group:'GPU & memory',type:'string',hint:'auto, or increasing split points such as 18 or 16,32.'},
 {id:'gpu_order',label:'GPU order',group:'GPU & memory',type:'enum',choices:['auto','as_given']},
 {id:'split_skip_if_fits',label:'Use one GPU when the model fits',group:'GPU & memory',type:'boolean'},
 {id:'vram_elastic',label:'Elastic GPU expert cache',group:'GPU & memory',type:'boolean'},
 {id:'vram_reserve_mib',label:'GPU memory for other apps (MiB)',group:'GPU & memory',type:'number',min:0,max:131072,hint:'Applied as --vram-reserve-mib when loading.'},
 {id:'vision_tokens',label:'Maximum image tokens',group:'Image reading',type:'number',min:1,max:16384},
 {id:'vision_min_tokens',label:'Minimum image tokens',group:'Image reading',type:'number',min:0,max:16384},
 {id:'mcp_timeout_s',label:'Tool timeout (seconds)',group:'Tools',type:'number',min:1,max:3600},
 {id:'mcp_max_rounds',label:'Maximum tool rounds',group:'Tools',type:'number',min:1,max:100},
 {id:'mcp_max_result_chars',label:'Maximum tool result characters',group:'Tools',type:'number',min:100,max:1000000},
];
/** Engine environment names the app shows and edits: STRATA_*. Any other name in a model's file is kept untouched and not shown. */
export const ENV_FLAG=/^STRATA_[A-Z0-9_]+$/;
/**
 * The speed features of 0.2.7 that have a switch in Expert settings. A switch has a default ('on' or 'off'); outside it, it writes exactly the
 * environment entries `env` into the model's settings file ("env"); back in its default it removes them, so a default launch carries none of them.
 * "Faster long prompts" (Qwen KVA) is OFF by default (owner decision, 26 October): STRATA_KVA=1 turns it on, and an earlier test build's STRATA_KVA=0 reads as off.
 * "Faster answers on 12 GB NVIDIA cards" (residency-biased routing) is OFF by default since 0.2.8 (owner decision, 29 October; #4402): STRATA_ROUTE_PRIOR=1.5
 * turns it on, and the STRATA_ROUTE_PRIOR=0 that 0.2.7's Off wrote is the default state now. The other two are on by default. The names are the engine switch contract (phaseA-tests/notes-027-plan.md); strata_control/speed_switches.py says the
 * same for the Terminal (tests/terminal/test_speed_switches.py compares the two).
 */
/* The table (ids, words, defaults, environment entries) is generated from strata_control/speed_switches.py into policy-generated.mjs; this file only adds the group the screen lists them under and the rules for where a switch applies (below), which tests/speed-switches.test.mjs and tests/terminal/test_speed_switches.py run against the same cases. */
export const SPEED_SWITCHES=SPEED_SWITCH_TABLE.map(s=>({id:s.id,label:s.label,group:s.group??'Speed',hint:s.hint,default:s.default,env:s.env,...(s.legacy?{legacy:s.legacy}:{}),...(s.words?{words:s.words}:{})}));
const GFX12=/^gfx120[01]$/i, RADEON_9000=/\b(?:RX\s*9\d{3}|AI\s*PRO\s*R97\d{2}|R97\d{2})\b/i;
/**
 * Which switches make sense for this model on this PC: only those are shown. `backend` is the engine's (cuda, hip, sycl, metal, cpu). `arch` is the "arch"
 * of the model's run config: setup writes it only for the families with an engine path of their own (Qwen3.6-35B-A3B "qwen35moe");
 * the Qwen3.8-Flash-Next files (the original, Swift, Coder, Unsloth, Gyro, NVFP4) have none. KVA was measured on the Flash-Next shape (48 layers, 512
 * experts) and the engine declines it on any other, so "Faster long prompts" is shown for a model without an "arch" only.
 */
/** A discrete NVIDIA card of 13 GiB or less: the engine's rule for residency-biased routing (not an integrated chip, not a card that shares system memory). */
const smallNvidia=gpus=>(gpus??[]).some(g=>{
 if(g.usable===false||g.uma||g.integrated||(g.unified_gb??0)>0)return false;
 const vendor=String(g.vendor??'').toLowerCase();
 if(vendor!=='nvidia'&&!(vendor===''&&/nvidia|geforce|\brtx\b|\bgtx\b|quadro|tesla/i.test(String(g.name??''))))return false;
 return g.vram_gb>0&&g.vram_gb<=13;
});
export function speedSwitchApplies(id,{deepseek=false,backend=null,gpus=[],arch=null,vision=false}={}){
 if(deepseek)return false;                                              // a DeepSeek settings file (no longer run): no switch
 if(id==='picture_reader')return !!vision;                              // a Qwen model that reads pictures (its run config has a "vision" section)
 if(id==='faster_long_prompts')return !arch&&(backend==='cuda'||backend==='hip');
 if(id==='faster_12gb_nvidia')return !arch&&backend==='cuda'&&smallNvidia(gpus);   // the engine's routing needs the Flash-Next shape (512 experts, top-10) and a CUDA build
 if(id==='faster_radeon_9000')return backend==='hip'&&(gpus??[]).some(g=>(!g.vendor||g.vendor==='amd')&&(GFX12.test(String(g.arch??''))||(!g.arch&&RADEON_9000.test(String(g.name??'')))));
 return false;
}
/** The switch is outside its default when every entry it writes is in the environment with its value; anything less or other (a hand-edited file, STRATA_KVA=0) is the default state and is left alone. */
export function speedSwitchIsSet(env,id){const sw=SPEED_SWITCHES.find(s=>s.id===id),has=entries=>!!entries&&Object.keys(entries).length>0&&Object.entries(entries).every(([name,value])=>env?.[name]===value);return !!sw&&(has(sw.env)||has(sw.legacy));}
/** Is the feature on for a launch with this environment? */
export function speedSwitchIsOn(env,id){const sw=SPEED_SWITCHES.find(s=>s.id===id);return !!sw&&speedSwitchIsSet(env,id)!==(sw.default==='on');}
/** The environment with the feature on or off: its default removes the switch's entries, the other state writes them. Nothing else in it changes. */
export function setSpeedSwitch(env,id,on){
 const sw=SPEED_SWITCHES.find(s=>s.id===id);if(!sw)throw new Error(`Unknown speed switch: ${id}`);
 const write=on!==(sw.default==='on'),next={...(env??{})};for(const [name,value] of Object.entries(sw.env)){if(write)next[name]=value;else delete next[name];}if(!write)for(const name of Object.keys(sw.legacy??{}))delete next[name];return next;
}
/** What the settings answer carries: each switch that applies, with its words, its default, the entries it writes outside the default, and whether the feature is on. */
export function speedSwitchRows(env,context){return SPEED_SWITCHES.filter(s=>speedSwitchApplies(s.id,context)).map(s=>({...s,on:speedSwitchIsOn(env,s.id)}));}
/** Settings the app itself applies when it starts a model (they are not read from the engine's source, so the engine's text never says it supports them). */
export const APP_OWNED_FIELDS=['mtp','batchMtp','drafting'];
/** The families that have a Drafting choice and the model names the DFlash2 drafter was made for (strata_control/drafting.py SUPPORTED_FAMILIES, _SUPPORTED_NAME): Qwen3.8-Flash-Next, Gyro and Unsloth's quantizations of the same weights; not Swift, Coder or NVFP4. */
export const DRAFTING_FAMILIES=GENERATED_DRAFTING_FAMILIES;
const DFLASH2_MODEL=new RegExp(DFLASH2_MODEL_PATTERN,'i');
/** The drafter a choice means (one per run; strata_control/drafting.py choice_of): Auto is DFlash2 where the engine has it (`served`, Control's word) and the model is one it was made for, else the MTP; off and mtp are what they say. `model` is a run config's model name or a family. */
/** `split` (Control's plan: the all-cards split engine will take this start) makes Auto the MTP, the draft layer that engine runs; a DFlash2 the person chose by name stays. */
export function resolveDrafter(choice,model,served,split){
 const c=String(choice??'auto').toLowerCase();if(c==='off'||c==='mtp')return c;
 const name=String(model??'');if(!(served===true&&(DRAFTING_FAMILIES.includes(name.toLowerCase())||DFLASH2_MODEL.test(name))))return 'mtp';
 return split===true&&c!=='dflash2'?'mtp':'dflash2';
}
/** The engine arguments without the draft layer: --mtp <folder or count>, --spec-min-p <p> (its confidence threshold) and --batch-mtp. --spec stays: prompt lookup still uses its windows. Same rule as strata_control/draft_layer.py. */
export function withoutDraftLayer(args){
 const out=[];const list=(args??[]).map(String);
 for(let i=0;i<list.length;i++){const a=list[i];if(a==='--batch-mtp')continue;if(a==='--mtp'||a==='--spec-min-p'){if(i+1<list.length&&!list[i+1].startsWith('--'))i++;continue;}out.push(a);}
 return out;
}
export function validateNativeSettings(values){
 if(!values||typeof values!=='object'||Array.isArray(values))throw new Error('Native settings must be an object.');
 const result={};
 for(const [key,value] of Object.entries(values)){

  if(key==='args'){if(!Array.isArray(value)||value.length>200||value.some(v=>typeof v!=='string'||v.length>4000||/[\n\r\0]/.test(v))||value.some(v=>['--host','--port','--api-key'].includes(v)))throw new Error('Engine arguments must be text values. Network access stays managed by Strata.');result.args=[...value];continue;}
  if(key==='env'){if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Engine environment must be an object.');const env={};for(const [name,text] of Object.entries(value)){if(!ENV_FLAG.test(name)||typeof text!=='string'||text.length>4000||text.includes('\0'))throw new Error('Use STRATA_ environment flags with text values.');env[name]=text;}result.env=env;continue;}
  if(key==='sampling'){if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Sampling must be an object.');const limits={temperature:[0,10],top_p:[Number.MIN_VALUE,1],top_k:[1,64],min_p:[0,1],repetition_penalty:[Number.MIN_VALUE,10],presence_penalty:[0,10],frequency_penalty:[0,10],penalty_last_n:[0,1048576],seed:[1,2147483647]};for(const [name,n] of Object.entries(value)){if(name==='experimental_speed_projection'){if(typeof n!=='boolean')throw new Error('Speed projection must be on or off.');continue;}if(!limits[name]||!Number.isFinite(n)||(['top_k','seed','penalty_last_n'].includes(name)&&!Number.isInteger(n))||n<limits[name][0]||n>limits[name][1])throw new Error(`Choose a valid sampling ${name}.`);}result.sampling={...value};continue;}
  const field=NATIVE_FIELDS.find(f=>f.id===key);if(!field)throw new Error(`Unknown native setting: ${key}`);
  if(value===null||value===''){result[key]=null;continue;}
  if(field.type==='number'&&(!Number.isFinite(value)||value<field.min||value>field.max||!Number.isInteger(value)))throw new Error(`${field.label}: choose a whole number from ${field.min} to ${field.max}.`);
  if(field.type==='boolean'&&typeof value!=='boolean')throw new Error(`${field.label}: choose on or off.`);
  if(field.type==='string'&&(typeof value!=='string'||value.length>256))throw new Error(`${field.label}: enter at most 256 characters.`);
  if(field.type==='enum'&&!field.choices.includes(value))throw new Error(`${field.label}: choose a supported option.`);
  if(field.type==='array'&&(!Array.isArray(value)||value.length>24||value.some(v=>typeof v!=='string'||!v.trim()||v.length>80)||new Set(value).size!==value.length))throw new Error('Choose distinct model aliases, up to 80 characters each.');
  if(key==='layer_split'&&value!=='auto'){const points=value.split(',').map(Number);if(!points.length||points.some((n,i)=>!Number.isInteger(n)||n<1||n>=48||(i>0&&n<=points[i-1])))throw new Error('Layer split points must increase from 1 to 47, for example 16,32.');}
  result[key]=value;
 }
 return result;
}
const MASK='••••••••';
export function publicMcpServers(servers){return Object.fromEntries(Object.entries(servers).map(([name,s])=>[name,{...s,...(s.env?{env:Object.fromEntries(Object.keys(s.env).map(k=>[k,MASK]))}:{}),...(s.headers?{headers:Object.fromEntries(Object.keys(s.headers).map(k=>[k,MASK]))}:{})}]));}
export function validateMcpServers(servers,previous={}){
 if(!servers||typeof servers!=='object'||Array.isArray(servers)||Object.keys(servers).length>30)throw new Error('MCP servers must be an object with at most 30 entries.');
 const out={};for(const [name,s] of Object.entries(servers)){
  if(!s||typeof s!=='object'||Array.isArray(s))throw new Error('Give each tool server a simple, unique name.');
  // a name that was saved before the 32-character rule stays saveable (unrelated settings must not be blocked); the agent reports it as left out
  if(!validMcpName(name)&&!Object.hasOwn(previous??{},name))throw new Error(MCP_NAME_HELP);
  if(!!s.command===!!s.url)throw new Error(`${name}: choose a command or HTTP URL.`);
  for(const key of Object.keys(s))if(!['command','args','env','cwd','url','headers','disabled'].includes(key))throw new Error(`${name}: unsupported property ${key}.`);
  if(s.command&&(typeof s.command!=='string'||s.command.length>1000||/[\n\r\0]/.test(s.command)))throw new Error(`${name}: choose a valid program.`);
  if(s.url){let url;try{url=new URL(s.url);}catch{throw new Error(`${name}: enter a valid HTTP URL.`);}if(!['http:','https:'].includes(url.protocol)||url.username||url.password)throw new Error(`${name}: use an HTTP URL without embedded credentials.`);}
  if(s.args&&(!Array.isArray(s.args)||s.args.length>100||s.args.some(v=>typeof v!=='string'||v.length>4000||v.includes('\0'))))throw new Error(`${name}: arguments must be a list of strings.`);
  if(s.cwd!==undefined&&(typeof s.cwd!=='string'||s.cwd.length>4000||/[\r\n\0]/.test(s.cwd)))throw new Error(`${name}: choose a valid working directory.`);
  if(s.disabled!==undefined&&typeof s.disabled!=='boolean')throw new Error(`${name}: disabled must be true or false.`);
  out[name]={...s};for(const key of ['env','headers'])if(s[key]){if(typeof s[key]!=='object'||Array.isArray(s[key]))throw new Error(`${name}: ${key} must be an object.`);out[name][key]={};for(const [k,v] of Object.entries(s[key])){if(typeof v!=='string'||v.length>16000||v.includes('\0')||(key==='headers'&&(/[\r\n]/.test(v)||!/^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/.test(k))))throw new Error(`${name}: ${key} values must be text.`);if(v===MASK){if(previous[name]?.[key]?.[k]===undefined)throw new Error(`${name}: enter the new ${k} value.`);out[name][key][k]=previous[name][key][k];}else out[name][key][k]=v;}}
 }
 return out;
}

const sensitive=name=>/KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL/i.test(name);
export function publicNativeEnv(env){return Object.fromEntries(Object.entries(env).filter(([name])=>ENV_FLAG.test(name)).map(([name,value])=>[name,sensitive(name)?MASK:value]));}
export function restoreNativeEnv(env,previous={}){const result=Object.fromEntries(Object.entries(previous).filter(([name])=>!ENV_FLAG.test(name)));for(const [name,value] of Object.entries(env)){if(value===MASK){if(previous[name]===undefined)throw new Error(`Enter a new value for ${name}.`);result[name]=previous[name];}else result[name]=value;}return result;}
