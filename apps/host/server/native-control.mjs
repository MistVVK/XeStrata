/** Native Strata control plane. All user actions are explicit; opening the UI only reads state. */
import {missingFilesError,missingNames} from './missing-files.mjs';
import { spawn } from 'node:child_process';
import {chooseParallel,applyParallel,withoutParallel,FALLBACK_NOTE} from './native-parallel.mjs';
import { randomBytes, randomUUID, createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, writeFileSync, mkdirSync, realpathSync, copyFileSync, renameSync, rmSync, statfsSync, unlinkSync, chmodSync, statSync, accessSync, constants } from 'node:fs';
import { join, resolve, dirname, basename, sep, relative, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'node:net';
import { draftingHint, NATIVE_FIELDS, APP_OWNED_FIELDS, withoutDraftLayer, resolveDrafter, validateNativeSettings, validateMcpServers, publicMcpServers, publicNativeEnv, restoreNativeEnv, speedSwitchRows } from './native-settings.mjs';
import { resourceSettings, saferStart, reserveChoice } from './native-resources.mjs';
import { USER_KEY, capture as captureUserArgs, merge as mergeUserArgs, setFlag as setUserFlag, stamped as stampedUserArgs } from './native-user-args.mjs';
import { resolvePython, venvPython } from './python-resolver.mjs';
import os from 'node:os';
import { markPictures, runtimeVision } from './native-vision.mjs';
import { dsEngineArgs, dsUserArgs, dsArgsNote, withoutVisionFlag } from './native-ds-args.mjs';
import { componentsDirectory, installerEnvironment, controlEnvironment, sourceServerEnvironment, UPDATE_LOCK_HELD } from './control-runtime.mjs';
import { createEngineStep } from './engine-step.mjs';
import { dataFolderFromPick, isDriveRoot, makeFolder, rememberDataFolder, syncDataFolder, terminalSettingsFile } from './model-folder.mjs';
import { createAutoTune, writeTuneConfig } from './native-autotune.mjs';
import { plainError, PHASE_TEXT, UPDATE_WAITING_FOR_LOCK } from './update-messages.mjs';
import { checkAndParseRunConfig, checkRunConfig, describeIssues } from './run-config-check.mjs';
import { openTextFile } from './open-text-file.mjs';
import { markEngine, engineRefusal } from './engine-version.mjs';
import { friendlyModelName } from './model-names.mjs';
import { parseSaved as parseRoles, mirror as mirrorRoles } from './gpu-roles.mjs';
import { configContextLimit } from './dsv41-context.mjs';

const ROOT=resolve(dirname(fileURLToPath(import.meta.url)),'..');
/* What the load says when the server had to start the normal layer split because the all-cards split engine would not come up (the engine's own reason stays in its log). */
const SPLIT_FELL_BACK='The all-cards split engine could not start, so the normal layer split runs.';
const safeId=value=>typeof value==='string'&&/^[a-z0-9](?:[a-z0-9_-]|\.(?!\.)){0,79}$/i.test(value);   /* a dot inside the name: deepseek-v4.1-flash-ist-das-2.00bpw */
const busy=requests=>requests.some(r=>['queued','generating'].includes(r.state));
function readJson(path,fallback={}){try{return JSON.parse(readFileSync(path,'utf8').replace(/^\uFEFF/,''));}catch{return fallback;}}

/** The plain words for "no usable engine": the model still downloads, but loading needs the engine (owner decision 2026-10-18). */
export const ENGINE_NOT_RECOGNISED='Strata didn’t recognise an engine for this computer. The model is downloaded, but it can’t run until the right engine is installed. Open Settings → About.';
export function createNativeControl({directory,requests=()=>[],changed=()=>{},spawnProcess=spawn,fetcher=fetch,modelsDirectory,nativeDataDirectory=value=>value,beforeLoad=async()=>{},hardwareProbe=()=>[],control=null,ready=null,appUpdater=null,lockWaitMs=60000,lockPollMs=2000,defaultRepo=process.env.STRATA_NATIVE_REPO??resolve(ROOT,'../../Strata')}={}){
 mkdirSync(directory,{recursive:true});
 const settingsPath=join(directory,'native-host.json');
 let preferences=readJson(settingsPath,{repo:defaultRepo});
 let child=null,seq=0,buffer='',pending=new Map(),definitions=[],refreshing=null,lastRead=0,connecting=null;
 /* QAGUI 0213: until Control has answered once, the state says it is still being asked (`checking`), not that nothing is installed. A failed read during Control's start is read again after a few seconds instead of standing for 45 s; after READ_TRIES failures the answer is final and the error shows. */
 let everRead=false,failedReads=0;const READ_TRIES=4,RETRY_MS=3000;
 let snapshot={available:false,repo:preferences.repo,summary:'Connect a local Strata installation.',catalog:null,installed:null,hardware:null,error:null};
 const instances=new Map(),processes=new Map(),jobs=new Map(),logs=[];
 const savedJobs=readJson(join(directory,'native-jobs.json'),[]);/* An app update ends the app on purpose: the installer replaces it and starts it again. The job that asked for it was saved as running together with the app version it started from (appUpdateFrom); a start that finds a different version knows the update finished, instead of calling it an interrupted operation. */
 const thisAppVersion=()=>process.env.STRATA_APP_VERSION||null;
 /* The job that started an app update is still 'running' when the app ends for the installer; an older host's close() saved it as 'cancelled' (the app update ends the host the normal way on Linux): with appUpdateFrom set, that is the same record (APP-LIN 0.2.14). */
 const closedForAppUpdate=j=>j.state==='running'||j.state==='cancelled'&&typeof j.appUpdateFrom==='string';
 /* A record an OLDER host wrote (no appUpdateFrom): the app's own program file was replaced after the job began, so the installer finished. <install>/host/server is ROOT's parent chain: ROOT = <install>/host. */
 const appProgramNewerThan=started=>{try{const t=Date.parse(started);if(!Number.isFinite(t))return false;const dir=dirname(ROOT),exe=join(dir,process.platform==='win32'?'strata-desktop.exe':'strata-desktop'),times=[];for(const f of [exe,join(dir,'uninstall.exe')])if(existsSync(f)){const st=statSync(f);times.push(st.mtimeMs,st.ctimeMs);}/* the installer keeps the build time as the program's modified time but writes uninstall.exe and changes the program file's metadata when it installs */return times.some(x=>x>t);}catch{return false;}};
 const settledByAppUpdate=j=>closedForAppUpdate(j)&&j.kind==='update-strata'&&((typeof j.appUpdateFrom==='string'&&!!thisAppVersion()&&thisAppVersion()!==j.appUpdateFrom)||(typeof j.appUpdateFrom!=='string'&&appProgramNewerThan(j.startedAt)));
 const neverFinishedAppUpdate=j=>closedForAppUpdate(j)&&j.kind==='update-strata'&&typeof j.appUpdateFrom==='string'&&!!thisAppVersion()&&thisAppVersion()===j.appUpdateFrom;
 for(const j of (Array.isArray(savedJobs)?savedJobs:[]).slice(-12))jobs.set(j.id,{...j,...(settledByAppUpdate(j)?{state:'done',error:null,summary:`Strata is updated to ${thisAppVersion()}.`,finishedAt:new Date().toISOString()}:neverFinishedAppUpdate(j)?{state:'error',error:`The update to ${j.appUpdateTo??'the new version'} did not finish: the app closed for the installer, but the old version ${j.appUpdateFrom} started again. Try Update again, or download the installer from the Strata release page (https://github.com/Niko1221/Strata/releases/latest).`,summary:'The app update did not finish',finishedAt:new Date().toISOString()}:j.state==='running'?{state:'error',error:'The host restarted before completion. Check diagnostics, then retry; completed files are kept.',summary:'Interrupted operation'}:{})});
 /* Packaged mode: the app ships Strata Control (control-runtime.mjs). Models are run configs in an install folder, the engine
    comes from the component folder, and a start is `strata runtime start` under the owner runtime. Nothing is installed,
    updated or fetched from here. Without `control` (a source checkout) every path below is the original one. */
 const packaged=!!control;
 /* The auto-tuner and the saved tunings are Strata Control's (`strata tune auto`, `strata presets`): the app runs them and follows their status file. */
 const autotuner=createAutoTune({spawnProcess,workDir:join(directory,'native-tune'),launcher:()=>{if(packaged)return {command:control.control.python,prefix:[control.control.cli],cwd:control.control.dir,env:controlEnvironment(control.control)};const py=python();return {command:py.command,prefix:[...py.args,join(repo(),'strata.py')],cwd:repo(),env:{...process.env,PYTHONUNBUFFERED:'1',PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8'}};}});
 /* In the packaged app the model catalog, the GPU probe and the install job are Control's bundled tools/strata_mcp.py and
    setup.py, run on Control's own Python in the installer's mode (STRATA_CONTROL_INSTALL: no pip, no engine, no compiler).
    The install only runs when the user clicks Download; starting and serving (control.start) keep STRATA_NO_ACQUIRE. */
 const defaultInstall=()=>process.env.STRATA_INSTALL_DIR?resolve(process.env.STRATA_INSTALL_DIR):join(dirname(componentsDirectory()),'install');
 let packagedInstall=null;
 /* The component updates and the rollback go through Control, not through the native snapshot, so they work in the first seconds after the app starts and on a PC that has no engine yet. */
 const COMPONENT_KINDS=['update-engine','update-strata','rollback','install-engine-file'];
 const updateLog=line=>{try{console.log(`update · ${line}`);}catch{/* no console */}};
 const NO_INSTALL='Installing, updating, tuning with calibration or rolling back model files and engines happens from the Terminal (strata install), never while the app runs.';
 let defaultId=null,timer=null,closing=false,polling=false;
 /* Models the engine update stopped (with the person's yes): kept so the app can offer to start them again with the same settings. */
 let stoppedForUpdate=null;
 const emit=()=>{const value=state();writeFileSync(join(directory,'native-jobs.json'),JSON.stringify(value.jobs),{mode:0o600});changed(value);};
 const append=(instanceId,line)=>{logs.push({at:new Date().toISOString(),instanceId,line:String(line).slice(0,2000)});if(logs.length>400)logs.shift();};
 function diskSpace(path){try{const s=statfsSync(path);return {free:Number(s.bavail)*Number(s.bsize),total:Number(s.blocks)*Number(s.bsize)};}catch{return null;}}
 /* Does the engine in this release serve DFlash2? Control knows (strata_control/drafting.py); asked once, false until it answers. The Models screen offers the choice only when true. */
 let dflash2Served=null,dflash2Support=null,dflash2Selq4=false,dflash2Split=false;
 function probeDrafting(){if(!packaged||dflash2Served!==null||typeof control.draftingPlan!=='function')return;dflash2Served=false;control.draftingPlan({mode:'auto'}).then(p=>{dflash2Served=p.serves===true;dflash2Support=p.serves===true&&p.support&&p.support_line?{level:p.support,line:p.support_line}:null;dflash2Selq4=p.serves===true&&p.selq4===true;dflash2Split=p.split===true;emit();}).catch(()=>{});}
 function state(){const checking=!everRead&&failedReads<READ_TRIES;const folder=preferences.downloadDirectory??(modelsDirectory?modelsDirectory():snapshot.installed?.data_folder),space=folder?diskSpace(folder):null;return {...snapshot,checking,downloadDirectory:folder,downloadFreeBytes:space?.free??null,downloadTotalBytes:space?.total??null,fields:nativeFields(),instances:[...instances.values()].map(({key,configPath,fingerprint,mcpServers,...i})=>structuredClone(i)),defaultId,stoppedForUpdate:stoppedForUpdate?{updated:stoppedForUpdate.updated,at:stoppedForUpdate.at,models:stoppedForUpdate.instances.map(i=>({id:i.id,label:i.label}))}:null,jobs:[...jobs.values()].slice(-12).map(({process,abort,engineAbort,rawResult,...j})=>j),capabilities:{responses:sourceText().includes('/v1/responses'),updates:!!snapshot.available,rollback:previousEngine(),componentUpdates:packaged&&typeof control.update==='function',calibration:packaged?false:snapshot.hardware?.gpus?.some(g=>/nvidia/i.test(g.name??''))??false,dflash2:dflash2Served===true,dflash2Support,dflash2Selq4,dflash2Split}};}
 let sourceCache={path:'',time:0,text:''};
 function sourceText(){try{const path=join(packaged?control.control.dir:preferences.repo,'serve/server.py'),time=statSync(path).mtimeMs;if(sourceCache.path===path&&sourceCache.time===time)return sourceCache.text;sourceCache={path,time,text:readFileSync(path,'utf8')};return sourceCache.text;}catch{return '';}}
 function nativeFields(){const source=sourceText();return NATIVE_FIELDS.map(field=>({...field,supported:APP_OWNED_FIELDS.includes(field.id)||field.id.startsWith('vision_')||field.id.startsWith('mcp_')||field.id==='vram_reserve_mib'||source.includes(field.id)}));}
 const modelArena=id=>({arenaGb:(snapshot.catalog?.families??[]).flatMap(f=>f.sizes??[]).find(s=>s.id===id)?.experts_gb,engineVersion:snapshot.engine?.version??snapshot.installed?.engine?.version??null});
 /* A start that ends with `server_exited` carries the engine's last log line with it; the app turns the code into words and
    keeps this text under "Details" (src/lib/plain-errors.ts). */
 function exitCodeText(code){const n=Number(code);if(!Number.isFinite(n)||n===0)return '';const u=n<0?n+2**32:n;return u>=0x80000000?`exit code 0x${u.toString(16).toUpperCase().padStart(8,'0')}`:`exit code ${u}`;}
 function humanStartError(error){const text=String(error?.message??error),tail=error?.output?.log_tail,code=exitCodeText(error?.output?.engine_exit_code);const head=code&&!text.includes(code)?`${text} (${code})`:text;return tail&&!head.includes(tail)?`${head}: ${tail}`:head;}
 /* A run config the app's install job wrote names files inside the app's own Control folder (`cwd`, `--expert-profile`). When the
    app is reinstalled to another folder, or an update moves Control, those absolute paths point at nothing and every installed
    model breaks. At every read of the installed models, a path that ends up inside a folder called `control` that no longer
    exists is pointed at the Control this app has now, when the file is there. The config is saved (with a .bak) only then. */
 function repairControlPaths(dir){if(!packaged||!dir||!existsSync(dir))return;let names=[];try{names=readdirSync(dir).filter(n=>/^strata-[a-z0-9_.-]+\.json$/i.test(n));}catch{return;}
  const here=control.control.dir,inside=/^(.+?[\/]control)(?=[\/]|$)(.*)$/i;
  const fix=value=>{if(typeof value!=='string')return value;const m=value.match(inside);if(!m||existsSync(value)||resolve(m[1]).toLowerCase()===resolve(here).toLowerCase())return value;const candidate=join(here,m[2]);return existsSync(candidate)?candidate:value;};
  for(const name of names){const path=join(dir,name),config=readJson(path,null);if(!config||typeof config!=='object')continue;
   const next={...config,...(config.cwd!==undefined?{cwd:fix(config.cwd)}:{}),...(Array.isArray(config.args)?{args:config.args.map(fix)}:{})};
   if(JSON.stringify(next)!==JSON.stringify(config)){try{writeModelConfig(path,next);}catch{/* read-only folder: the load reports it */}}}}
 /* Every folder that holds run configs. Also run before a component job (the engine update test-starts an installed model with its run config) and before a load:
    an AppImage's Control lives in /tmp/.mount_<random>, a new name at every start, so a path setup wrote there is gone after the app restarts (QAGUI-LIN 0213). */
 function repairAllControlPaths(){try{for(const dir of [preferences.installDir,packagedInstall,installRoot()])repairControlPaths(dir);}catch{/* the read that follows reports a bad folder */}}
 /* DeepSeek reads pictures with the engine's own tower, which is part of the pack (pack_gguf writes the tower's geometry into ds4.txt when the picture file was there): a pack with it is "prepared", a text-only pack (made with --vision off, as the app did before 0.2.9) is not, and the Models screen then offers the one missing shared file. null for every other model. */
 const dsPictures=cfg=>{if(cfg?.arch!=='deepseek41')return null;const a=Array.isArray(cfg.args)?cfg.args:[];const at=a.indexOf('--pack');if(at<0||typeof a[at+1]!=='string')return false;try{const t=readFileSync(join(a[at+1],'ds4.txt'),'utf8');return /image_token_id/.test(t)&&/vision[.]num_hidden_layers/.test(t);}catch{return false;}};
 /* DeepSeek's picture switch is a pair, --vision on|off, in the saved arguments: set it to the person's choice at every load (on only where the pack carries the tower). */
 const dsVisionPair=(a,value)=>{if(!value)return a;const at=a.indexOf('--vision');return at<0?[...a,'--vision',value]:a.map((x,i)=>i===at+1?value:x);};
 function installedInfo(info){if(!info)return info;return {...info,models:(info.models??[]).map(entry=>{const cfg=/^strata-[a-z0-9_.-]+\.json$/i.test(entry.config??'')?readJson(entry.configPath??join(repo(),entry.config)):{};const at=cfg.args?.indexOf('--native')??-1,nativeFile=at>=0?cfg.args[at+1]:null;const dataFolder=typeof nativeFile==='string'&&basename(dirname(dirname(nativeFile)))==='models'?dirname(dirname(dirname(nativeFile))):info.data_folder;/* The folder this model's own run config keeps its prepared data in: the one that holds its packs/<tag> (or mtp/rt). It is the real data folder of the model, wherever the app's models folder is; a model in the person's own GGUF folder has its pack and draft layer here, not in the app's folder. A layout that only looks like <data>/models/<tag> says nothing: no pack, no folder. */const flagValue=flag=>{const i=cfg.args?.indexOf(flag)??-1;return i>=0&&typeof cfg.args[i+1]==='string'?cfg.args[i+1]:undefined;};const holder=(path,child,name)=>{if(!path||basename(dirname(path)).toLowerCase()!==child||(name&&basename(path).toLowerCase()!==name))return undefined;return dirname(dirname(path));};const runDataFolder=holder(flagValue('--pack'),'packs')??holder(flagValue('--mtp'),'mtp','rt');/* A model the person pointed at their own GGUF folder (not <data>/models/<tag>): remembered, so a later load runs setup with that folder again instead of downloading the files (#3199). */const ownGgufDir=typeof nativeFile==='string'&&basename(dirname(dirname(nativeFile)))!=='models'&&existsSync(nativeFile)?dirname(nativeFile):undefined;return {...entry,ownGgufDir,runDataFolder,contextMax:configContextLimit(cfg)??undefined,diskGb:entry.diskGb??entry.disk_gb,diskBytes:entry.diskBytes??entry.disk_bytes,sharedBytes:entry.sharedBytes??entry.shared_bytes,removableGb:entry.removableGb??entry.removable_gb,dataFolder,contextChosen:cfg.ui_context_chosen===true,visionPrepared:dsPictures(cfg)??(!!cfg.vision&&entry.picturesMissing!==true),/* The picture files are prepared when the run config has a vision section: the Models screen then loads straight away instead of running setup again (it used to read 'off' for every model, which sent every load of a picture-capable model through "Apply settings and load", and setup writes the arguments, the draft layer and the reader back each time). */images:(dsPictures(cfg)??(!!cfg.vision&&entry.picturesMissing!==true))?'yes':'off',draftLayer:cfg.mtp!==false,drafting:cfg.drafting??'auto',/* Picture reading is ON unless the person turned it off: no saved choice and not loaded is undefined, not false (owner rule: pictures on by default). */visionChosen:cfg.ui_vision_chosen===true,visionEnabled:cfg.ui_vision_enabled??(()=>{const running=[...instances.values()].find(i=>i.model===entry.model&&i.status==='ready');return running?!!running.vision:undefined;})(),visionDevice:cfg.ui_vision_device,resourceProfile:cfg.ui_resource_profile,lowRamMode:cfg.args?.includes('--resident-experts')?'resident':cfg.args?.includes('--mmap-experts')?'mmap':'auto'};})};}
 function installRoot(){const found=[preferences.installDir].find(d=>d&&existsSync(d));if(found)return realpathSync(found);mkdirSync(defaultInstall(),{recursive:true});return realpathSync(defaultInstall());}
 /* The settings file Strata Control and the installer read (control-runtime.mjs sets STRATA_SETTINGS_FILE to it): the app's data folder is written there. */
 function controlSettingsFile(){return process.env.STRATA_SETTINGS_FILE||join(dirname(componentsDirectory()),'app-settings.json');}
 /* One models folder for the app and the Terminal: the one changed last wins and the other settings file is updated (nothing is moved). */
 function shareDataFolder(){if(!packaged)return;try{let at=0;if(preferences.downloadDirectory)try{at=statSync(settingsPath).mtimeMs/1000;}catch{/* no file yet */}const found=syncDataFolder({appFile:controlSettingsFile(),terminalFile:terminalSettingsFile(),app:preferences.downloadDirectory?{folder:preferences.downloadDirectory,at}:null});if(found.from==='terminal'&&found.folder&&found.folder!==preferences.downloadDirectory){preferences={...preferences,downloadDirectory:found.folder};writeFileSync(settingsPath,JSON.stringify(preferences,null,2),{mode:0o600});}}catch{/* the shared setting is a convenience: never stop the app */}}
 function repo(){if(packaged)return installRoot();const path=realpathSync(preferences.repo);if(!existsSync(join(path,'setup.py'))||!existsSync(join(path,'tools/strata_mcp.py'))||!existsSync(join(path,'serve/server.py')))throw new Error('Choose the Strata repository folder containing setup.py and tools/strata_mcp.py.');return path;}
 function python(){return resolvePython({repo:preferences.repo});}
 function send(method,params,timeout=30000){return new Promise((resolve,reject)=>{const id=++seq;const alarm=setTimeout(()=>{pending.delete(id);reject(new Error('The native engine did not respond in time. Check engine logs.'));},timeout);pending.set(id,{resolve,reject,alarm});child.stdin.write(JSON.stringify({jsonrpc:'2.0',id,method,params})+'\n');});}
 function disconnect(){child?.kill();child=null;definitions=[];for(const p of pending.values()){clearTimeout(p.alarm);p.reject(new Error('Native control connection closed.'));}pending.clear();}
 async function connect(){if(closing)throw new Error("The host is shutting down.");if(child&&definitions.length)return;if(!connecting)connecting=startClient().finally(()=>{connecting=null;});return connecting;}
 async function startClient(){
  const folder=repo();
  buffer='';const py0=packaged?{command:control.control.python,args:[]}:python();
  const script=packaged?join(control.control.dir,'tools/strata_mcp.py'):join(folder,'tools/strata_mcp.py');
  const client=child=spawnProcess(py0.command,[...py0.args,script,...(packaged?['--root',folder,'--src',control.control.dir]:[])],{cwd:packaged?control.control.dir:folder,windowsHide:true,stdio:['pipe','pipe','pipe'],env:packaged?installerEnvironment(control.control,{installDir:folder}):{...process.env,PYTHONUNBUFFERED:'1',PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8'}});
  client.stdout.on('data',chunk=>{if(child!==client)return;buffer+=chunk.toString();let end;while((end=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,end);buffer=buffer.slice(end+1);let message;try{message=JSON.parse(line);}catch{continue;}const p=pending.get(message.id);if(p){clearTimeout(p.alarm);pending.delete(message.id);message.error?p.reject(new Error(message.error.message)):p.resolve(message.result);}}});
  child.stderr.on('data',chunk=>append('control',chunk.toString()));
  client.on('error',error=>{if(child===client){snapshot.error=error.message;disconnect();}});
  client.on('exit',()=>{if(child===client){child=null;for(const p of pending.values()){clearTimeout(p.alarm);p.reject(new Error('Strata control stopped. Python 3.10 or newer is required.'));}pending.clear();}});
  await send('initialize',{protocolVersion:'2024-11-05',capabilities:{},clientInfo:{name:'Strata UI',version:'0.1.0'}});
  child.stdin.write(JSON.stringify({jsonrpc:'2.0',method:'notifications/initialized'})+'\n');
  definitions=(await send('tools/list',{})).tools??[];
 }
 async function controlTool(name){
  if(name==='strata_status'){
   repairAllControlPaths();
   // Always the app's own install folder (or the one the person chose in Settings): Control is told explicitly, so it never falls
   // back to an older command-line Strata's folder remembered in %APPDATA% (tester report 2026-10-17: logs landed in T:\Strata).
   const list=await control.installedModels({installDir:installRoot()});
   packagedInstall=list.install_dir;
   const engine=await control.engine();
   const installer=await mcpCall('strata_status',{hardware:true}).catch(error=>({error:error.message}));
   const disk=Object.fromEntries((installer.installed?.models??[]).map(m=>[m.model,{diskGb:m.disk_gb,diskBytes:m.disk_bytes,sharedBytes:m.shared_bytes,removableGb:m.removable_gb,partsMissing:m.parts_missing??0,missingFiles:m.missing_files}]));
   const onlyPictures=m=>m.component_status==='missing_or_invalid'&&Array.isArray(m.missing)&&m.missing.length>0&&m.missing.every(p=>p?.part==='vision/mmproj')&&Array.isArray(m.issue_codes)&&m.issue_codes.length===m.missing.length&&m.issue_codes.every(c=>/^COMPONENT_(MISSING|INVALID)$/.test(c));
   const models=(list.models??[]).filter(m=>m.component_status!=='invalid'&&m.source_config).map(m=>({...(disk[m.id]??{}),model:m.id,config:basename(m.source_config),configPath:m.source_config,model_name:m.name,context:m.context_tokens??0,backend:m.backend,ready:(m.component_status==='present'||onlyPictures(m))&&!(disk[m.id]?.partsMissing),/* Only the picture file is gone (run config has the picture section, the file does not exist): the model is not "missing files". It loads, and the Load fetches the picture file again. */picturesMissing:m.component_status!=='present'&&onlyPictures(m),missing:missingNames(m.missing,disk[m.id]?.missingFiles),images:'off',port:undefined,last_used:null}));
   const cores=os.cpus().length;
   return {summary:engine.usable?`Strata Control is ready; engine ${engine.engine.version} (${engine.engine.platform}-${engine.engine.backend}).`:(engine.notice??'No engine for this PC was found.'),
    installDir:list.install_dir,engine:{usable:engine.usable,notice:engine.notice,wrongMake:engine.wrongMake===true,version:engine.engine?.version??null,backend:engine.backend,componentsDir:engine.components_dir,previous:engine.previous??null,pinned:engine.pinned??null,updateNotice:engine.update_notice??null,updateAvailable:engine.update_available??null},
    installed:{strata_folder:list.install_dir,data_folder:null,engine:engine.engine?{version:engine.engine.version,backend:engine.engine.backend,folder:engine.engine.root}:{folder:null,version:null},models,usage:installer.installed?.usage??null},
    hardware:installer.hardware?{...installer.hardware,cpu:{...(installer.hardware.cpu??{}),cores:installer.hardware.cpu?.cores??cores},gpus:installer.hardware.gpus??[]}:{cpu:{cores},ram_gb:Math.round(os.totalmem()/1024**3),gpus:[]},
    recommendation:installer.recommendation??null,installerError:installer.error??null,install:installer.install??null,running:null};
  }
  throw new Error(`Strata Control has no ${name}.`);
 }
 async function tool(name,args={}){if(packaged&&name==='strata_status')return controlTool(name,args);return mcpCall(name,args);}
 async function mcpCall(name,args={}){await connect();if(!definitions.some(d=>d.name===name))throw new Error(`This Strata installation does not support ${name}. Update the installation first.`);const result=await send('tools/call',{name,arguments:args},120000);let value=result.structuredContent;if(!value){const text=result.content?.find(c=>c.type==='text')?.text;try{value=JSON.parse(text);}catch{value={summary:text};}}if(result.isError)throw new Error(value?.error??value?.summary??'Native operation failed.');return value;}
 async function refresh(force=false){
  if(ready)await ready.catch(()=>{});
  if(closing)return state();
  /* A read that is not forced never makes the window wait for the (slow) status of Control: it gets the last state at once, and a stale one
     is refreshed in the background. Only the first read, and a forced one after an operation, waits. */
  if(refreshing)return !force&&snapshot.available?state():force?refreshing.then(()=>refresh(true)):refreshing;      /* a forced read that meets one already running (started before an engine update finished) waits for it and reads again, so it never settles on the old answer */if(!force&&Date.now()-lastRead<45000)return state();
  probeDrafting();
  refreshing=(async()=>{try{const [catalog,status]=packaged?await (async()=>{const s=await tool('strata_status',{hardware:true});const c=await tool('strata_models').catch(error=>({families:[],contexts:[],recommendation:{family:null,model:null},context_note:'',error:error.message}));return [c,s];})():await Promise.all([tool('strata_models'),tool('strata_status',{hardware:true})]);let visionReader=false;if(packaged){try{const e=await control.engine();const root=e?.engine?.root??(e?.engine?.exe?dirname(e.engine.exe):'');visionReader=!!root&&['strata-vision','strata-vision.exe'].some(n=>existsSync(join(root,n)));}catch{visionReader=snapshot.visionReader===true;}}const markImages=c=>markPictures(c,visionReader);snapshot={available:true,repo:packaged?status.installDir:repo(),version:packaged?(control.version??null):await execute('git',['describe','--tags','--always'],repo()).then(s=>s.trim()).catch(()=>null),summary:status.summary,catalog:markEngine(packaged?markImages(catalog):catalog,status.engine),engineInfo:{version:status.engine?.version??null,source:status.engine?.source??null},visionReader:packaged?visionReader:true,installed:installedInfo(status.installed),hardware:status.hardware,error:null,install:status.install,external:status.running??null,...(packaged?{packaged:true,installDir:status.installDir,engine:status.engine,installerError:status.installerError}:{})};lastRead=Date.now();everRead=true;failedReads=0;if(status.install?.running&&![...jobs.values()].some(j=>j.state==='running'))begin('install',j=>watchInstall(j,status.install));}catch(error){failedReads++;snapshot={...snapshot,available:false,error:error.message};lastRead=!everRead&&failedReads<READ_TRIES?Date.now()-45000+RETRY_MS:Date.now();}emit();return state();})().finally(()=>{refreshing=null;});const run=refreshing;if(!force&&snapshot.available){run.catch(()=>{});return state();}return run;
 }
 function configure({repository}){if(packaged){if(instances.size||[...jobs.values()].some(j=>j.state==='running'))throw new Error('Unload models and finish operations before changing the installation.');const next=realpathSync(String(repository));if(!statSync(next).isDirectory())throw new Error('Choose the folder that holds your Strata model configurations.');preferences={...preferences,installDir:next};packagedInstall=null;disconnect();writeFileSync(settingsPath,JSON.stringify(preferences,null,2),{mode:0o600});lastRead=0;return refresh(true);}if(instances.size||[...jobs.values()].some(j=>j.state==='running'))throw new Error('Unload models and finish operations before changing the installation.');const next=realpathSync(String(repository));const old=preferences;preferences={...preferences,repo:next};try{repo();}catch(e){preferences=old;throw e;}disconnect();writeFileSync(settingsPath,JSON.stringify(preferences,null,2),{mode:0o600});lastRead=0;return refresh(true);}
 function setDownloadDirectory(input,{insideJob=false}={}){if(!insideJob&&[...jobs.values()].some(j=>j.state==='running'))throw new Error('Finish the current operation before changing the folder.');if(typeof input!=='string'||!input.trim()||/[\r\n\0]/.test(input))throw new Error('Choose a model folder.');let path;try{path=realpathSync(input.trim());}catch{throw new Error('That folder does not exist. Check the path, or that the drive is connected.');}if(!statSync(path).isDirectory())throw new Error('Choose a folder, not a file.');/* A whole drive (D:\) is not filled with models: D:\Strata is made and used. */if(isDriveRoot(path)){const own=dataFolderFromPick(path).path;makeFolder(own);path=realpathSync(own);}try{accessSync(path,constants.R_OK|constants.W_OK);}catch{throw new Error('Strata cannot write in that folder. Choose a folder you can save files in.');}preferences={...preferences,downloadDirectory:path};/* the same choice for the installer and the Terminal: the Control settings file */if(packaged)rememberDataFolder(controlSettingsFile(),path,Date.now()/1000,true);writeFileSync(settingsPath,JSON.stringify(preferences,null,2),{mode:0o600});shareDataFolder();emit();return state();}
 function recordFile(model){if(!safeId(model))throw new Error('Choose an installed model.');const entry=snapshot.installed?.models?.find(i=>i.model===model);if(!entry)throw new Error('This model is not installed. Download and prepare it first.');const path=realpathSync(entry.configPath??join(repo(),entry.config));if(dirname(path)!==repo()||!/^strata-[a-z0-9_.-]+\.json$/i.test(basename(path)))throw new Error('The model configuration is outside the Strata installation.');return {entry,path};}
 function record(model){const {entry,path}=recordFile(model);return {entry,path,config:readJson(path)};}
 /* The text file a power user edits by hand: where it is and what is wrong in it (run-config-check.mjs). */
 function settingsFile(model){const {path}=recordFile(model);return {model,path,name:basename(path),...checkRunConfig(readFileSync(path,'utf8')),note:'Changes you make in the file apply the next time the model starts.'};}
 async function openSettingsFile(model,options={}){const info=settingsFile(model);await openTextFile(info.path,options);return info;}
 /* A mistake in the file holds the start back with the line, instead of an engine that dies on it. Warnings (an unknown key) never do. */
 function checkedSettingsConfig(text){const check=checkAndParseRunConfig(text);if(!check.ok)throw new Error(`The settings file for this model has a mistake, so the model was not started. ${describeIssues(check.issues,'error').join(' ')}`);return check.config;}
 function settings(model){const {entry,config}=record(model);const values=Object.fromEntries(NATIVE_FIELDS.map(f=>[f.id,config[f.id]??null]));if(typeof values.aliases==='string')values.aliases=values.aliases.split(',').map(v=>v.trim()).filter(Boolean);values.sampling=config.sampling??{};values.args=config.args??[];values.env=publicNativeEnv(config.env??{});return {model,values,mcp:publicMcpServers(config.mcp_servers??config.mcpServers??{}),restartNeeded:true,capabilities:{dsExperts:ownEngine(config)&&dsExpertsSupported(config)},speedSwitches:speedSwitchRows(config.env??{},{deepseek:ownEngine(config),arch:config.arch??null,backend:config.backend??entry?.backend??snapshot.installed?.engine?.backend??null,gpus:snapshot.hardware?.gpus??[]})};}
 /* The Drafting hint of this model's settings: the size of the drafter DFlash2 would fetch for it (Control's plan with the choice DFlash2); the generic hint when Control does not say. */
 async function settingsHints(model){try{if(!packaged||typeof control.draftingPlan!=='function')return {};const {path}=recordFile(model),d=await control.draftingPlan({configPath:path,mode:'dflash2'});return Number(d?.drafter_gb)>0?{drafting:draftingHint(d.drafter_gb)}:{};}catch{return {};}}
 function save(model,{values,mcp}){if([...jobs.values()].some(j=>j.state==='running'))throw new Error('Finish the current operation before saving native settings.');const {path,config}=record(model);if(busy(requests().filter(r=>instances.get(r.instanceId)?.model===model)))throw new Error('Finish this model’s requests before saving its settings.');const patch=validateNativeSettings(values??{}),next={...config};if(patch.ds_experts!=null&&!dsExpertsSupported(config))throw new Error('The Speed choice needs the newer DeepSeek engine. Update the engine in Settings, then choose it.');for(const field of nativeFields())if(field.supported===false&&patch[field.id]!=null&&JSON.stringify(patch[field.id])!==JSON.stringify(config[field.id]))throw new Error(`${field.label} needs a newer native Strata installation. Update it in Settings → About.`);for(const [key,value] of Object.entries(patch)){if(key==='env')next.env=restoreNativeEnv(value,config.env??{});else if(value===null)delete next[key];else next[key]=value;}if('vram_reserve_mib' in patch){const base=captureUserArgs(next);const mine=setUserFlag(base,'--vram-reserve-mib',patch.vram_reserve_mib);if(mine.length)next[USER_KEY]=mine;else delete next[USER_KEY];next.args=mine.length&&patch.vram_reserve_mib!==null?mergeUserArgs(next.args??[],mine):setUserFlag(next.args??[],'--vram-reserve-mib',null);delete next.ui_reserve_applied;if(Array.isArray(next.args_written))next.args_written=[...next.args];}if(mcp!==undefined){next.mcp_servers=validateMcpServers(mcp,config.mcp_servers??config.mcpServers??{});delete next.mcpServers;}copyFileSync(path,path+'.bak');chmodSync(path+'.bak',0o600);const temp=path+'.ui-tmp';writeFileSync(temp,JSON.stringify(next,null,2)+'\n',{mode:0o600});renameSync(temp,path);return settings(model);}
 function previousEngine(){if(packaged)return !!snapshot.engine?.previous;try{const data=snapshot.installed?.data_dir??snapshot.installed?.data_folder;return [join(preferences.repo,'engine/.previous'),data&&join(data,'engine/.previous')].filter(Boolean).some(existsSync)&&readFileSync(join(preferences.repo,'setup.py'),'utf8').includes('--rollback-engine');}catch{return false;}}
 async function plan(args){await refresh();if(!snapshot.available)throw new Error(snapshot.error);args=await controlDrafter(args);const result=await tool('strata_install',installArgs(args,false));
  /* A chosen drafter that is not on this computer yet is part of the download: its size is listed beside the model's (Control decides; nothing is fetched here). */
  const choice=args.drafting??(dflash2Served===true?'auto':undefined);if(packaged&&typeof choice==='string'&&typeof control.draftingPlan==='function'){try{const size=snapshot.catalog?.families?.find(f=>f.family===args.family)?.sizes.find(s=>s.model===args.model),here=snapshot.installed?.models?.find(m=>m.model===size?.id);const d=await control.draftingPlan({mode:choice,family:args.family??null,model:args.model??null,dataDir:here?.runDataFolder??snapshot.installed?.data_folder??null,...(unloadsRunning(args)?{unloading:true}:{})});if(d.needed||d.mtp_needed)return {...result,drafting:{needed:true,drafter:d.needed?'dflash2':'mtp',gb:d.needed?d.gb:d.mtp_gb,line:d.line,...(d.needed&&d.tuned===true?{tuned:true}:{})}};}catch{/* the model plan still stands */}}
  return result;}
 /* Which drafter the install prepares: Control's plan decides, as at the load (strata_control/drafting.py, the per-card Auto). The host's own rule said DFlash2 wherever it is
    served, so a card where Auto means the MTP (a 12 GB CUDA card streaming IQ3_S, an RX 9070) was set up without the MTP layer and its first load ran undrafted (QAGUI 0213).
    No plan (an older Control, an error): the old rule (installArgs). A drafter named by the caller (the MTP layer fetch) is kept. */
 async function controlDrafter(args){
  if(!packaged||args.drafter!==undefined||typeof control.draftingPlan!=='function')return args;
  const choice=args.drafting??(dflash2Served===true?'auto':undefined);if(typeof choice!=='string')return args;
  try{const size=snapshot.catalog?.families?.find(f=>f.family===args.family)?.sizes.find(s=>s.model===args.model),here=snapshot.installed?.models?.find(m=>m.model===size?.id);
   const d=await control.draftingPlan({mode:choice,family:args.family??null,model:args.model??null,dataDir:here?.runDataFolder??snapshot.installed?.data_folder??null});
   if(d?.choice==='mtp'||d?.choice==='dflash2')return {...args,drafter:d.choice};}catch{/* the old rule below */}
  return args;
 }
 /* The folder the person picked in the first-run chooser can be the one that holds their GGUF files (or a folder inside it). It is theirs, not a Strata data folder: packs are never made in it, and a Strata folder around it (<Strata>/models/<tag>) is used instead (#3507, #3424). */
 function ownFilesDataFolder(candidate,args){if(!candidate||args.data_dir!==undefined||!args.gguf_dir)return candidate;const same=(a,b)=>{const x=resolve(a),y=resolve(b);return process.platform==='win32'?x.toLowerCase()===y.toLowerCase():x===y;};const inside=(a,b)=>{const rel=relative(resolve(b),resolve(a));return rel===''||(!!rel&&!rel.startsWith('..')&&!isAbsolute(rel));};if(!same(candidate,args.gguf_dir)&&!inside(candidate,args.gguf_dir))return candidate;const models=dirname(resolve(args.gguf_dir)),top=dirname(models);return basename(models).toLowerCase()==='models'&&/strata/i.test(basename(top))?top:undefined;}
 function installArgs(args,confirm){const allowed=['family','model','context','vision','kv','gpu','gpus','backend','low_ram','port','gguf_dir','drafter'];/* One drafter per run: when the choice means DFlash2 the installer does not fetch the MTP layer's ~5 GB (the drafter is fetched by the load); choosing the MTP later fetches the layer once. */if(args.drafter===undefined&&args.drafting!==undefined||args.drafter===undefined&&dflash2Served===true){const d=resolveDrafter(args.drafting,args.family,dflash2Served,dflash2Split);if(d==='dflash2')args={...args,drafter:'dflash2'};}const size=snapshot.catalog?.families?.find(f=>f.family===args.family)?.sizes.find(s=>s.model===args.model);const existing=snapshot.installed?.models?.find(m=>m.model===size?.id);/* A model adopted from the person's own GGUF folder reports that folder's parent as its data folder: it is theirs, not Strata's, so a second preparation never uses it for the packs. */if(args.gguf_dir===undefined&&existing?.ownGgufDir)args={...args,gguf_dir:existing.ownGgufDir};/* A model that is installed keeps working in the data folder its own run config names (where its pack and draft layer are), whether its files are under it or in the person's own folder: that is never changed or written to the settings from here. Only a folder taken from the app's models folder is the person's choice, and the one the installer is told about. */const own=args.data_dir===undefined?existing?.runDataFolder:undefined;const data=own??nativeDataDirectory(ownFilesDataFolder(args.data_dir??(args.gguf_dir?undefined:existing?.dataFolder)??preferences.downloadDirectory??modelsDirectory?.(),args));if(data&&packaged&&!args.data_dir&&!own)rememberDataFolder(controlSettingsFile(),data);return {...(data?{data_dir:data}:{}),...Object.fromEntries(allowed.filter(k=>args[k]!==undefined).map(k=>[k,args[k]])),/* One card is "gpu", never "gpus" (the installer wants two or more there: "Gpus='0' is not valid", #3203). */...(/^\d{1,2}$/.test(String(args.gpus??''))?{gpu:Number(args.gpus),gpus:undefined}:{}),/* Picture reading is only downloaded when it was asked for (Read pictures); the default stays the smaller download. */...(packaged&&args.vision===undefined&&args.family!=='deepseek41'?{vision:'no'}:{}),confirm};}

 function begin(kind,run,metadata={}){if([...jobs.values()].some(j=>j.state==='running'))throw new Error('Finish or cancel the current engine operation first.');const job={id:randomUUID(),kind,state:'running',startedAt:new Date().toISOString(),summary:'Starting…',result:null,error:null,...metadata};for(const [id,old] of jobs)if(jobs.size>=12&&old.state!=='running')jobs.delete(id);jobs.set(job.id,job);emit();void run(job).then(result=>{if(job.state==='running'){Object.assign(job,{state:'done',summary:result?.summary??'Complete',result,finishedAt:new Date().toISOString()});}}).catch(error=>{if(job.state==='running')Object.assign(job,{state:'error',error:error.message,summary:error.message,...(error.details?{details:error.details}:{}),finishedAt:new Date().toISOString()});}).finally(()=>{if(job.kind.startsWith('update')||job.kind==='rollback')disconnect();lastRead=0;emit();void refresh(true);});return state().jobs.find(j=>j.id===job.id);}
 async function watchInstall(job,current){job.summary=current.step??'Resuming preparation status…';job.result={install:current};while(job.state==='running'){await new Promise(r=>setTimeout(r,1500));const result=await tool('strata_install');job.summary=result.summary;job.result=result;emit();if(!result.install?.running){if(result.install?.exit_code!==0)throw new Error(result.install?.error??'Preparation did not finish. Retry to reuse completed files.');return {summary:'Model preparation complete. Load it from Models.'};}}}
 function externalRunning(){return Array.isArray(snapshot.external)?snapshot.external.filter(s=>![...instances.values()].some(i=>i.port===s.port)):[];}
 async function install(job,args){
  if(ready)await ready.catch(()=>{});
  if(job.state!=='running')return;
  if(externalRunning().length)throw new Error('Stop the native server running outside this UI before preparing files.');
  if(busy(requests().filter(r=>instances.has(r.instanceId))))throw new Error('Finish or cancel requests before preparing model files.');
  for(const id of [...instances.keys()])await unload(id);
  if(job.state!=='running')return;
  args=await controlDrafter(args);if(job.state!=='running')return;
  let result=await tool('strata_install',installArgs(args,true));
  /* The engine comes with the model: the same job makes sure this PC has a usable engine (none installed, or a newer compatible
     one in the signed feed) through Control's component updater, and shows it in the job's own progress (install.engine, install.parts,
     one combined download total). A failed engine download keeps the model files and ends with a retry message. */
  job.engineAbort=new AbortController();
  const engineStep=packaged&&typeof control.update==='function'?createEngineStep({control,signal:job.engineAbort.signal,onChange:()=>{if(job.rawResult)publish(job.rawResult);else emit();}}):null;
  const publish=r=>{job.rawResult=r;job.summary=r.summary;job.result=engineStep&&r.install?{...r,install:engineStep.merge(r.install)}:r;emit();};
  publish(result);
  const engineTask=engineStep?engineStep.run():null;
  try{
   for(;;){
    if(job.state!=='running')return;
    await new Promise(resolve=>setTimeout(resolve,1500));
    result=await tool('strata_install');
    publish(result);
    if(result.install&&!result.install.running){
     if(result.install.exit_code!==0)throw new Error(result.install.error??result.summary);
     if(engineTask){job.summary='Model files are ready. Finishing the engine for this PC…';emit();await engineTask;publish(result);}
     await refresh(true);
     const note=engineStep?.summaryNote();
     const finalResult=engineStep?job.result:result;
     if(note)return {...finalResult,summary:note};
     /* A context the person picked is kept with a marker (ui_context_chosen), so only an old saved 32K, which has no marker, is read as the old default. */
     if(args.contextChosen===true){try{const id=(result.install.family??args.family??'qwen')==='qwen'?String(args.model).toLowerCase():`${args.family}-${String(args.model).toLowerCase()}`;rememberSetting(record(id).path,'ui_context_chosen',true);}catch{/* the install itself is done */}}
     if(args.loadAfter){const model=(result.install.family??args.family??'qwen')==='qwen'?String(args.model).toLowerCase():`${args.family}-${String(args.model).toLowerCase()}`;return loadEngine(job,{model,firstLoad:true,multiple:!!args.multiple,resourceProfile:args.resourceProfile,vision:args.loadVision,...(args.visionUnavailable===true?{visionUnavailable:true}:{}),...(typeof args.drafting==='string'?{values:{drafting:args.drafting}}:{})});}
     return finalResult;
    }
   }
  }catch(error){job.engineAbort?.abort();throw error;}
 }
 function terminate(proc,signal='SIGTERM'){if(!proc)return;if(process.platform!=='win32'&&proc.pid){try{process.kill(-proc.pid,signal);return;}catch{}}proc.kill(signal);}
 async function stopProcess(proc){if(!proc||proc.exitCode!=null)return;await new Promise(resolve=>{const timeout=setTimeout(()=>{terminate(proc,'SIGKILL');resolve();},8000);proc.once('exit',()=>{clearTimeout(timeout);resolve();});terminate(proc);});}
 async function unload(id,force=false){const instance=instances.get(id);if(!instance)throw new Error('This model is no longer loaded.');if(!force&&[...jobs.values()].some(j=>j.kind==='benchmark'&&j.state==='running'&&j.instanceId===id))throw new Error('Stop the benchmark before unloading this model.');if(!force&&busy(requests().filter(r=>r.instanceId===id)))throw new Error('Finish or cancel this model’s requests first.');instance.status='unloading';if(instance.profile){try{await control.stop(instance.profile);}catch(error){if(!force){instance.status='error';instance.error=error.message;emit();throw error;}}}else await stopProcess(processes.get(id));processes.delete(id);instances.delete(id);try{unlinkSync(instance.configPath);}catch{}if(defaultId===id)defaultId=instances.keys().next().value??null;emit();return state();}
 /* Delete a downloaded model: stop it first, then the installer removes the model's own files and its run config and keeps
    whatever another installed model still uses (shared MTP draft, shared PLE shard). */
 async function removeModel(job,args){if(!safeId(args.model))throw new Error('Choose an installed model.');
  if(externalRunning().length)throw new Error('Stop the native server running outside this UI before deleting a model.');
  for(const instance of [...instances.values()].filter(i=>i.model===args.model)){if(busy(requests().filter(r=>r.instanceId===instance.id)))throw new Error('Finish or cancel requests before deleting this model.');await unload(instance.id);}
  job.summary='Deleting downloaded files…';emit();
  const result=await tool('strata_remove',{model:args.model,confirm:true});
  if(result.failed?.length)throw new Error(`Some files could not be removed: ${result.failed.slice(0,3).join('; ')}`);
  await refresh(true);
  /* Verify it is really gone: the model must no longer be listed as installed, and the files the plan named must not exist. */
  const still=(result.plan?.remove??[]).map(i=>i.path).filter(p=>{try{return existsSync(p);}catch{return false;}});
  if(still.length)throw new Error(`Some files could not be removed: ${still.slice(0,3).map(p=>basename(p)).join('; ')}`);
return {summary:result.summary??'Deleted',removed:result.plan?.gb??null};}
 /* The configuration a load starts the engine with (before its port and key): the saved model config, the user's choices and the resource profile. The tuner measures exactly this, so a saved tuning belongs to the same choices a load makes. */
 /* DeepSeek-V4.1-Flash runs on its own engine (strata-dsv41) that takes none of the Flash-Next options the app adds on its own (request slots, the RAM budget, the output clamp, the resource profile): a flag it does not know ends the start with "unknown option". */
 const ownEngine=config=>config?.arch==='deepseek41'||/strata-dsv41/i.test(String(config?.exe??''));
 /* The DeepSeek engine's version, read from the folder the run config names (engine-dsv41/<version>-<platform>-...). --experts exists from engine 0.1.50: an older engine ends the start with "unknown option", so the choice is offered and passed only for 0.1.50 or newer. */
 const DS_EXPERTS_MIN=[0,1,50];
 function dsEngineVersion(config){const m=/engine-dsv41[\\/](\d+(?:\.\d+)+)-/.exec(String(config?.exe??''))||/engine-dsv41[\\/](\d+(?:\.\d+)+)-/.exec(String((config?.lib_dirs??[])[0]??''));return m?m[1].split('.').map(Number):null;}
 function dsExpertsSupported(config){const v=dsEngineVersion(config);if(!v)return false;for(let i=0;i<DS_EXPERTS_MIN.length;i++){const a=v[i]??0,b=DS_EXPERTS_MIN[i];if(a!==b)return a>b;}return true;}
 /* Control's ceiling for one start wait, read from the Control this app actually found rather than guessed from its version. The
    app starts a model with `strata runtime start --wait`, and that wait is checked three times before the engine is reached: by
    management.runtime_action, by the operation record written before the start (journal), and by runtime.start itself. 27cbbfe6
    raised only the runtime's, so a longer wait died with `start wait must be finite seconds in 0..600` and then with
    `direct runtime intent has an invalid timeout`. The app therefore asks for the longer wait only when a Control's own sources
    say all three agree, and keeps today's 600 s whenever one of them still names a ceiling of its own. Reading the installed
    Control to learn what it supports is what the field list and the /v1/responses capability already do. */
 const START_WAIT_CEILING=600;
 let ceilingCache={dir:'',time:0,value:START_WAIT_CEILING};
 function followsRuntimeCeiling(text,marker){
  /* The marker picks the one line that bounds the wait (`"wait_s": (` in the record's range table, `"start wait"` in the CLI
     validator), because both files name those keys elsewhere. A line that still spells out 600 is a ceiling of its own. */
  const line=(String(text).split(/\r?\n/).find(l=>l.includes(marker))??'');
  return /runtime\.MAX_START_WAIT|start_wait_max/.test(line)&&!/\b600\b/.test(line);
 }
 function startWaitCeiling(){
  const dir=control?.control?.dir??'';if(!dir)return START_WAIT_CEILING;
  const files=['management.py','journal.py','runtime.py'].map(name=>join(dir,'strata_control',name));
  let time=0;
  try{time=Math.max(...files.map(f=>statSync(f).mtimeMs));}catch{return START_WAIT_CEILING;}
  if(ceilingCache.dir===dir&&ceilingCache.time===time)return ceilingCache.value;
  let value=START_WAIT_CEILING;
  try{
   const [management,journal]=files.slice(0,2).map(f=>readFileSync(f,'utf8'));
   // both gates must hand the ceiling to the runtime, or the wait would be refused on the way to it
   if(followsRuntimeCeiling(management,'"start wait"')&&followsRuntimeCeiling(journal,'"wait_s": (')){
    const declared=Number(readFileSync(files[2],'utf8').match(/^MAX_START_WAIT\s*=\s*(\d+)/m)?.[1]);
    if(Number.isFinite(declared)&&declared>value)value=declared;
   }
  }catch{value=START_WAIT_CEILING;}
  ceilingCache={dir,time,value};return value;
 }
 /* How long a start waits for this model, in the same shape as `strata launch`'s figure (#3949 #3459 #4191): ten minutes, about eight
    more seconds per GB above 40 GB, never more than this Control accepts. A model nobody can size keeps the 300 s it always had.
    The run config's engine_ready_s is the floor, not the ceiling: the server reads that number and ends the engine when it runs out,
    so the start must never give up before the server does (setup writes 3600 for DeepSeek V4.1-Flash). */
 function startWaitFor(config,entry,model){
  const ceiling=startWaitCeiling(),ready=Number(config?.engine_ready_s)||0;
  const known=Number(entry?.diskGb),fromDisk=Number.isFinite(known)&&known>0?known:0;
  const gb=fromDisk||Number((snapshot.catalog?.families??[]).flatMap(f=>f.sizes??[]).find(s=>s.id===model)?.download_gb)||0;
  const sized=gb>40?Math.max(300,600+(gb-40)*8):300;
  return Math.trunc(Math.max(300,Math.min(ceiling,Math.max(ready,sized))));
 }
 function buildConfig(args,config,drafter=null){
  /* `visionUnavailable`: this engine has no picture reader, so the run goes without pictures but nothing is saved: the model's own setting stays what the person had (it used to save "off", and the model stayed off after the engine got the reader). */
  const noReader=args.visionUnavailable===true;
  const vision=ownEngine(config)?undefined:runtimeVision(config,noReader?'no':args.vision); /* DeepSeek has its own picture tower: its picture setting is the --vision flag setup wrote */
  const visionPreferences=args.vision!==undefined&&!noReader?{ui_vision_enabled:args.vision!=='no',ui_vision_device:args.vision==='no'?config.ui_vision_device??'yes':args.vision,ui_vision_chosen:true}:{};/* ui_vision_chosen: this choice was made with a reader present (0.2.14); an older off without it may be the one 0.2.9 saved when the engine had no reader */
  /* The engine options the person set by hand (user_args, and what they wrote into "args" since the last write) go over everything the app generates, and an
     explicit GPU-memory reserve (the Expert setting, user_args, a value they wrote) is never replaced by the safety default (#3215 #3270 #3216). */
  const values0=args.values??{},cleared='vram_reserve_mib' in values0&&(values0.vram_reserve_mib===null||values0.vram_reserve_mib===''),base0=cleared?{...config,vram_reserve_mib:undefined,[USER_KEY]:setUserFlag(config[USER_KEY],'--vram-reserve-mib',null)}:config;
  let mine=captureUserArgs(base0);const asked=cleared?null:reserveChoice({...base0,[USER_KEY]:mine},snapshot.hardware,values0).asked;
  if(asked!==null)mine=setUserFlag(mine,'--vram-reserve-mib',asked);
  const resources=args.resourceProfile&&!ownEngine(config)?resourceSettings({...base0,[USER_KEY]:mine,vision},snapshot.hardware,args.resourceProfile,modelArena(args.model),values0):null;
  const cfg={...config,...visionPreferences,...(args.contextChosen===true?{ui_context_chosen:true}:{}),...(resources?{...resources,ui_resource_profile:args.resourceProfile}:{}),host:'127.0.0.1',open_browser:false,...validateNativeSettings(args.values??{}),vision};
  /* The app's suggested cards only fill a gap: a model whose settings file already names its GPUs ("gpu": [1, 0], with "gpu_order": "as_given") keeps them (#3256). A card the person picked in the screen always wins. */
  const namedInFile=((Array.isArray(config.gpu)?config.gpu.length>0:config.gpu!==undefined&&config.gpu!==null&&config.gpu!=='')||!!config.gpu_roles)&&!setupPickedOneCard(config,snapshot.hardware);
  /* Roles chosen in the screen (Main / Helper / Off per card, gpu-roles.mjs): the card list is the roles' mirror - Helpers first, the Main card LAST, Off cards left out - with "gpu_roles" beside it. Every card Off is allowed: the server says plainly what the engine can do. */
  if(args.gpuRoles!==undefined){const make=snapshot.hardware?.engine_vendor,cards=(snapshot.hardware?.gpus??[]).filter(g=>!make||!g.vendor||g.vendor===make),named=parseRoles(args.gpuRoles);if(!named)throw new Error('Choose Main, Helper or Off for each graphics card, with one Main.');const known=new Set(cards.map(g=>g.index));if(Object.keys(named).some(k=>!known.has(Number(k))))throw new Error('Choose detected graphics cards.');const roles=Object.fromEntries(cards.map(g=>[g.index,named[g.index]??'off'])),m=mirrorRoles(roles,cards);for(const k of ['gpu','gpu_order','gpu_none','gpu_roles','gpu_cards','hip_ordinal'])delete cfg[k];Object.assign(cfg,m);const count=Array.isArray(m.gpu)?m.gpu.length:m.gpu_none?0:1;cfg.layer_split=count>1?validateNativeSettings({layer_split:args.layerSplit??'auto'}).layer_split:'auto';if(cfg.layer_split!=='auto'&&cfg.layer_split.split(',').length!==count-1)throw new Error('Choose one layer split point between each pair of GPUs.');}
  else   if(args.gpus!==undefined&&!(args.gpusFromRecommendation&&namedInFile)){const detected=new Set((snapshot.hardware?.gpus??[]).map(g=>g.index));if(!Array.isArray(args.gpus)||!args.gpus.length||new Set(args.gpus).size!==args.gpus.length||args.gpus.some(g=>!Number.isInteger(g)||!detected.has(g)))throw new Error('Choose distinct detected GPUs. CPU-only text inference is not supported by this native adapter.');cfg.gpu=args.gpus.length===1?args.gpus[0]:args.gpus;for(const k of ['gpu_roles','gpu_cards','gpu_none'])delete cfg[k];delete cfg.hip_ordinal;cfg.layer_split=args.gpus.length>1?validateNativeSettings({layer_split:args.layerSplit??'auto'}).layer_split:'auto';if(cfg.layer_split!=='auto'&&cfg.layer_split.split(',').length!==args.gpus.length-1)throw new Error('Choose one layer split point between each pair of GPUs.');}
  if(cfg.vram_reserve_mib!==undefined&&cfg.vram_reserve_mib!==null){cfg.args=[...(cfg.args??[])];const at=cfg.args.indexOf('--vram-reserve-mib');if(at>=0)cfg.args.splice(at,2);cfg.args.push('--vram-reserve-mib',String(cfg.vram_reserve_mib));}
  if(mine.length){cfg.args=mergeUserArgs(cfg.args??[],mine);cfg[USER_KEY]=mine;}else delete cfg[USER_KEY];
  if(cfg.vision){if(cfg.vision_tokens!=null)cfg.vision={...cfg.vision,max_tokens:cfg.vision_tokens};if(cfg.vision_min_tokens!=null)cfg.vision={...cfg.vision,min_tokens:cfg.vision_min_tokens};}cfg.mcp={...(cfg.mcp??{}),...(cfg.mcp_timeout_s!=null?{timeout_s:cfg.mcp_timeout_s}:{}),...(cfg.mcp_max_rounds!=null?{max_rounds:cfg.mcp_max_rounds}:{}),...(cfg.mcp_max_result_chars!=null?{max_result_chars:cfg.mcp_max_result_chars}:{})};
  /* The model's own choices win over what setup wrote into the arguments: "Draft layer (MTP)" off leaves --mtp out of every start, and a picture reading that is off leaves --vision out (setup writes both again whenever it runs; the saved settings are applied here, on every load). */
  if((cfg.mtp===false||cfg.drafting==='off')&&Array.isArray(cfg.args))cfg.args=withoutDraftLayer(cfg.args);
  /* The Qwen engine's --vision stands alone, so it goes (with a picture word after it, if an older file has one). The DeepSeek engine's is a pair, `--vision on|off`, that setup wrote and the engine reads: it stays whole (taking out only the name left `off` behind: `--eos 1 off`, "unknown option off", #4358 #4359 #4505 #4507). */
  if(!cfg.vision&&!ownEngine(config)&&Array.isArray(cfg.args))cfg.args=withoutVisionFlag(cfg.args);
  if(ownEngine(config)&&Array.isArray(cfg.args)&&args.vision!==undefined)cfg.args=dsVisionPair(cfg.args,args.vision==='no'?'off':dsPictures(cfg)?'on':null);
  for(const arg of cfg.args??[])if(arg.endsWith('.gguf')&&!existsSync(resolve(repo(),arg)))throw new Error('An engine argument names a missing GGUF. Restore the saved arguments or repair the files.');
  /* The app's own choice of request slots (4, fewer where the card has no room) is part of the configuration, so a tune measures the engine a load will start: slots take VRAM from the expert cache. */
  const parallelChoice=packaged&&!ownEngine(config)?chooseParallel({config:cfg,values:args.values??{},drafter,hardware:snapshot.hardware?.gpus?.length?snapshot.hardware:{gpus:hardwareProbe()??[]}}):null;
  let parallelNote=null,dsNote='';
  /* The agent asks for long answers (a thinking model can use most of the window): let the engine clamp the output to what is left of the context instead of refusing the request. */
  if(packaged&&!ownEngine(config)&&cfg.fit_max_tokens===undefined)cfg.fit_max_tokens=true;
  if(parallelChoice){const applied=applyParallel(cfg,parallelChoice);for(const k of Object.keys(cfg))if(!(k in applied.cfg))delete cfg[k];Object.assign(cfg,applied.cfg);parallelNote=applied.note;}
  /* DeepSeek's Speed choice: the engine decides by itself (--experts auto: 5 when the cards total 16 GB or less, else 6), so nothing is passed unless the person chose 5 or 6. */
  if(ownEngine(config)){const list=[];const old=Array.isArray(cfg.args)?cfg.args:[];for(let i=0;i<old.length;i++){if(old[i]==='--experts'){i++;continue;}list.push(old[i]);}if(dsExpertsSupported(config)&&(cfg.ds_experts==='5'||cfg.ds_experts==='6'))list.push('--experts',cfg.ds_experts);else if(!dsExpertsSupported(config)&&old.includes('--experts'))list.splice(0,list.length,...old);cfg.args=list;delete cfg.ds_experts;
   /* Only the options strata-dsv41 takes go to it (#4504): --vram-reserve-mib is its --reserve-mib, and --pool-workers and the other Qwen-engine names are left out, in the arguments and in the person's saved options. */
   const built=dsEngineArgs(cfg.args);cfg.args=built.args;dsNote=dsArgsNote(built);
   if(Array.isArray(cfg[USER_KEY])){const kept=dsUserArgs(cfg[USER_KEY]);if(kept.length)cfg[USER_KEY]=kept;else delete cfg[USER_KEY];}}
  else delete cfg.ds_experts;
  return {cfg,resources,visionPreferences,parallelChoice,parallelNote,dsNote};
 }
 /* One drafter per run. Control (strata_control/drafting.py) says which one this load runs and whether its file is on this computer: the plan of the saved model file with the choice the Load carries. */
 /* The engines this Load is about to unload (the load below unloads every instance unless it adds one beside them): their graphics memory is given back before the new engine starts, so Control must not count it as busy memory when it asks whether the all-cards split engine fits (`--engines-unloading`; the Terminal does the same, Runner.card_probe). A model added beside the others (`multiple`) keeps their memory busy. */
 function unloadsRunning(args){return instances.size>0&&args?.multiple!==true;}
 async function draftPlan(args,configFile,{noSplit=false}={}){
  if(!packaged||typeof control.draftingPlan!=='function')return null;
  const asked=args.values?.drafting;
  try{return await withPlanFile(args,configFile,planFile=>control.draftingPlan({configPath:planFile,...(asked===undefined?{}:{mode:asked===null?'auto':asked}),...(noSplit?{noSplit:true}:{}),...(unloadsRunning(args)?{unloading:true}:{})}));}catch{return null;}
 }
 /* The plan is for the cards this Load uses: the screen's own card choice goes over the ones the saved settings name (the all-cards split engine needs two or more). `use` gets the file to hand to Control; a temporary one is removed after. */
 async function withPlanFile(args,configFile,use){
  let planFile=configFile,temporary=null;
  if(Array.isArray(args.gpus)&&args.gpus.length&&args.gpus.every(Number.isInteger)){try{const work=join(directory,'native-instances');mkdirSync(work,{recursive:true});temporary=join(work,'drafting-plan-'+randomUUID()+'.json');writeFileSync(temporary,JSON.stringify({...readJson(configFile),gpu:args.gpus.length===1?args.gpus[0]:args.gpus}),{mode:0o600});planFile=temporary;}catch{temporary=null;planFile=configFile;}}
  try{return await use(planFile);}
  finally{if(temporary){try{rmSync(temporary,{force:true});}catch{/* best effort */}}}
 }
 /* Auto and its drafter (the tuned one where the set has one, else the standard DFlash2 drafter: whatever Control's Auto plan says is `needed`), on every path that starts a model: a plain load, a restart of the stopped models and a tuning run fetch it as the Models screen's own Load does (Control: pinned, checked, marked, licence and notice), but they were not asked for it, so
    the 12 hour rule of the Terminal's launch applies (--throttle: a feed that had nothing a short while ago is not asked again, a feed that does not answer is not an error). A choice the person named or saved (MTP, DFlash2, off) wins: this never runs for it. */
 function autoChoice(value){return value==null||String(value).toLowerCase()==='auto';}
 async function fetchAutoDrafter(job,args,configFile,drafts,{noSplit=false,after}={}){
  if(!packaged||typeof control.draftingFetch!=='function'||drafts?.needed!==true||!autoChoice(readJson(configFile).drafting))return false;
  job.engineAbort??=new AbortController();
  try{
   const got=await withPlanFile(args,configFile,planFile=>control.draftingFetch({configPath:planFile,mode:'auto',throttle:true,...(noSplit?{noSplit:true}:{}),...(unloadsRunning(args)?{unloading:true}:{}),signal:job.engineAbort.signal,onEvent:event=>{if(event.event==='progress'&&job.state==='running'){job.summary=event.step;job.details=event.total>1&&event.done<event.total?`${(event.done/1e9).toFixed(2)} of ${(event.total/1e9).toFixed(2)} GB`:undefined;emit();}}}));
   if(job.state==='running'&&after){job.summary=after;job.details=undefined;emit();}
   return got?.pack!=null&&got?.skipped!==true;
  }catch(error){if(job.state==='running'){job.drafterNote=error.message;if(after){job.summary=after;job.details=undefined;}emit();}return false;}
 }
 /* The same for a path that has no Load of its own around it (a restart of stopped models, a tuning run): plan the saved settings file, then fetch. */
 async function autoDrafterFor(job,model,after){
  try{const {path}=recordFile(model),args={model,multiple:true},drafts=await draftPlan(args,path);return await fetchAutoDrafter(job,args,path,drafts,{after});}catch{return false;}
 }
 /* The choice changed to the MTP and its layer is not here yet: it is fetched once, by the installer (finished steps are skipped), in this Load. Only the Models screen's own Load carries the choice; a restart or a tuning run never fetches. */
 async function installMtpLayer(job,args,config){
  const fam=snapshot.catalog?.families?.find(f=>f.sizes.some(s=>s.id===args.model)),size=fam?.sizes.find(s=>s.id===args.model);
  if(!fam||!size)return;
  job.summary='Downloading the MTP drafter…';emit();
  /* the model's own files folder (the run config's --native): without it the installer, asked for this model by name, would fetch the whole model again next to the layer (an app install adopted from the person's own GGUF folder has its files elsewhere) */const native=(()=>{const a=(config.args??[]).map(String),i=a.indexOf('--native');return i>=0?(a[i+1]??''):'';})();try{await install(job,{family:fam.family,model:size.model,drafter:'mtp',vision:args.vision??(config.vision?'yes':'no'),...(native?{gguf_dir:dirname(native)}:{})});return true;}
  catch{if(job.state==='running'){job.drafterNote='The MTP drafter could not be downloaded.';emit();}return false;}/* one short line, not the installer's own text; Drafting below says what runs instead */
 }
 /* The DFlash2 drafter, before a load that uses it: Control says whether the choice needs it (and the file is not here), downloads it pinned and checked, and packs it, with plain progress in the job.
    Never stops the load: a drafter that cannot be had leaves the MTP running if its layer is here, else no drafter, and the job says so. The choice itself is remembered in the model's settings file. */
 async function prepareDrafter(job,args,cfg,configFile,drafts,{noSplit=false}={}){
  const asked=args.values?.drafting;
  if(asked!==undefined)rememberSetting(configFile,'drafting',asked===null||asked==='auto'?null:asked);
  /* Only the Models screen's own Load passes the choice (its dialog has listed the download): a restart, a tuning run or any other load leaves `asked` undefined and never fetches; the model then starts with the drafter that is on this computer and says why. */
 if(!packaged||typeof control.draftingFetch!=='function'||asked===undefined||drafts?.needed!==true)return;
  const work=join(directory,'native-instances');mkdirSync(work,{recursive:true});const probe=join(work,'drafting-'+randomUUID()+'.json');
  try{
   writeFileSync(probe,JSON.stringify(cfg),{mode:0o600});
   job.engineAbort??=new AbortController();
   await control.draftingFetch({configPath:probe,mode:cfg.drafting??'auto',...(noSplit?{noSplit:true}:{}),...(unloadsRunning(args)?{unloading:true}:{}),signal:job.engineAbort.signal,onEvent:event=>{if(event.event==='progress'&&job.state==='running'){job.summary=event.step;job.details=event.total>1&&event.done<event.total?`${(event.done/1e9).toFixed(2)} of ${(event.total/1e9).toFixed(2)} GB`:undefined;emit();}}});
   if(job.state==='running'){job.summary='Loading the model…';job.details=undefined;emit();}
  }catch(error){if(job.state==='running'){job.drafterNote=error.message;job.summary=drafts.mtp_here===false?'The DFlash2 drafter could not be downloaded, so the model starts without a drafter. Loading the model…':'The DFlash2 drafter could not be downloaded, so the model starts with the MTP. Loading the model…';emit();}}
  finally{try{rmSync(probe,{force:true});}catch{/* best effort */}}
 }
 /* One key of a model's settings file, set or removed (the Drafting choice): written like the Settings screen writes it, to be there for the next load. */
 function rememberSetting(configFile,key,value){try{const current=readJson(configFile);if(value===null?!(key in current):current[key]===value)return;const next={...current};if(value===null)delete next[key];else next[key]=value;const temp=configFile+'.ui-tmp';writeFileSync(temp,JSON.stringify(next,null,2)+'\n',{mode:0o600});renameSync(temp,configFile);}catch{/* the load goes on with the choice for this start */}}
 /* Which drafter runs, when it is not the one that was asked for (the engine's own words, "Drafting: MTP (DFlash2 didn't fit)"): under Details of the load. */
 async function noteDrafting(job,instance,key){let note='',fellBack=false;try{const r=await fetcher(`http://127.0.0.1:${instance.port}/metrics`,{headers:{authorization:`Bearer ${key}`},signal:AbortSignal.timeout(2000)});if(r.ok){const engine=(await r.json())?.engine;note=String(engine?.drafting_note??'');fellBack=engine?.split_fallback===true;}}catch{/* a convenience */}/* the all-cards split engine: the line of what runs (the server starts the normal layer split by itself when the split engine would not come up) */const lines=[job.drafterNote,fellBack?SPLIT_FELL_BACK:instance.splitNote,note].filter(Boolean);if(lines.length)job.details=[job.details,...lines].filter(Boolean).join('\n');}
 /* ---- The picture files of an installed model that has none (strata_control/pictures.py, run by Control) ----
    Only the Load or the "Download picture-reading files" button the person pressed reaches the network, and only for the picture file (about 0.9 GB, one copy
    shared by the models of a data folder). The model's own files, its pack, its draft layer, the data folder and the settings' data_dir are not touched; the one change
    to the model's run config is the picture section. A restart, a tuning run or any other load that does not ask for pictures never downloads. */
 const PICTURES_WORDS={download:'Downloading the picture files (0.9 GB, once)…',check:'Checking the picture files…',loading:'Loading the model…',
  offline:'No internet connection: the model is loaded without picture reading. Load it again when you are online.',
  disk:'There is not enough free disk space for the picture files, so the model is loaded without picture reading.',
  failed:'The picture files could not be downloaded, so the model is loaded without picture reading. Try again later.',
  gone:'The picture files are missing, so the model is loaded without picture reading.'};
 const PLAIN_PICTURE_CODES=['offline','disk','failed','checksum','unsupported','no_reader'];
 const askedPictures=value=>['yes','gpu','cpu'].includes(value);
 const pictureFileGone=config=>!config.vision?.mmproj||!existsSync(resolve(repo(),String(config.vision.mmproj)));
 /* The card the picture reader will be chosen for: the cards the model runs on, the one with the most memory (setup's rule is the card's memory: the CPU reader under 24 GB). */
 function pictureCard(config){const make=snapshot.hardware?.engine_vendor,named=Array.isArray(config.gpu)?config.gpu:config.gpu===undefined||config.gpu===null?null:[config.gpu];const cards=(snapshot.hardware?.gpus??[]).filter(g=>g.usable!==false&&(!make||!g.vendor||g.vendor===make)&&(!named||named.includes(g.index)));const best=[...cards].sort((a,b)=>(Number(b.vram_gb)||0)-(Number(a.vram_gb)||0))[0];return best?{vram_gb:Number(best.vram_gb)||0,make:best.vendor??make??null,arch:best.arch??null}:null;}
 async function ensurePictures(job,args,configFile,config,{strict=false,missing=false}={}){
  const keep={config,args,note:null};
  if(!packaged||ownEngine(config)||args.visionUnavailable===true)return keep;
  const prepared=!!config.vision,asked=askedPictures(args.vision);
  /* Control says the picture file is gone (missing) or, when pictures were asked for, this look at the disk says so; a load that asked for nothing never second-guesses Control. */
  if(prepared&&!missing&&!(asked&&pictureFileGone(config)))return keep;
  if(!asked){const none={...args,vision:'no',visionUnavailable:true};return prepared?{config,args:none,note:PICTURES_WORDS.gone}:keep;}
  if(typeof control.picturesFetch!=='function')return keep;
  job.abort=new AbortController();job.summary=PICTURES_WORDS.download;job.details=undefined;emit();
  try{
   const engine=await control.engine();const engineDir=engine?.engine?.root;
   if(!engineDir)throw Object.assign(new Error('This engine has no picture reader.'),{code:'no_reader'});
   const family=snapshot.catalog?.families?.find(f=>f.sizes?.some(s=>s.id===args.model))?.family??null;
   const here=snapshot.installed?.models?.find(m=>m.model===args.model);
   let fallbackData=null;try{fallbackData=here?.runDataFolder??here?.dataFolder??nativeDataDirectory(preferences.downloadDirectory??modelsDirectory?.())??null;}catch{fallbackData=here?.runDataFolder??here?.dataFolder??null;}
   const got=await control.picturesFetch({configPath:configFile,engineDir,vision:args.vision,dataDir:fallbackData,family,hardware:pictureCard(config),signal:job.abort.signal,
    onEvent:event=>{if(event?.event!=='progress'||job.state!=='running')return;const total=Number(event.total)||0,done=Number(event.done)||0;
     if(/^Checking/i.test(String(event.step))){job.summary=PICTURES_WORDS.check;}
     else{job.summary=PICTURES_WORDS.download;if(total>0)job.result={install:{download:{file:'picture reader',done_gb:done/1e9,total_gb:total/1e9,remaining_gb:Math.max(0,total-done)/1e9,percent:Math.min(99,Math.floor(100*done/total))}}};}
     emit();}});
   if(job.state!=='running')return keep;
   const next={...config,vision:got.vision};
   writeModelConfig(configFile,next);
   if(got.note)append(args.model,`Picture files: ${got.note}`);
   job.result=null;job.summary=PICTURES_WORDS.loading;job.details=undefined;emit();
   return {config:next,args,note:null};
  }catch(error){
   job.result=null;
   if(job.state!=='running')return keep;
   const code=error?.code,details=error?.output?.error?.details;
   append(args.model,`Picture files: ${error?.message??error}${details?` (${details})`:''}`);
   if(strict)throw Object.assign(new Error(PLAIN_PICTURE_CODES.includes(code)?String(error.message):'The picture files could not be downloaded. Try again.'),{details:details?[details]:undefined});
   job.summary=PICTURES_WORDS.loading;job.details=undefined;emit();
   return {config,args:{...args,vision:'no',visionUnavailable:true},note:code==='offline'?PICTURES_WORDS.offline:code==='disk'?PICTURES_WORDS.disk:PICTURES_WORDS.failed};
  }
 }
 /* "Download picture-reading files": the same step without a load. */
 async function picturesOnly(job,args){
  await refresh(true);if(job.state!=='running')return;
  const {path:configFile}=recordFile(args.model);const config=checkedSettingsConfig(readFileSync(configFile,'utf8'));
  if(!packaged)throw new Error('The picture files are downloaded by the installed app.');
  if(ownEngine(config))throw new Error('This model reads pictures with its own files.');
  const {entry}=recordFile(args.model);
  if(config.vision&&!entry.picturesMissing&&!pictureFileGone(config))return {summary:'The picture files are ready.'};
  await ensurePictures(job,{...args,vision:askedPictures(args.vision)?args.vision:'yes'},configFile,config,{strict:true,missing:entry.picturesMissing===true});
  if(job.state!=='running')return;
  await refresh(true);
  return {summary:'The picture files are ready.'};
 }
 async function loadEngine(job,args){
  repairAllControlPaths();await refresh(true);if(job.state!=='running')return;const {entry,path:configFile}=recordFile(args.model);let config=checkedSettingsConfig(readFileSync(configFile,'utf8'));if(packaged){const engine=await control.engine();if(!engine.usable)throw new Error(ENGINE_NOT_RECOGNISED+(engine.notice?` ${engine.notice}`:''));}if(!entry.ready){const error=missingFilesError(args.model,entry.missing);append(args.model,error.details[0]);throw error;}
  if(externalRunning().length)throw new Error('A Strata server is already running outside this UI. Stop it before loading here, or connect its API in Settings → Models.');
  if(args.multiple&&instances.size>=8)throw new Error('Unload an instance before adding another. Up to eight native instances can run here.');
  /* Read pictures is on and the picture files are not here: the Load fetches those files (about 0.9 GB, once) and nothing else of the model, then goes on. No internet: the model loads without pictures and the page says so. */
  let picturesNote=null;{const got=await ensurePictures(job,args,configFile,config,{missing:entry.picturesMissing===true});if(job.state!=='running')return;config=got.config;args=got.args;picturesNote=got.note;}
  /* One drafter per run (strata_control/drafting.py): switching to the MTP on a model set up for DFlash2 fetches the layer once, here; DFlash2 is fetched below. */
  let drafts=await draftPlan(args,configFile),noSplit=false;
  if(drafts?.mtp_needed===true&&(args.values?.drafting!==undefined||(drafts.split===true||args.firstLoad===true)&&(config.drafting==null||String(config.drafting).toLowerCase()==='auto'))){/* firstLoad: the load right after a setup (Prepare and load), whose plan listed the layer: Auto meaning the MTP on this card fetches it, as the Models screen does (QAGUI 0213) *//* the Models screen's own Load carries the choice; a plain Auto load fetches the layer too when Control says the all-cards split engine will take the start (it runs the MTP and nothing else): the same as the Terminal */await installMtpLayer(job,args,config);if(job.state!=='running')return;await refresh(true);config=checkedSettingsConfig(readFileSync(recordFile(args.model).path,'utf8'));drafts=await draftPlan(args,configFile);
   /* The draft layer for the split could not be had (no internet, disk): the load goes on as the 0.2.10 app did, the normal layer split with the plan DFlash2's again (the Models screen's own Load fetches that drafter below). One plain line says so. */
   if(drafts?.split===true&&drafts.mtp_needed===true){noSplit=true;job.drafterNote=job.drafterNote??'The MTP drafter could not be downloaded.';job.summary='Loading the model…';emit();drafts=await draftPlan(args,configFile,{noSplit:true});}}
  /* A plain load (the Models screen's own Load carries its choice and fetches below): Auto on a set that has the tuned drafter fetches it now, so the start uses it. Not on this computer and not reachable: the MTP or the stock drafter runs as before. */
  if(args.values?.drafting===undefined&&drafts?.needed===true){const got=await fetchAutoDrafter(job,args,configFile,drafts,{noSplit,after:'Loading the model…'});if(job.state!=='running')return;if(got)drafts=await draftPlan(args,configFile,{noSplit});}
  const {cfg,resources,visionPreferences,parallelChoice,parallelNote:chosenNote,dsNote}=buildConfig(args,config,drafts?.choice==='dflash2'&&drafts.needed===true&&args.values?.drafting===undefined?'mtp':drafts?.choice??null);/* (a drafter that is not here and will not be fetched by this load: the MTP runs, and the request slots stay) */
  let parallelNote=chosenNote;
  await prepareDrafter(job,args,cfg,configFile,drafts,{noSplit});if(job.state!=='running')return;
  const fingerprint=createHash('sha256').update(JSON.stringify({...cfg,api_key:undefined,port:undefined})).digest('hex');
  const placement=Array.isArray(cfg.gpu)?cfg.gpu:cfg.gpu===undefined?[]:[cfg.gpu];const matching=[...instances.values()].find(i=>i.model===args.model&&JSON.stringify(i.gpus)===JSON.stringify(placement));
  if(matching?.status==='ready'&&matching.fingerprint===fingerprint&&(args.multiple||instances.size===1))return {summary:`${matching.label} is already ready.`,instanceId:matching.id};
  const wasDefault=matching?.id===defaultId;
  const id=matching?.id??randomUUID();let apiModelId=matching?.apiModelId??args.model,suffix=1;while(!matching&&[...instances.values()].some(i=>i.apiModelId===apiModelId))apiModelId=`${args.model}--${++suffix}`;
  cfg.aliases=[...new Set([...(Array.isArray(cfg.aliases)?cfg.aliases:typeof cfg.aliases==='string'?cfg.aliases.split(',').map(s=>s.trim()).filter(Boolean):[]),apiModelId])];
  const key=randomBytes(32).toString('hex');cfg.api_key=key;cfg.port=await freePort(args.port);
  const work=join(directory,'native-instances');mkdirSync(work,{recursive:true});const configPath=join(work,`${id}.json`);
  const py=packaged?null:venvPython(repo());if(!packaged&&!existsSync(py))throw new Error('The Strata Python environment is missing. Use Download / repair first.');
  if(job.state!=='running')return;
  await beforeLoad();if(job.state!=='running')return;
  if(!args.multiple){if(busy(requests().filter(r=>instances.has(r.instanceId))))throw new Error('Finish or cancel native requests before switching models.');for(const id of [...instances.keys()])await unload(id);}else if(matching)await unload(matching.id);
  if(job.state!=='running')return;
  writeFileSync(configPath,JSON.stringify(cfg,null,2),{mode:0o600});
  const instance={id,apiModelId,fingerprint,context:entry.context??0,aliases:cfg.aliases??[],vision:!!cfg.vision,resourceProfile:cfg.ui_resource_profile??null,model:args.model,label:friendlyModelName(snapshot.catalog,args.model)??entry.model_name??args.model,port:cfg.port,gpus:Array.isArray(cfg.gpu)?cfg.gpu:cfg.gpu===undefined?[]:[cfg.gpu],status:'loading',phase:'loading',parallel:cfg.parallel??1,mcpServers:cfg.mcp_servers??cfg.mcpServers??{},key,configPath,loadedAt:new Date().toISOString(),metrics:null,error:null,...(picturesNote?{picturesNote}:{})};
  /* Another model on the same graphics card: the engine takes the card's free memory, so a second one usually finds none. Loading goes on (recommend, never force); if it fails, the reason is said. */
  const sharedWith=instance.gpus.length?[...instances.values()].find(i=>i.id!==id&&i.status!=='error'&&i.model!==args.model&&i.gpus.some(g=>instance.gpus.includes(g))):null;
  const sharedSentence=sharedWith?`GPU ${instance.gpus.find(g=>sharedWith.gpus.includes(g))+1} is already used by ${sharedWith.label}, so this model did not fit there. Load it on another GPU, or unload ${sharedWith.label} first.`:null;
  let proc=null,started=null,retried=null,startWait=null,adopted=false;
  if(packaged){
   /* How long this start waits, and the server's own engine wait kept above it. The server ends the engine when its engine wait runs
      out (900 s, or the config's engine_ready_s), so a start that waits longer than that can never win: as in `strata launch`, the
      server gets a wait above the start's. It is only ever raised, never lowered, so a config that already names a longer one
      (setup's 3600 for DeepSeek V4.1-Flash) is left alone. */
   startWait=startWaitFor(config,entry,args.model);
   /* The split engine reads every expert into the cards before it answers, and a start that falls back to the layer split begins again: the 10 minutes the Terminal gives a launch (never above the ceiling). */
   if(!noSplit&&drafts?.split===true)startWait=Math.min(startWaitCeiling(),Math.max(startWait,600));
   if(startWait+60>Math.max(900,Number(cfg.engine_ready_s)||0))cfg.engine_ready_s=startWait+60;
   /* The engine starts under Control's owner runtime, not as our child: a saved profile per slot, then `runtime start`. */
   const used=new Set([...instances.values()].map(i=>i.profile));let slot=1;while(used.has('app-'+slot))slot++;instance.profile='app-'+slot;
   instances.set(id,instance);if(!defaultId||!args.multiple||wasDefault)defaultId=id;emit();
   if(dsNote)append(id,dsNote);
   /* The all-cards split engine: where the plan says it takes this start, Control turns the run config into the split engine's (the step `strata launch` does itself: the rule, the engine package and the draft layer are checked on this computer) and says so in one line. Any "no" leaves the layer split config as it is. The second try after a failed start is never split. */
   const splitStart=async value=>{
    if(noSplit||drafts?.split!==true||typeof control.splitEngine!=='function')return value;
    const input=join(work,id+'.layer.json'),output=join(work,id+'.split.json');
    try{
     writeFileSync(input,JSON.stringify(value,null,2),{mode:0o600});rmSync(output,{force:true});
     const answer=await control.splitEngine({configPath:input,outPath:output});
     if(answer.line){instance.splitNote=answer.line;append(id,answer.line);}
     if(answer.split===true&&answer.written)return JSON.parse(readFileSync(output,'utf8'));
    }catch{/* the layer split config stands */}
    finally{for(const f of [input,output]){try{rmSync(f,{force:true});}catch{/* best effort */}}}
    return value;
   };
   const startWith=async(value,split=true)=>{const run=split?await splitStart(value):value;writeFileSync(configPath,JSON.stringify(run,null,2),{mode:0o600});return control.stop(instance.profile).catch(()=>{}).then(()=>control.saveProfile(instance.profile,configPath)).then(()=>control.start(instance.profile,{waitSeconds:startWait}));};
   /* A Control whose wait ran out has NOT stopped the engine: `runtime start` answers "startup_wait_expired" (exit 8), or "runtime operation wait expired" when the owner was too busy to answer in time, while the engine is still loading (#5177 #4293). That is not a failure: Control is asked, and a start it still reports as starting (or ready) is kept and watched below instead of being stopped and marked failed. */
   const stillLoading=async e=>{if(!/startup_wait_expired|runtime operation wait expired/i.test(String(e?.message??'')+JSON.stringify(e?.output??'')))return false;if(!instance.profile||typeof control.status!=='function'||job.state!=='running')return false;try{const st=await control.status(instance.profile);return st?.state==='starting'||st?.state==='ready';}catch{return false;}};
   started=startWith(cfg);
   started.then(()=>append(id,'Strata Control reports the engine is ready.'),error=>{append(id,error.message);});
   try{await started;}catch(firstError){let error=firstError;
    /* A settings, profile or run file an elevated run made can be closed to this user: repair the access in Strata's own folders and start again (this is not the memory try below). */
    const isAccess=e=>/PermissionError|access is denied|EACCES/i.test(String(e?.message??'')+JSON.stringify(e?.output??''));
    if(isAccess(error)&&typeof control.repairAccess==='function'&&job.state==='running'){await control.repairAccess();append(id,'Strata repaired access to its own files; starting again.');try{await startWith(cfg);started=Promise.resolve();error=null;}catch(again){error=again;}}
    if(error&&await stillLoading(error)){adopted=true;error=null;}
    if(error&&sharedSentence&&!isAccess(error)){instance.status='error';instance.error=sharedSentence;emit();throw new Error(sharedSentence);}
    if(error){const said=humanStartError(error);
    if(isAccess(error)){instance.status='error';instance.error=said;emit();throw new Error(said);}   // access errors are not memory stops: no safer retry
    /* ONE automatic second try in all, never saved, never repeated (args.retrySafer false turns it off): with a safer memory setting when the
       failure looks like a memory stop, and without several-requests-at-once when that was the app's own default (not a number the user chose).
       Both changes ride on the same single try, with the plain sentence of the memory change when there is one. */
    const allowed=args.retrySafer!==false&&job.state==='running'&&!ownEngine(config);
    const safer=allowed?saferStart(cfg,snapshot.hardware,modelArena(args.model),said):null;
    const dropParallel=allowed&&cfg.parallel>1&&!parallelChoice?.explicit;
    if(!safer&&!dropParallel){instance.status='error';instance.error=said;emit();throw new Error(said);}
    let retryConfig=safer?safer.config:cfg;if(dropParallel)retryConfig=withoutParallel(retryConfig);
    const words=safer?safer.words:FALLBACK_NOTE,detail=[safer?.detail,dropParallel?'one request at a time':null].filter(Boolean).join('; ');
    retried={words,detail};if(dropParallel){instance.parallel=1;if(!safer)parallelNote=FALLBACK_NOTE;}
    append(id,`The first start stopped (${said}). Starting once more: ${detail}`);job.summary=words;job.details=[said,`Second try: ${detail}`];instance.notice=words;emit();
    delete instance.splitNote;
    try{await startWith(retryConfig,false);append(id,'Strata Control reports the engine is ready (second try).');}
    catch(second){if(await stillLoading(second))adopted=true;else{const said2=humanStartError(second);instance.status='error';instance.error=said2;/* the first stop and the second try stay in job.details; the pop-up shows the last stop once */emit();throw new Error(instance.error);}}}}
  } else {
  proc=spawnProcess(py,[join(repo(),'serve/server.py'),'--engine','strata','--config',configPath,'--port',String(cfg.port),'--host','127.0.0.1'],{cwd:repo(),detached:process.platform!=='win32',stdio:['ignore','pipe','pipe'],env:sourceServerEnvironment()});
  instances.set(id,instance);processes.set(id,proc);if(!defaultId||!args.multiple||wasDefault)defaultId=id;
  for(const stream of [proc.stdout,proc.stderr])stream.on('data',chunk=>{const lines=chunk.toString().split('\n').filter(Boolean);lines.forEach(line=>append(id,line.replaceAll(key,'[redacted]')));job.summary=lines.at(-1)??job.summary;emit();});
  proc.on('error',error=>{instance.status='error';instance.error=error.message;emit();});proc.on('exit',code=>{if(instances.has(id)&&instance.status!=='unloading'){instance.status='error';instance.error=`Engine stopped${code===null?'':` (exit ${code})`}. Open diagnostics or reload.`;emit();}});emit();
  }
  /* A start Control stopped waiting for is watched for as long as the start asked for (the server ends the engine itself when its own wait runs out, a minute after this one), and Control is asked now and then whether the engine is still there. */
  const deadline=Date.now()+(adopted?Math.max(300,startWait??300)*1000:300000);let lastAsked=0;while(Date.now()<deadline&&job.state==='running'){
   if(instance.status==='error')throw new Error(instance.error);
   if(adopted&&instance.profile&&Date.now()-lastAsked>10000){lastAsked=Date.now();const st=await control.status(instance.profile).catch(()=>null);if(st&&['failed','stopped','stale'].includes(st.state)){instance.status='error';instance.error=humanStartError({message:st.error||'server_exited',output:st});emit();throw new Error(instance.error);}}
   try{const response=await fetcher(`http://127.0.0.1:${instance.port}/health`,{headers:{authorization:`Bearer ${key}`},signal:AbortSignal.timeout(2000)});if(job.state!=='running'){await unload(id);return;}if(response.ok){instance.status='ready';instance.phase='idle';await noteDrafting(job,instance,key);if(resources||args.vision!==undefined||args.contextChosen===true||JSON.stringify(cfg[USER_KEY]??null)!==JSON.stringify(config[USER_KEY]??null))writeModelConfig(record(args.model).path,{...config,...(resources?{...resources,ui_resource_profile:args.resourceProfile}:{}),...visionPreferences,...(args.contextChosen===true?{ui_context_chosen:true}:{}),...(cfg[USER_KEY]?{[USER_KEY]:cfg[USER_KEY]}:{})});emit();ensurePoll();instance.parallelNote=parallelNote;return {summary:retried?`${instance.label} is ready, after a second try: ${retried.words}`:`${instance.label} is ready.${parallelNote?' '+parallelNote:''}${picturesNote?' '+picturesNote:''}`,instanceId:id};}}catch{}
   await new Promise(resolve=>setTimeout(resolve,1200));
  }
  if(job.state!=='running'){await unload(id);return;}
  if(instance.profile)await control.stop(instance.profile).catch(()=>{});else await stopProcess(proc);instance.status='error';instance.error='Loading took more than five minutes. Check the engine log and free memory.';throw new Error(instance.error);
 }
 async function freePort(requested){for(let port=Number(requested??18500);port<=Number(requested??18500)+20;port++){if(!Number.isInteger(port)||port<1024||port>65535)throw new Error('Choose a port from 1024 to 65535.');if([...instances.values()].some(i=>i.port===port)){if(requested)throw new Error('This private engine port is already in use.');continue;}const available=await new Promise(resolve=>{const probe=createServer();probe.once('error',()=>resolve(false));probe.listen(port,'127.0.0.1',()=>probe.close(()=>resolve(true)));});if(available)return port;if(requested)throw new Error('This private engine port is already in use.');}throw new Error('No free private engine port was found.');}
 let pollCount=0;
 /* The engine's metrics carry every finished request with `?requests=all`: a long session makes that a large JSON that the host parses and
    pushes to the window every 2 s. The last dozen is what the app shows live; the full list is refreshed every tenth poll. */
 async function poll(){if(polling||closing)return;polling=true;const full=pollCount++%10===0;try{await Promise.all([...instances.values()].map(async instance=>{if(instance.status!=='ready')return;try{const response=await fetcher(`http://127.0.0.1:${instance.port}/metrics${full?'?requests=all':''}`,{headers:{authorization:`Bearer ${instance.key}`},signal:AbortSignal.timeout(1500)});if(!response.ok)throw new Error('Metrics unavailable');const fresh=await response.json();const older=instance.metrics?.requests,recent=fresh.requests;instance.metrics=full||!Array.isArray(older)||!Array.isArray(recent)||!recent.length?fresh:{...fresh,requests:[...recent,...older.filter(r=>Number(r.time)<Math.min(...recent.map(x=>Number(x.time))))]};const tools=await fetcher(`http://127.0.0.1:${instance.port}/mcp`,{headers:{authorization:`Bearer ${instance.key}`},signal:AbortSignal.timeout(1500)});if(tools.ok)instance.metrics.mcp=await tools.json();instance.phase=instance.metrics.live?.state??'idle';instance.misses=0;}catch{instance.metrics=null;instance.phase='unavailable';instance.misses=(instance.misses??0)+1;await reconcileLost(instance);}}));emit();}finally{polling=false;}}
 /* The engine of a ready instance stopped answering. After a few looks in a row Control is asked: an owner that says the engine is stopped, failed or lost ends the "ready" the app was still showing (the model looked loaded for good while every request failed, #4973); an engine that is only busy stays as it is. */
 async function reconcileLost(instance){if(instance.misses<3||instance.status!=='ready'||!instance.profile||typeof control.status!=='function')return;let st=null;try{st=await control.status(instance.profile);}catch{return;}if(instance.status==='ready'&&['stopped','failed','stale'].includes(st?.state)){instance.status='error';instance.error='Engine stopped. Open diagnostics or reload.';}}
 function ensurePoll(){if(!timer){timer=setInterval(()=>void poll(),2000);timer.unref?.();}void poll();}
 async function benchmark(job,args){const instance=resolveInstance(args.instanceId);job.instanceId=instance.id;if(busy(requests().filter(r=>r.instanceId===instance.id)))throw new Error('Wait for this model to finish before benchmarking.');const started=Date.now();job.abort=new AbortController();const response=await fetcher(`http://127.0.0.1:${instance.port}/v1/chat/completions`,{method:'POST',headers:{'content-type':'application/json',authorization:`Bearer ${instance.key}`},signal:AbortSignal.any([job.abort.signal,AbortSignal.timeout(180000)]),body:JSON.stringify({model:instance.apiModelId,messages:[{role:'user',content:'Explain how a rainbow forms in clear, simple language.'}],temperature:0,reasoning_effort:'none',max_tokens:200,stream:false})});const body=await response.json();if(!response.ok)throw new Error(body.error?.message??'Benchmark failed.');if(job.state!=='running')return;const result={summary:'Measured benchmark complete.',at:new Date().toISOString(),model:instance.apiModelId,durationMs:Date.now()-started,tokensIn:body.usage?.prompt_tokens??null,tokensOut:body.usage?.completion_tokens??null,outputTps:body.timings?.predicted_per_second??null,promptTps:body.timings?.prompt_per_second??null,source:'engine',settings:'Greedy · thinking off · up to 200 output tokens'};writeFileSync(join(directory,'native-benchmark.json'),JSON.stringify(result,null,2),{mode:0o600});return result;}
 function resolveInstance(id){const aliases=[...instances.values()].filter(i=>i.aliases?.includes(id));if(aliases.length>1&&!instances.has(id)&&![...instances.values()].some(i=>i.apiModelId===id))throw new Error('This alias matches more than one instance. Choose its exact model ID.');const i=instances.get(id)??[...instances.values()].find(i=>i.apiModelId===id)??aliases[0]??(!id&&instances.size===1?[...instances.values()][0]:undefined);if(!i)throw new Error('Choose a loaded model ID.');if(i.status!=='ready')throw new Error('This model is not ready.');return i;}
 async function program(job,args){if(packaged)return componentProgram(job,args);if(externalRunning().length)throw new Error('Stop the native server running outside this UI before changing engine files.');if(instances.size)throw new Error('Unload native models before changing engine files.');const folder=repo();let {command,args:prefix}=python(),argv;
  if(job.kind==='calibrate'||job.kind==='tune'){if(!state().capabilities.calibration)throw new Error('This installation calibrates NVIDIA GPUs only. Use the short benchmark on other devices.');const model=args.model??snapshot.installed?.models?.[0]?.model;const chosen=record(model);if(!chosen.entry.ready)throw new Error('Prepare this model before calibrating.');command=venvPython(folder);prefix=[];if(!existsSync(command))throw new Error('Prepare the Python environment first.');argv=['-c','import sys; from pathlib import Path; import setup; sys.exit(0 if setup.calibrate_config(Path(sys.argv[1])) else 1)',chosen.path];}
  else if(job.kind==='rollback'){if(!previousEngine())throw new Error('No saved previous engine is available.');argv=[join(folder,'setup.py'),'--rollback-engine'];}
  else if(job.kind==='update-engine'){if(!snapshot.installed?.models?.length)throw new Error('Prepare a model first. Its installer downloads the matching engine.');argv=[join(folder,'setup.py'),'--update'];}
  else if(job.kind==='update-strata'){const status=await execute('git',['status','--porcelain'],folder);if(status.trim())throw new Error('This Strata checkout has local edits. Save or commit them before updating.');const remote=await execute('git',['remote','get-url','origin'],folder);if(!/^https:\/\/github\.com\/Niko1221\/Strata(?:\.git)?\s*$/.test(remote))throw new Error('The native installation must use the official Strata repository to update here.');await execute('git',['pull','--ff-only'],folder);argv=[join(folder,'setup.py'),'--update'];}
  else throw new Error('Unsupported native operation.');
  return new Promise((resolve,reject)=>{const proc=spawnProcess(command,[...prefix,...argv],{cwd:folder,windowsHide:true,detached:process.platform!=='win32',stdio:['ignore','pipe','pipe'],env:{...process.env,PYTHONUNBUFFERED:'1',PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8'}});job.process=proc;for(const stream of [proc.stdout,proc.stderr])stream.on('data',chunk=>{chunk.toString().split('\n').filter(Boolean).forEach(line=>append(job.id,line));job.summary=chunk.toString().trim().slice(-500);emit();});proc.on('error',reject);proc.on('exit',code=>{delete job.process;code===0?resolve({summary:job.kind==='calibrate'?'Calibration complete. Native settings were saved.':'Native installation updated.'}):reject(new Error(`Operation stopped (exit ${code}). Open diagnostics for details.`));});});
 }
 /* Packaged app: engine and Control updates and the engine rollback go through Control's explicit update step (`strata update`,
    `strata components rollback`), never through git, setup.py or a download in this process. The app itself is updated by the
    shell's signed Tauri updater. Models are not part of any of it. */
 async function componentProgram(job,args={}){
  if(!COMPONENT_KINDS.includes(job.kind)||typeof control.update!=='function')throw new Error(NO_INSTALL);
  repairAllControlPaths();
  if(instances.size&&!(job.kind==='update-engine'&&args.stopModel===true))throw new Error('Unload native models before changing engine files.');
  job.abort=new AbortController();
  const subject=job.kind==='update-strata'?'Strata':'engine';
  try{return await withUpdateLock(job,()=>componentSteps(job,subject,args));}catch(error){updateLog(`${job.kind} failed: ${String(error?.message??error).slice(0,400)}`);throw error?.details!==undefined&&error.message?error:plainError(error,subject,args?.firstRun===true?'Continue':undefined);}
 }
 /* Control's update lock is held by another update or a check (the app's own check at start holds it for a few seconds: QAGUI 0213 finding 3). The step waits for it, up to
    LOCK_WAIT_MS, instead of failing at once; nothing was changed by a refused try, so trying again is safe. */
 async function withUpdateLock(job,run){
  const until=Date.now()+lockWaitMs;
  for(;;){
   try{return await run();}
   catch(error){
    const busy=error?.code==='update_busy'||UPDATE_LOCK_HELD.test(String(error?.message??''));
    if(!busy||Date.now()>=until||job.state!=='running'||job.abort?.signal.aborted)throw error;
    job.summary=UPDATE_WAITING_FOR_LOCK;emit();
    await new Promise(r=>setTimeout(r,lockPollMs));
    if(job.state!=='running')throw error;
   }
  }
 }
 /* The first-run card's Download also brings a newer Strata Control the feed offers (the engine first: a Control that cannot be had never costs the engine). Then the app
    uses it at once instead of after the next start, but only while nothing else runs (no model, no other job, no call to Control in flight); anything that goes wrong
    leaves today's behaviour: the new Control takes effect at the next start. Never fails the job. */
 async function firstRunControl(job,onEvent){
  try{await control.update({engine:false,control:true,onEvent,signal:job.abort.signal});updateLog('first run: Control updated');}
  catch(error){updateLog(`first run: the Control update did not finish: ${String(error?.message??error).slice(0,300)}`);return;}
  switchControl();
 }
 function switchControl(){
  try{
   if(instances.size||pending.size||connecting||[...jobs.values()].some(j=>j.state==='running'&&!COMPONENT_KINDS.includes(j.kind))||typeof control.reload!=='function')return false;
   const before=control.control.dir;
   if(!control.reload())return false;
   disconnect();sourceCache={path:'',time:0,text:''};lastRead=0;dflash2Served=null;
   updateLog(`now using Strata Control ${control.version??"?"} (${control.control.dir}, was ${before})`);
   return true;
  }catch(error){updateLog(`Control switch skipped: ${String(error?.message??error).slice(0,200)}`);return false;}
 }
 async function componentSteps(job,subject,args={}){
  const onEvent=event=>{
   if(event.event==='progress'){job.summary=`Downloading: ${event.percent}%`;job.result={install:{step:job.summary,download:{file:event.label,percent:event.percent}}};}
   else if(event.event==='phase'&&PHASE_TEXT[event.phase])job.summary=PHASE_TEXT[event.phase];
   else if(event.event==='waiting')job.summary='Waiting for the running model to stop…';
   emit();
  };
  updateLog(`${job.kind} started (app ${thisAppVersion()??'unknown'})`);
  if(job.kind==='install-engine-file'){if(typeof args.path!=='string'||!args.path)throw new Error('Choose the engine file first.');let done;try{done=await control.installEngineFile(args.path,{onEvent,signal:job.abort.signal});}finally{try{const own=resolve(directory,'incoming-engine')+sep;if(resolve(args.path).startsWith(own))rmSync(args.path,{force:true});}catch{/* the copy in the host's own folder; a leftover is harmless */}}return {summary:updateSummary(done)==='Everything is already up to date.'?(done.results?.[0]?.note??'Nothing was changed.'):updateSummary(done),result:done};}
  if(job.kind==='rollback'){const done=await control.rollbackEngine();return {summary:'The previous engine is back. Your models and settings are unchanged.',result:done};}
  if(job.kind==='update-engine'){
   /* A model is loaded and the person said yes to stopping it (the dialog names it): Control downloads while it serves, stops it cleanly, updates, and leaves it stopped.
      Whatever the result, the models it stopped are kept so the app offers to start them again, on the new engine or, after a failed update, on the old one. */
   const gate=args.stopModel===true&&instances.size?updateGate(job):null;
   try{/* force: the person pressed Update, so a version that failed its test start here earlier (held back for a day) is tried again (QAGUI-LIN 0213) */const done=await control.update({engine:true,control:false,force:true,stopModel:!!gate,onEvent:gate?e=>{onEvent(e);gate.event(e);}:onEvent,signal:job.abort.signal});gate?.finish(true);if(args.firstRun===true&&args.withControl===true&&!gate)await firstRunControl(job,onEvent);return {summary:updateSummary(done),result:done};}
   catch(error){gate?.finish(false);if(gate?.refusal)throw Object.assign(new Error(gate.refusal),{details:`Running outside this app: ${gate.outside.join(', ')}`});throw error;}
  }
  const done=await control.update({engine:false,control:true,onEvent,signal:job.abort.signal});
  updateLog(`Control step finished: ${updateSummary(done)}`);
  const notes=[updateSummary(done)];
  if(appUpdater){
   job.summary='Looking for a new version of the app…';emit();
   let verdict;
   try{verdict=await appUpdater.check();}catch(error){if(error?.code!=='app-update-unavailable')throw error;verdict={available:false};notes.push(appUpdater.unavailable||error.message);}
   updateLog(`app check: ${verdict.available?`version ${verdict.version} is available`:'no newer app version'}`);
   if(verdict.available){job.appUpdateFrom=thisAppVersion();job.appUpdateTo=verdict.version??null;emit();updateLog(`app update to ${verdict.version??'?'} starting; the installer will end this app`);await appUpdater.install({signal:job.abort.signal,onProgress:p=>{if(p.total){job.summary=`Downloading the app update: ${Math.round(100*p.downloaded/p.total)}%`;job.result={install:{step:job.summary,download:{file:'Strata app',percent:Math.round(100*p.downloaded/p.total)}}};emit();}}});notes.push('The app update is installed. Strata will restart.');}
  }
  return {summary:notes.filter(Boolean).join(' '),result:done};
 }
 /* The engine update with a model loaded (strata update --stop-model --output jsonl). The `models` events say what Control is doing to the models; this turns them into the app's own
    state: the model leaves the list when it is stopped, and a model the app did not start is never stopped (the update is ended before anything is downloaded). */
 function updateGate(job){
  const mine=new Map([...instances.values()].filter(i=>i.profile).map(i=>[i.profile,i])),stopped=[],gate={refusal:null,event(e){
   if(e.event!=='models')return;
   const models=Array.isArray(e.models)?e.models:[],label=m=>mine.get(m.profile)?.label??m.name??'The model';
   if(e.phase==='consent'){
    const other=models.filter(m=>!mine.has(m.profile));
    if(other.length){gate.refusal='A model that was not started from this app is running, so nothing was changed. Stop it, then update the engine again.';gate.outside=other.map(label);job.abort.abort();return;}
    job.step='getting-ready';job.models=models.map(label);job.summary=`Getting the new engine ready. ${job.models.join(' and ')} keeps running until it is ready.`;
   }
   else if(e.phase==='stopping'){for(const m of models){const i=mine.get(m.profile);if(i)i.status='unloading';}job.step='stopping';job.models=models.map(label);job.summary=`Stopping ${job.models.join(' and ')}. Answers in progress finish first.`;}
   else if(e.phase==='stopped'){for(const m of models){const i=mine.get(m.profile);if(i&&instances.get(i.id)===i){instances.delete(i.id);stopped.push(i);if(defaultId===i.id)defaultId=instances.keys().next().value??null;}}job.step='stopped';job.models=models.map(label);job.summary=`${job.models.join(' and ')} stopped. Updating the engine…`;}
   emit();
  },finish(updated){stoppedForUpdate=stopped.length?{updated,at:new Date().toISOString(),instances:stopped}:null;emit();}};
  return gate;
 }
 /* "Start again with the same settings": the stopped model's saved profile (the run file it was started with) is started again; the engine in use is whichever Control has primary now. */
 async function restartStopped(job){
  const offer=stoppedForUpdate;if(!offer?.instances?.length)throw new Error('There is no stopped model to start again.');
  stoppedForUpdate=null;emit();
  const labels=[];
  for(const old of offer.instances){
   const instance={...old,status:'loading',phase:'loading',error:null,metrics:null,loadedAt:new Date().toISOString()};
   instances.set(instance.id,instance);if(!defaultId||!instances.has(defaultId))defaultId=instance.id;job.summary=`Starting ${instance.label}. Loading can take a few minutes.`;emit();
   await autoDrafterFor(job,instance.model,`Starting ${instance.label}. Loading can take a few minutes.`);
   let wait=960;try{const {config,entry}=record(instance.model);wait=startWaitFor(config,entry,instance.model);}catch{/* the usual wait */}
   try{
    await control.start(instance.profile,{waitSeconds:wait});
    const deadline=Date.now()+300000;let ready=false;
    while(!ready&&Date.now()<deadline&&job.state==='running'){try{const response=await fetcher(`http://127.0.0.1:${instance.port}/health`,{headers:{authorization:`Bearer ${instance.key}`},signal:AbortSignal.timeout(2000)});ready=response.ok;}catch{/* not yet */}if(!ready)await new Promise(resolve=>setTimeout(resolve,1200));}
    if(!ready)throw new Error('Loading took more than five minutes. Check the engine log and free memory.');
    instance.status='ready';instance.phase='idle';ensurePoll();emit();labels.push(instance.label);
   }catch(error){
    await control.stop(instance.profile).catch(()=>{});
    instance.status='error';instance.error=humanStartError(error);emit();throw new Error(`${instance.label} did not start again. ${instance.error}`);
   }
  }
  return {summary:`${labels.join(' and ')} ${labels.length>1?'are':'is'} running again with the same settings.`};
 }
 function dismissRestart(){stoppedForUpdate=null;emit();return state();}
 function updateSummary(done){const r=(done.results??[]).filter(x=>x.status==='updated').map(x=>x.component==='engine'||x.component==='dsv41'?'The engine is updated and ready.':'Strata is updated. It takes effect the next time you open the app.');return r.length?r.join(' '):'Everything is already up to date.';}
 function execute(cmd,args,cwd){return new Promise((resolve,reject)=>{let output='';const process=spawnProcess(cmd,args,{cwd,stdio:['ignore','pipe','pipe']});process.stdout.on('data',c=>{output+=c;if(output.length>100000)process.kill();});process.stderr.on('data',c=>append('update',c.toString()));process.on('error',reject);process.on('exit',code=>code===0?resolve(output):reject(new Error('Git could not update this installation. Open diagnostics.')));});}
 /* Every write of a model's run config keeps the person's options: those they wrote into "args" or "user_args" since the file was last written are read from the file itself, go over what the caller built, and "args_written" is refreshed. */
 function writeModelConfig(path,next){const mine=mergeUserArgs(captureUserArgs(readJson(path)),next[USER_KEY]??[]);next=stampedUserArgs(next,mine);copyFileSync(path,path+'.bak');chmodSync(path+'.bak',0o600);const temp=path+'.ui-tmp';writeFileSync(temp,JSON.stringify(next,null,2)+'\n',{mode:0o600});renameSync(temp,path);}
 async function tune(job,args){
  await refresh(true);if(job.state!=='running')return;
  const chosen=record(args.model);
  /* The resource preference writes the Qwen engine's options (--pool-workers, --vram-reserve-mib); the DeepSeek engine takes neither (#4504), so its settings file is left as it is. */
  const resourceFor=config=>ownEngine(config)?{}:resourceSettings(config,snapshot.hardware,args.profile,modelArena(args.model));
  resourceFor(chosen.config);
  if(!chosen.entry.ready)throw new Error('Prepare the model before tuning.');
  if(externalRunning().length)throw new Error('Stop the native server outside this UI before tuning.');
  if(busy(requests().filter(r=>instances.has(r.instanceId))))throw new Error('Finish requests before tuning.');
  for(const id of [...instances.keys()])await unload(id);
  if(job.state!=='running')return;
  const initialValues=resourceFor(chosen.config);writeModelConfig(chosen.path,{...chosen.config,...initialValues,ui_resource_profile:args.profile});
  if(args.calibrate!==false&&state().capabilities.calibration){job.summary='Measuring the best engine settings for this computer…';emit();await program(job,args);}
  if(job.state!=='running')return;
  const current=record(args.model),values=resourceFor(current.config);
  const next={...current.config,...values,ui_resource_profile:args.profile};
  writeModelConfig(current.path,next);
  job.summary='Applying your resource preference and loading the model…';emit();
  const loaded=await loadEngine(job,{model:args.model,multiple:false,vision:current.config.vision?args.vision:'no',...(args.visionUnavailable===true?{visionUnavailable:true}:{})});
  if(job.state!=='running'||!loaded?.instanceId)return;
  job.summary='Checking your model’s speed…';emit();
  const measured=await benchmark(job,{instanceId:loaded.instanceId});
  return {...loaded,...measured,summary:'Your model is tuned and ready.',profile:args.profile};
 }
 /* What the tuner measures is what a load would start: the saved model config with the user's choices and the resource profile. */
 function tuneConfig(args){const {config,entry}=record(args.model);if(!entry.ready)throw new Error('Prepare the model before tuning it.');const {cfg}=buildConfig({model:args.model,vision:args.vision,resourceProfile:args.resourceProfile,gpus:args.gpus,values:args.values},config);return {path:writeTuneConfig(directory,args.model,cfg),entry};}
 async function tuneStatus(args){await refresh();const {path}=tuneConfig(args);return autotuner.status(path);}
 async function tuneEffective(args){await refresh();const {path}=tuneConfig(args);return autotuner.effective(path);}
 async function tuneOff(args){await refresh();const {path}=tuneConfig(args);return autotuner.off(path);}
 async function autotune(job,args){
  await refresh(true);if(job.state!=='running')return;
  if(packaged){const engine=await control.engine();if(!engine.usable)throw new Error(ENGINE_NOT_RECOGNISED+(engine.notice?` ${engine.notice}`:''));}
  const {path}=tuneConfig(args);
  if(externalRunning().length)throw new Error('Stop the native server running outside this UI before tuning.');
  if(busy(requests().filter(r=>instances.has(r.instanceId))))throw new Error('Finish requests before tuning.');
  await autoDrafterFor(job,args.model,'Finding the fastest settings for your PC…');if(job.state!=='running')return;
  for(const id of [...instances.keys()])await unload(id);
  if(job.state!=='running')return;
  job.summary='Finding the fastest settings for your PC…';emit();
  const room=args.resourceProfile==='work';
  return autotuner.start(job,job.kind==='recheck'?'recheck':'tune',{configPath:path,presetId:args.presetId,mode:args.mode,use:args.use,goal:room?'room':'speed',leaveRam:room?20:0,allowSmallDifferences:args.allowSmallDifferences===true,name:args.name},{onProgress:p=>{job.summary=p.text??job.summary;emit();}});
 }
 function operation(kind,args={}){if(!['install','load','pictures','benchmark','tune','autotune','recheck','calibrate','update-engine','update-strata','rollback','install-engine-file','remove','restart-models'].includes(kind))throw new Error('Choose a supported native operation.');if(kind==='restart-models'&&!packaged)throw new Error('Choose a supported native operation.');if(!snapshot.available&&!(packaged&&(COMPONENT_KINDS.includes(kind)||kind==='restart-models')))throw new Error(snapshot.error??'Connect the native Strata installation first.');if(kind==='install'||kind==='load'){const refusal=engineRefusal(snapshot.catalog,args.model,snapshot.engineInfo);if(refusal)throw new Error(refusal);}return begin(kind,job=>kind==='install'?install(job,args):kind==='load'?loadEngine(job,args):kind==='pictures'?picturesOnly(job,args):kind==='benchmark'?benchmark(job,args):kind==='tune'?tune(job,args):kind==='autotune'||kind==='recheck'?autotune(job,args):kind==='remove'?removeModel(job,args):kind==='restart-models'?restartStopped(job):program(job,args),{guided:args.guided===true,model:args.model,...(args.family?{family:args.family}:{})});}
 /* ---- The models folder card (0.2.2): what a picked folder means, and the three choices. The copying is Control's (strata models move). ---- */
 const currentModelsFolder=()=>preferences.downloadDirectory??(modelsDirectory?modelsDirectory():snapshot.installed?.data_folder)??null;
 const moveFile=()=>join(directory,'models-move.cancel');
 const folderWords=error=>({message:error.message,details:error.details||''});
 /** A picked folder as the data folder: a whole drive (D:\) becomes D:\Strata, and the person is told (the models then live in D:\Strata\Models). */
 function pickedModelsFolder(input){
  if(typeof input!=='string'||!input.trim()||/[\r\n\0]/.test(input))throw new Error('Choose a models folder.');
  const picked=resolve(input.trim()),target=dataFolderFromPick(picked);
  return {picked,folder:target.path,wholeDrive:!!target.note,shown:target.note?join(target.path,'Models'):target.path};
 }
 async function planModelsFolder(input){
  if(!packaged)throw new Error('Changing the models folder is part of the Strata app.');
  const now=currentModelsFolder();if(!now)throw new Error('Strata is still reading your models folder. Try again in a moment.');
  const pick=pickedModelsFolder(input),plan=await control.planModelsMove(now,pick.folder);
  return {...plan,...pick,current:now,running:instances.size,busy:[...jobs.values()].some(j=>j.state==='running')};
 }
 /* "Start empty": the old models are no longer listed. Their run configs are moved (not deleted) into old-models beside the others; the model files are not touched. */
 function setAsideConfigs(old){
  const folder=repo(),kept=join(folder,'old-models'),aside=[];
  const under=value=>typeof value==='string'&&[ 'models','packs','mtp' ].some(item=>{const base=join(old,item).toLowerCase();const v=value.toLowerCase();return v===base||v.startsWith(base+sep)||v.startsWith(base+'/');});
  const scan=value=>Array.isArray(value)?value.some(scan):value&&typeof value==='object'?Object.values(value).some(scan):under(value);
  for(const name of readdirSync(folder)){
   if(!/^strata-[a-z0-9_.-]+\.json$/i.test(name)||name.endsWith('.shared-settings.json'))continue;
   if(!scan(readJson(join(folder,name))))continue;
   mkdirSync(kept,{recursive:true});renameSync(join(folder,name),join(kept,name));aside.push(name);
  }
  return aside;
 }
 async function startModelsFolder({folder:input,choice,stopRunning=false}){
  if(!packaged)throw new Error('Changing the models folder is part of the Strata app.');
  if(!['move','keep','empty'].includes(choice))throw new Error('Choose what to do with your models.');
  if([...jobs.values()].some(j=>j.state==='running'))throw new Error('Finish or stop the download to change the folder.');
  const plan=await planModelsFolder(input);
  if(plan.issue)throw Object.assign(new Error(plan.issue.message),{details:plan.issue.details,code:plan.issue.code});
  if(!plan.choices[choice].possible)throw Object.assign(new Error(plan.choices[choice].reason||'That choice is not possible here.'),{code:'not_possible'});
  if(instances.size){
   if(!stopRunning)throw Object.assign(new Error('A model is running. Stop it first, then change the folder.'),{code:'running'});
   for(const id of [...instances.keys()])await unload(id);
  }
  const old=plan.current,target=plan.folder;
  if(choice!=='move'){
   makeFolder(target);
   const aside=choice==='empty'?setAsideConfigs(old):[];
   setDownloadDirectory(target);lastRead=0;await refresh(true);
   return {done:true,choice,folder:target,oldFolder:old,setAside:aside};
  }
  return runModelsMove(old,target,(from,to,options)=>control.moveModels(from,to,options));
 }
 /* The move as a job (the progress dialog reads it). `run` is moveModels for a new move, resumeModelsMove for "Finish it". */
 function runModelsMove(old,target,run){
  rmSync(moveFile(),{force:true});
  return begin('models-move',async job=>{
   job.cancelFile=moveFile();job.summary='Starting…';
   try{
    const result=await run(old,target,{installDir:repo(),cancelFile:moveFile(),onEvent:event=>{
     if(event.event!=='progress')return;
     job.progress={done:event.done,total:event.total,file:event.file,group:event.group,index:event.index,count:event.count,secondsLeft:event.seconds_left||0,sameDrive:!!event.same_drive};
     job.summary=`${Math.round(event.done/1e8)/10} of ${Math.round(event.total/1e8)/10} GB`;emit();}});
    const finished={done:!!result.done,stopped:!!result.stopped,moved:result.moved??[],remaining:result.remaining??[],bytes:result.bytes_moved??0,total:result.total_bytes??0,from:old,to:target,files:result.files_moved??0,sameDrive:!!result.same_drive};
    /* Only a whole move changes the folder the app uses: a stopped move leaves it where it was, and every model works where it is. */
    if(result.done){setDownloadDirectory(target,{insideJob:true});finished.folder=target;}
    job.progress=null;
    return {summary:result.done?'Your models are moved.':'Moving stopped. Nothing was lost.',...finished};
   }catch(error){
    const left=error.output;
    job.result={failed:true,code:error.code,moved:left?.moved??[],remaining:left?.remaining??[],bytes:left?.bytes_moved??0,total:left?.total_bytes??0,from:old,to:target};
    job.progress=null;
    throw error;
   }finally{rmSync(moveFile(),{force:true});}
  },{family:undefined});
 }
 /* "A move was interrupted. Finish it or leave it": the next start finds the unfinished move (Control keeps a note in the models folder in use). */
 async function interruptedMove(){
  if(!packaged)return null;
  const now=currentModelsFolder();if(!now)return null;
  if([...jobs.values()].some(j=>j.kind==='models-move'&&j.state==='running'))return null;
  try{return await control.interruptedModelsMove(now);}catch{return null;}
 }
 async function finishInterruptedMove({stopRunning=false}={}){
  const st=await interruptedMove();if(!st)throw new Error('No move was interrupted.');
  if([...jobs.values()].some(j=>j.state==='running'))throw new Error('Finish or stop the download to change the folder.');
  if(instances.size){
   if(!stopRunning)throw Object.assign(new Error('A model is running. Stop it first, then change the folder.'),{code:'running'});
   for(const id of [...instances.keys()])await unload(id);
  }
  return runModelsMove(st.from,st.to,(from,to,options)=>control.resumeModelsMove(from,options));
 }
 async function leaveInterruptedMove(){
  const st=await interruptedMove();if(!st)throw new Error('No move was interrupted.');
  if([...jobs.values()].some(j=>j.state==='running'))throw new Error('Finish or stop the download to change the folder.');
  const result=await control.leaveModelsMove(st.from,{installDir:repo()});
  lastRead=0;await refresh(true);
  return {done:true,folder:st.from,partial:st.to,removed:result.removed??0,restored:result.restored??0};
 }
 async function cancel(id){const job=jobs.get(id);if(!job||job.state!=='running')throw new Error('This operation is no longer running.');if(job.kind==='install'){job.engineAbort?.abort();await tool('strata_install',{cancel:true});}else if(job.kind==='models-move'){/* a safe stop: the copy in flight is dropped, files already moved stay */try{writeFileSync(job.cancelFile,'stop');}catch{/* the run ends anyway */}job.summary='Stopping…';emit();return state();}else if(job.abort)job.abort.abort();else if(job.process){if(!['tune','calibrate','autotune','recheck'].includes(job.kind))throw new Error('File updates cannot be cancelled safely while files are being replaced.');job.state='cancelled';await stopProcess(job.process);}job.state='cancelled';job.summary=job.kind==='install'?'Stopped. Completed downloads are kept.':'Stopped.';emit();return state();}
 async function close(){closing=true;clearInterval(timer);const installing=packaged&&child&&[...jobs.values()].some(j=>j.kind==='install'&&j.state==='running');if(installing)await send('tools/call',{name:'strata_install',arguments:{cancel:true}},15000).catch(()=>{});const operations=[];for(const job of jobs.values()){job.abort?.abort();if(job.process)operations.push(stopProcess(job.process));if(job.state==='running'&&!(job.kind==='update-strata'&&typeof job.appUpdateFrom==='string'))job.state='cancelled';}/* (an app update ends the app on purpose: its job stays running, so the next start can say it finished) */await Promise.all(operations);await Promise.all([...instances.keys()].map(id=>unload(id,true).catch(()=>{})));disconnect();}
 // Remove nonserializable process references from public jobs.
 const publicState=()=>{const value=state();value.jobs=value.jobs.map(({process,engineAbort,rawResult,...job})=>job);return value;};
 async function localPreview(path){if(!packaged)throw new Error('Checking local model files is part of the Strata app.');if(typeof path!=='string'||!path.trim()||[...path].some(c=>c.charCodeAt(0)<32))throw new Error('Choose a folder.');return control.importPlan(path.trim());}
 return {state:publicState,settingsHints,tuneStatus,tuneEffective,tuneOff,tunePresets:{list:()=>autotuner.list(),action:(kind,id,extra)=>autotuner.action(kind,id,extra),undoFor:async args=>autotuner.undoFor(tuneConfig(args).path)},refresh,localPreview,configure,setDownloadDirectory,planModelsFolder,startModelsFolder,interruptedMove,finishInterruptedMove,leaveInterruptedMove,plan,settings,save,settingsFile,openSettingsFile,operation,dismissRestart,cancel,unload,resolveInstance,poll,logs:()=>logs.slice(),benchmarkReport:()=>readJson(join(directory,'native-benchmark.json'),null),setDefault(id){defaultId=resolveInstance(id).id;emit();return publicState();},hasLive:()=>[...instances.values()].some(i=>i.status==='ready'),readyInstances:()=>[...instances.values()].filter(i=>i.status==='ready'),defaultInstanceId:()=>defaultId,close};
}
/* A DeepSeek config on AMD written by setup 0.2.8 or earlier names ONE card and marks it "asked", though setup picked it itself (strata_control/gpu_pick.py): it must not stop the app's recommended cards from applying. A card setup named for itself is the first of the usable cards, largest memory first; another card number was the person's. Roles and a "gpus_by" marker are always honoured. */
export function setupPickedOneCard(config,hardware){
 if(!config||config.arch!=='deepseek41'||config.backend!=='hip'||config.gpus_by||config.gpu_roles||!config.gpus_asked)return false;
 const one=Array.isArray(config.gpu)?(config.gpu.length===1?config.gpu[0]:null):config.gpu;if(!Number.isInteger(Number(one))||one===null||one==='')return false;
 const cards=(hardware?.gpus??[]).filter(g=>(g.vendor==='amd')&&g.usable!==false),shares=g=>!!(g.uma||g.integrated||g.unified_gb),vram=g=>Number(g.vram_gb)||0;
 const real=cards.some(g=>!shares(g)&&vram(g)>=4),keep=cards.filter(g=>!(shares(g)&&real)&&vram(g)>=8);
 if(process.platform==='win32'||keep.length<2)return false;
 const first=[...keep].sort((a,b)=>Math.round(vram(b))-Math.round(vram(a))||a.index-b.index)[0];
 return Number(one)===first.index;
}
