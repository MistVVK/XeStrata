export interface DownloadProgress {percent?:number;done_gb?:number;total_gb?:number;remaining_gb?:number;file?:string}
/** What the plan says is already on this computer (Control's plan: files_complete = every file whole at its pinned size, partly_downloaded = some file is there,
 *  download_left_gb = what is still to fetch). */
export interface FilesHere{complete?:boolean|null;partly?:boolean|null;left?:number|null}
export function filesHere(plan:{files_complete?:unknown;partly_downloaded?:unknown;download_left_gb?:unknown}|null|undefined):FilesHere{
 const left=typeof plan?.download_left_gb==='number'&&Number.isFinite(plan.download_left_gb)&&plan.download_left_gb>=0?plan.download_left_gb:null;
 return {complete:plan?.files_complete===true,partly:plan?.partly_downloaded===true,left};
}
/** FIX-APP A1: what a version card on the "Your model" step shows in its size slot. `here` is the plan's answer for that version (read on this step, one step earlier than
 *  the Prepare page): the files all on this computer say so instead of a download size; part of them here shows what is left. No answer (yet): the download size, as before. */
export interface CardSize{kind:'download'|'here'|'left';/** the number in the slot (GB); none when the files are all here */value?:number;/** the words in the slot when there is no number */text?:string;/** the small words beside it; none = the page's own caption */caption?:string}
export function versionCardSize(version:{download_gb:number},here?:FilesHere|null):CardSize{
 if(here?.complete)return {kind:'here',text:'Already on this computer'};
 if(here?.partly&&typeof here.left==='number'&&here.left>0&&here.left<version.download_gb)return {kind:'left',value:here.left,caption:'Left to download'};
 return {kind:'download',value:version.download_gb};
}
/** The Prepare page's title: the check of the files is not a download, and files that are all here are only prepared. */
export function prepareTitle(running:boolean,progress:DownloadProgress|undefined,here:FilesHere|undefined){
 if(running&&progress?.file==='checking files')return 'Checking the model files...';
 if(here?.complete)return 'Preparing the model...';
 return 'Downloading the model...';
}
/** The native installer reports each shard, not whole-model progress. */
export interface Counter{value:number|undefined;label:string;key:string;/** no number, a check mark instead */plain?:boolean;/** a line under the label */note?:string}
export function downloadCounter(size:number|undefined,progress:DownloadProgress|undefined,running:boolean,here?:FilesHere):Counter{
 if(running&&progress?.file==='checking files')return {value:undefined,label:'Preparing files…',key:'preparing'};   // nothing is downloaded while setup checks the files (DL99)
 if(running&&progress){
  if(Number.isFinite(progress.remaining_gb))return {value:Math.max(0,progress.remaining_gb!),label:'GB left to download',key:'total'};
  if(Number.isFinite(progress.total_gb)&&Number.isFinite(progress.done_gb))return {value:Math.max(0,progress.total_gb!-progress.done_gb!),label:'GB left in this file',key:progress.file??'file'};
  if(Number.isFinite(progress.total_gb)&&Number.isFinite(progress.percent))return {value:Math.max(0,progress.total_gb!*(1-Math.min(100,Math.max(0,progress.percent!))/100)),label:'GB left in this file',key:progress.file??'file'};
 }
 if(running)return {value:undefined,label:progress?'Downloading…':'Preparing files…',key:'preparing'};
 if(here?.complete)return {value:undefined,label:'Already on this computer',key:'here',plain:true};     // every file is here: nothing to download
 if(here?.partly)return typeof here.left==='number'?{value:here.left,label:'GB left to download',key:'left',note:'Part of the model is already here'}
   :{value:undefined,label:'Part of the model is already here',key:'part',plain:true};
 return {value:size,label:'GB to download',key:'size'};
}
/** The setup wizard moves on from its download page only when the model is ready. An install that ended with every file in, for a model that is not ready
 *  because the engine for this PC is missing or can't be used, left the page at its last figure (99%) with no word (DL99): it then says so, with the next step. */
export function engineNotReadyAfterInstall(installState:string|undefined,modelReady:boolean|undefined,engineUsable:boolean|undefined){
 return installState==='done'&&modelReady!==true&&engineUsable===false;
}
export function onboardingError(message:string){
 if(/(?:not enough.*(?:disk|storage)|disk.*(?:space|full)|ENOSPC)/i.test(message)){
  const needed=message.match(/(?:about\s+)?([\d.]+)\s*GB\s+(?:needed|required)/i)?.[1];
  const free=message.match(/([\d.]+)\s*GB\s+(?:free|available)/i)?.[1];
  return {title:'You need more storage.',message:needed?`This model needs about ${needed} GB.${free?` You have ${free} GB free.`:''}`:'There isn’t enough free space to download this model.',next:'Choose another folder or a smaller model.',storage:true};
 }
 return {title:'Something needs your attention.',message,next:undefined,storage:false};
}
export function modelFolderLabel(path:string){
 return /[\\/]Documents[\\/]Strata[\\/]Models[\\/]?$/i.test(path)?'Documents / Strata / Models':path;
}
