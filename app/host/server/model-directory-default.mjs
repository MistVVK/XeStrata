import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { posix, win32 } from 'node:path';
import { utf8Script } from './powershell-utf8.mjs';

/** Resolve the user's Documents folder without evaluating shell configuration. */
export function defaultModelDirectory({platform=process.platform,home=homedir(),env=process.env,read=readFileSync,windowsDocuments=()=>execFileSync('powershell.exe',['-NoProfile','-NonInteractive','-Command',utf8Script("[Environment]::GetFolderPath('MyDocuments')")],{encoding:'utf8',timeout:5000,windowsHide:true}).trim()}={}) {
 const path=platform==='win32'?win32:posix;
 if(env.STRATA_MODELS_DIR)return path.resolve(env.STRATA_MODELS_DIR);
 return path.join(documentsFolder({platform,home,env,read,windowsDocuments}),'Strata','Models');
}

/** The user's Documents folder as this system names it (Linux: XDG_DOCUMENTS_DIR, so "Dokumente" on a German desktop), else <home>/Documents. */
export function documentsFolder({platform=process.platform,home=homedir(),env=process.env,read=readFileSync,windowsDocuments=()=>execFileSync('powershell.exe',['-NoProfile','-NonInteractive','-Command',utf8Script("[Environment]::GetFolderPath('MyDocuments')")],{encoding:'utf8',timeout:5000,windowsHide:true}).trim()}={}) {
 const path=platform==='win32'?win32:posix;
 let documents=env.STRATA_DOCUMENTS_DIR;
 if(!documents&&platform==='win32'){try{documents=windowsDocuments();}catch{} }
 if(!documents&&platform==='linux'){
  let value=env.XDG_DOCUMENTS_DIR;
  if(!value){try{value=read(path.join(env.XDG_CONFIG_HOME||path.join(home,'.config'),'user-dirs.dirs'),'utf8').match(/^\s*XDG_DOCUMENTS_DIR="([^"\n]+)"/m)?.[1];}catch{} }
  if(value){const expanded=value.replace(/\$\{HOME\}|\$HOME(?=\/|$)/g,home);if(!expanded.includes('$')&&path.isAbsolute(expanded))documents=expanded;}
 }
 if(!documents||!path.isAbsolute(documents))documents=path.join(home,'Documents');
 return documents;
}

/**
 * Where the default chat workspace goes, best place first (#3749). The models folder is <Documents>/Strata/Models with Documents as the system names it, and the
 * workspace is "next to it": on a German Linux desktop the models were in Dokumente/Strata and the workspace in Documents/Strata. On Linux the system's
 * Documents folder comes first; every other system keeps <home>/Documents (a Windows Documents folder moved to OneDrive would move the default workspace of
 * people who already use it).
 */
export function workspaceBases(user, {platform=process.platform,env=process.env,read=readFileSync,exists=existsSync}={}) {
 const path=platform==='win32'?win32:posix;
 const old=[path.join(user,'Documents'),user];
 if(platform!=='linux')return old;
 const system=documentsFolder({platform,home:user,env,read});
 // A workspace already made in <home>/Documents (every earlier version) is kept while the system's folder has none: its chats and files stay where they are.
 if(!exists(path.join(system,'Strata','Workspace'))&&exists(path.join(old[0],'Strata','Workspace')))return old;
 return [system,...old.filter(p=>p!==system)];
}
