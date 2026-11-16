import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdir, mkdtemp, realpath, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, extname, join, resolve, sep } from 'node:path';
import { externalEnvironment } from './external-env.mjs';
const run=(command,args)=>promisify(execFile)(command,args,{env:externalEnvironment()});   // xdg-open and the apps it starts: the environment the AppImage was started from
const extensions=new Set(['.md','.txt','.pdf','.html','.htm','.png','.jpg','.jpeg','.webp','.gif','.svg','.csv','.ts','.tsx','.js','.jsx','.py','.json','.css','.rs','.go','.sh','.mp4','.webm']);
/** A request this opener will not carry out (a scheme other than http(s), a file outside the previews): a 400 with its reason, not a server error. */
const refused=message=>Object.assign(new Error(message),{status:400});
export async function openArtifact(body,{examplesDir,previewsDir,platform=process.platform,launch=run}) {
  if(!['open','reveal'].includes(body.action))throw refused('Choose Open or Show in file explorer.');
  if(typeof body.url==='string') {
    // #4721: a web address goes to the person's default browser (the app window drops window.open: WebView2 has no new-window handler). http and https only, no shell.
    if(body.action!=='open')throw refused('A web address can only be opened.');
    let address;try{address=new URL(body.url);}catch{throw refused('This is not a web address.');}
    if(!['http:','https:'].includes(address.protocol)||body.url.length>4096)throw refused('Only http and https addresses can be opened.');
    if(platform==='darwin')await launch('open',[address.href]);
    else if(platform==='win32')await launch('rundll32.exe',['url.dll,FileProtocolHandler',address.href]);
    else await launch('xdg-open',[address.href]);
    return {ok:true};
  }
  let file;
  if(typeof body.path==='string') {
    // Only public example assets can be addressed by path. Never accept host paths.
    if(!body.path.startsWith('/examples/'))throw refused('This file is not a local preview.');
    const base=await realpath(examplesDir);
    file=await realpath(resolve(base,body.path.slice('/examples/'.length)));
    if(!file.startsWith(base+sep)||!(await stat(file)).isFile())throw refused('This file is not a local preview.');
  } else {
    const name=basename(String(body.name??'document.txt')).replace(/[^\w. -]/g,'_').slice(0,120);
    if(!extensions.has(extname(name).toLowerCase()))throw refused('This file type cannot be opened from a preview.');
    if(typeof body.content!=='string'&&typeof body.base64!=='string')throw refused('No file content was supplied.');
    const bytes=typeof body.base64==='string'?Buffer.from(body.base64,'base64'):Buffer.from(body.content,'utf8');
    if(bytes.length>20*1024*1024)throw refused('Choose a file smaller than 20 MB.');
    await mkdir(previewsDir,{recursive:true});const folder=await mkdtemp(join(previewsDir,'preview-'));
    file=join(folder,name);await writeFile(file,bytes,{flag:'wx',mode:0o600});
  }
  if(!extensions.has(extname(file).toLowerCase()))throw refused('This file type cannot be opened from a preview.');
  if(platform==='darwin')await launch('open',body.action==='reveal'?['-R',file]:[file]);
  else if(platform==='win32')await launch('explorer.exe',body.action==='reveal'?[`/select,${file}`]:[file]);
  else await launch('xdg-open',[body.action==='reveal'?dirname(file):file]);
  return {ok:true};
}
