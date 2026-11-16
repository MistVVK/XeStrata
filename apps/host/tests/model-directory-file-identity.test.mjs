import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,realpathSync,rmSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';

// Deterministic filesystem-ID controls, not a physical Windows/NTFS reproduction.
// Import the actual store after syncing only the selected fs wrappers into ESM.
const cases = [];
for (const mode of ['collision-ino','collision-dev','collision-both','distinct-small','other-device','zero-inode','stat-failure']) {
 for (const kind of ['directory','link']) cases.push({mode,kind,conflict:true});
}
cases.push({mode:'valid-identity-alias',kind:'link',conflict:false});
cases.push({mode:'valid-canonical-alias',kind:'link',conflict:false});

for (const {mode,kind,conflict} of cases) test(`native file identity: ${mode} / ${kind}`,t=>{
 const scratch=realpathSync(mkdtempSync(join(tmpdir(),'strata-file-identity-')));
 t.after(()=>rmSync(scratch,{recursive:true,force:true}));
 const home=join(scratch,'home');mkdirSync(home);
 const env={...process.env,STRATA_HOME:home,STRATA_DOCUMENTS_DIR:join(scratch,'documents'),STRATA_ID_MODE:mode,STRATA_ID_KIND:kind,STRATA_ID_EXPECT_CONFLICT:String(conflict)};
 delete env.STRATA_MODELS_DIR;
 const result=spawnSync(process.execPath,['--input-type=module','-e',`
  import assert from 'node:assert/strict';
  import fs from 'node:fs';
  import {syncBuiltinESMExports} from 'node:module';
  import {join} from 'node:path';
  const {mkdirSync,writeFileSync,readFileSync,symlinkSync}=fs;
  const home=process.env.STRATA_HOME,mode=process.env.STRATA_ID_MODE,kind=process.env.STRATA_ID_KIND;
  const original=join(home,'models'),other=join(home,'other-library'),alias=join(home,'Strata-data','models');
  mkdirSync(original);mkdirSync(other);mkdirSync(join(home,'Strata-data'));
  writeFileSync(join(original,'original.gguf'),'GGUForiginal');writeFileSync(join(other,'other.gguf'),'GGUFother');
  const valid=mode.startsWith('valid-');
  if(kind==='directory'){mkdirSync(alias);writeFileSync(join(alias,'other.gguf'),'GGUFconflict');}
  else symlinkSync(valid?original:other,alias,process.platform==='win32'?'junction':'dir');
  const wide=2n**60n;
  let rootID={ino:101n,dev:42n},aliasID={ino:102n,dev:42n};
  if(mode==='collision-ino'){rootID={ino:wide,dev:42n};aliasID={ino:wide+1n,dev:42n};}
  if(mode==='collision-dev'){rootID={ino:101n,dev:wide};aliasID={ino:101n,dev:wide+1n};}
  if(mode==='collision-both'){rootID={ino:wide,dev:wide};aliasID={ino:wide+1n,dev:wide+1n};}
  if(mode==='other-device'){rootID={ino:101n,dev:42n};aliasID={ino:101n,dev:43n};}
  if(mode==='zero-inode'){rootID={ino:0n,dev:42n};aliasID={ino:0n,dev:42n};}
  if(valid){rootID={ino:wide,dev:wide};aliasID={...rootID};}
  const originalStat=fs.statSync,originalRealpath=fs.realpathSync,observed=[];
  fs.statSync=function(path,options){
   const actual=originalStat(path,options);
   if(path!==original&&path!==alias)return actual;
   if(mode==='stat-failure'&&path===alias){const error=new Error('EACCES identity read');error.code='EACCES';throw error;}
   const id=path===original?rootID:aliasID,big=options?.bigint===true;
   const values={ino:big?id.ino:Number(id.ino),dev:big?id.dev:Number(id.dev)};
   observed.push({path:path===original?'original':'alias',bigint:big,ino:String(values.ino),dev:String(values.dev),exactIno:String(id.ino),exactDev:String(id.dev)});
   return new Proxy(actual,{get(target,key){return key in values?values[key]:Reflect.get(target,key);}});
  };
  fs.realpathSync=function(path,options){
   const actual=originalRealpath(path,options);
   return mode==='valid-identity-alias'&&path===alias?actual+'/.':actual;
  };
  syncBuiltinESMExports();
  const store=await import('./server/store.mjs');
  assert.equal(store.MODELS_DIR_NOTICE,'');
  assert.equal(store.MODELS_DIR,originalRealpath(original));
  const before={original:readFileSync(join(original,'original.gguf'),'utf8'),other:readFileSync(join(other,'other.gguf'),'utf8')};
  let error=null,value;
  try{value=store.nativeDataDirectory(store.MODELS_DIR);}catch(e){error=e;}
  const trace={mode,kind,expectedConflict:process.env.STRATA_ID_EXPECT_CONFLICT==='true',rootID,aliasID,observed,threw:!!error,error:error?.message??null,value:value??null};
  const json=JSON.stringify(trace,(_k,v)=>typeof v==='bigint'?String(v):v);
  console.log(json);
  if(process.env.STRATA_IDENTITY_TRACE_DIR)writeFileSync(join(process.env.STRATA_IDENTITY_TRACE_DIR,mode+'-'+kind+'.json'),json+'\\n');
  if(mode==='stat-failure')assert.match(error?.message??'',/EACCES/,'filesystem identity errors must propagate');
  else if(process.env.STRATA_ID_EXPECT_CONFLICT==='true')assert.throws(()=>{if(error)throw error;},/conflicts with another library/);
  else{assert.equal(error,null);assert.equal(value,join(home,'Strata-data'));}
  assert.equal(readFileSync(join(original,'original.gguf'),'utf8'),before.original);
  assert.equal(readFileSync(join(other,'other.gguf'),'utf8'),before.other);
  if(kind==='directory')assert.equal(readFileSync(join(alias,'other.gguf'),'utf8'),'GGUFconflict');
  else assert.equal(originalRealpath(alias),originalRealpath(valid?original:other));
 `],{cwd:fileURLToPath(new URL('..',import.meta.url)),env,encoding:'utf8',windowsHide:true});
 assert.equal(result.status,0,`${result.stdout}\n${result.stderr}`);
});
