import test from 'node:test';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {createRequire} from 'node:module';
import {build} from 'esbuild';

/* The family menu rows for the update feed's models: a model this engine cannot use is a note with its reason, not a choice; one that is allowed shows the reason beside its name. */
const root=resolve(import.meta.dirname,'..');
const virtual={'./ui':"export const Icon=()=>null;",'./FloatingMenu':"export const FloatingMenu=()=>null;"};
const stubs={name:'stubs',setup(b){b.onResolve({filter:/^\.\/(ui|FloatingMenu)$/},a=>({path:a.path,namespace:'stub'}));b.onLoad({filter:/.*/,namespace:'stub'},a=>({contents:virtual[a.path],loader:'js'}));}};
const out=await build({entryPoints:[resolve(root,'src/components/FamilySwitch.tsx')],bundle:true,write:false,platform:'node',format:'cjs',jsx:'automatic',plugins:[stubs],external:['react','react-dom','react/jsx-runtime'],loader:{'.json':'json'}});
const require=createRequire(resolve(root,'package.json'));
const mod={exports:{}};new Function('module','exports','require',out.outputFiles[0].text)(mod,mod.exports,require);
const {renderToStaticMarkup}=require('react-dom/server');
const React=require('react');
const {FamilyMenuList}=mod.exports;
const fam=(family,title,extra={})=>({family,title,about:'',experimental:false,images:true,sizes:[{id:family+'-x'}],...extra});
test('a model the engine cannot use is a note with its reason; an allowed one shows the reason beside its name',()=>{
 const reason='needs a newer Strata engine: update Strata to use it';
 const shown=[fam('qwen3.8','Qwen3.8-Flash-Next'),fam('qwen4-test','Qwen4 Test',{feed:true,available:false,unavailable_reason:reason})];
 const missing=[fam('later-one','Later One',{sizes:[],feed:true,available:false,unavailable_reason:"this Strata version doesn't know this model yet: update Strata to use it"})];
 const html=renderToStaticMarkup(React.createElement(FamilyMenuList,{shown,missing,current:shown[0],onPick(){}}));
 assert.equal((html.match(/role="menuitemradio"/g)||[]).length,2);
 assert.equal((html.match(/role="note"/g)||[]).length,1);
 assert(html.includes('Qwen4 Test</strong><small>'+reason+'</small>'));
 assert(html.includes('Later One</strong><small>this Strata version doesn&#x27;t know this model yet: update Strata to use it</small>'));
 const plain=renderToStaticMarkup(React.createElement(FamilyMenuList,{shown:[shown[0]],missing:[],current:shown[0],onPick(){}}));
 assert(!plain.includes('<small>'));assert(!plain.includes('role="note"'));
});
