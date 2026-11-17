import test from 'node:test';
import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {createRequire} from 'node:module';
import {build} from 'esbuild';

/* PR #3798 follow-up: when the host reports a card with no dedicated memory, the setup wizard takes the engine's figure. Each engine card
   counts once: two identical cards are not both given the first one's size, and a "7900 XT" does not take the "7900 XTX" figure. */
const root=resolve(import.meta.dirname,'..');
const virtual={'./ui':"export const Icon=()=>null;"};
const stubs={name:'stubs',setup(b){b.onResolve({filter:/^\.\/ui$/},a=>({path:a.path,namespace:'stub'}));b.onLoad({filter:/.*/,namespace:'stub'},a=>({contents:virtual[a.path],loader:'js'}));}};
const out=await build({entryPoints:[resolve(root,'src/components/OnboardingMemory.tsx')],bundle:true,write:false,platform:'node',format:'cjs',jsx:'automatic',plugins:[stubs],external:['react','react-dom','react/jsx-runtime'],loader:{'.json':'json'}});
const require=createRequire(resolve(root,'package.json'));
const mod={exports:{}};new Function('module','exports','require',out.outputFiles[0].text)(mod,mod.exports,require);
const {renderToStaticMarkup}=require('react-dom/server');
const React=require('react');
const {OnboardingMemory}=mod.exports;

const hostCard=(name,i,vramGb=0)=>({id:`pci:${i}`,index:i,name,vramGb,unified:false,backend:'hip'});
const engineCard=(name,i,vram_gb)=>({index:i,name,vram_gb,usable:true});
const render=(host,engine)=>renderToStaticMarkup(React.createElement(OnboardingMemory,{hardware:{memory:{totalGb:64},accelerator:{unified:false},gpus:host},native:{ram_gb:64,gpus:engine}}));
const perCard=(html)=>[...html.matchAll(/<li[^>]*>.*?<strong>([^<]*)<\/strong><\/li>/g)].map(m=>m[1]);
const vramStat=(html)=>/VRAM.*?<strong>([^<]*)<\/strong>/.exec(html)?.[1];

test('two identical cards with different sizes keep their own figures',()=>{
 const html=render([hostCard('AMD Radeon PRO W7600',0),hostCard('AMD Radeon PRO W7600',1)],[engineCard('AMD Radeon PRO W7600',0,8),engineCard('AMD Radeon PRO W7600',1,16)]);
 assert.deepEqual(perCard(html),['8 GB','16 GB']);
 assert.equal(vramStat(html),'24 GB');
});

test('a 7900 XT takes its own figure, not the 7900 XTX listed before it',()=>{
 const html=render([hostCard('AMD Radeon RX 7900 XT',0),hostCard('AMD Radeon RX 7900 XTX',1)],[engineCard('AMD Radeon RX 7900 XTX',0,24),engineCard('AMD Radeon RX 7900 XT',1,20)]);
 assert.deepEqual(perCard(html),['20 GB','24 GB']);
 assert.equal(vramStat(html),'44 GB');
});

test('a card the host reports keeps the host figure; one the engine does not list stays at 0',()=>{
 const html=render([hostCard('NVIDIA GeForce RTX 4090',0,24),hostCard('AMD Radeon RX 9070 XT',1)],[engineCard('NVIDIA GeForce RTX 4090',0,23.5)]);
 assert.deepEqual(perCard(html),['24 GB','0.0 GB']);
});

test('a name that only contains the other still matches when nothing better is listed',()=>{
 const html=render([hostCard('Radeon RX 9070 XT',0)],[engineCard('AMD Radeon RX 9070 XT (gfx1201)',0,16)]);
 assert.equal(vramStat(html),'16 GB');
});
