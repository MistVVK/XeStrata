import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import {transform} from 'esbuild';
const {code}=await transform(await readFile(new URL('../src/lib/hardware-summary.ts',import.meta.url),'utf8'),{loader:'ts',format:'esm'});
const {memoryInventory,listedGpus}=await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
test('RAM and every dedicated GPU contribute to the aggregate inventory',()=>{
 const hardware={memory:{totalGb:64},accelerator:{unified:false},gpus:[{unified:false,backend:'cuda',vramGb:24},{unified:false,backend:'cuda',vramGb:16}]};
 assert.deepEqual(memoryInventory(hardware),{ram:64,vram:40,total:104,shared:false});
});
test('unified GPU memory is counted once, even with multiple reported shared GPU pools',()=>{
 const hardware={memory:{totalGb:16},accelerator:{unified:true},gpus:[{unified:true,backend:'metal',vramGb:16},{unified:true,backend:'metal',vramGb:16}]};
 assert.deepEqual(memoryInventory(hardware),{ram:16,vram:0,total:16,shared:true});
});
test('shared GPU memory is excluded when dedicated GPUs are also connected',()=>{
 const hardware={memory:{totalGb:32},accelerator:{unified:true},gpus:[{unified:true,backend:'metal',vramGb:32},{unified:false,backend:'cuda',vramGb:24},{unified:false,backend:'cuda',vramGb:16}]};
 assert.deepEqual(memoryInventory(hardware),{ram:32,vram:40,total:72,shared:false});
});
test('a processors built-in graphics with a 0.5 GB share is not a second GPU next to a real card (GUI-TUI QA: "2 GPUs connected" on a PC with one card)',()=>{
 const igpu={name:'AMD Radeon(TM) Graphics',vramGb:0.5,unified:false,backend:'rocm'},card={name:'NVIDIA GeForce RTX 5070',vramGb:11.9,unified:false,backend:'cuda'};
 assert.deepEqual(listedGpus([igpu,card]),[card]);
 assert.deepEqual(listedGpus([card,igpu]),[card]);
 assert.deepEqual(listedGpus([igpu]),[igpu],'a PC with only the built-in graphics keeps it');
 const unread={name:'AMD Radeon RX 9070 XT',vramGb:0,unified:false,backend:'rocm'};
 assert.deepEqual(listedGpus([unread,card]),[unread,card],'a size that was not read (0) is kept');
 const two=[{...card},{...card,name:'NVIDIA RTX A4000',vramGb:16}];
 assert.deepEqual(listedGpus(two),two);
});
