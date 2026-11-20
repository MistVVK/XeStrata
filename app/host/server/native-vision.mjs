/** Prepared vision files stay in the source config; activation belongs to a loaded instance. */
import { isHy4, isHy4Family, hy4VisionShard } from './hy4.mjs';

export function runtimeVision(config,requested){
 const mode=requested??(config.ui_vision_enabled===false?'no':config.ui_vision_device);
 if(mode!==undefined&&!['yes','no','cpu','gpu'].includes(mode))throw new Error('Choose whether to enable picture reading.');
 if(isHy4(config)){
  if(mode&&mode!=='no'&&!hy4VisionShard(config))throw new Error('Prepare this model’s vision files before enabling picture reading.');
  return undefined;
 }
 if(mode==='no')return undefined;
 if(mode===undefined)return config.vision;
 if(!config.vision)throw new Error('Prepare this model’s vision files before enabling picture reading.');
 if(mode==='cpu'&&config.vision.cpu_exe){const {cpu_exe,...rest}=config.vision;return {...rest,exe:cpu_exe,gpu:false};} // AMD: the CPU reader program, not the HIP one with the GPU hidden
 return {...config.vision,...(mode==='cpu'?{gpu:false}:mode==='gpu'?{gpu:true}:{})};
}

/**
 * The catalog with each picture-capable size marked as ready (`images`) or not (`imagesCapable` stays true either way). The Qwen engines
 * read pictures with a separate program (strata-vision) that has to sit in the engine folder; Hy4-Flash reads pictures inside strata-hy4 itself.
 */
export function markPictures(catalog,visionReader){
 if(!catalog||!Array.isArray(catalog.families))return catalog;
 return {...catalog,families:catalog.families.map(f=>{
  const builtin=isHy4Family(f.family);
  const ready=(s)=>!!s.images&&(builtin||visionReader===true);
  const sizes=(f.sizes??[]).map(s=>({...s,imagesCapable:!!s.images,images:ready(s)}));
  return {...f,imagesCapable:!!f.images,images:sizes.length?sizes.some(s=>s.images):!!f.images&&(builtin||visionReader===true),sizes};
 })};
}
