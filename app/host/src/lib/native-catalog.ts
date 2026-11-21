import type {NativeFamily,NativeSize} from './native';
import {tr} from './i18n';
import {nvfp4Version} from './native-names';
export interface CatalogVersion extends NativeSize{sourceFamily:string;edition:string;images:boolean}
export interface CatalogModel extends Omit<NativeFamily,'sizes'>{sizes:CatalogVersion[]}
export function nativeVersionLabel(model:string){const nv=nvfp4Version(model);if(nv)return tr(nv.name);return tr(({Q2_0:'Fastest',IQ2_XS:'Compact',IQ3_XXS:'Balanced',IQ3_S:'High quality',IQ1_M:'Coding','Gyro-S':'Smallest','Gyro-M':'Small memory','SQ-64':'Flagship','SQ-96':'High quality','SQ-128':'Highest quality','SQ-20':'Lossless','SQ-15':'Flagship','SQ-10':'Compact'} as Record<string,string>)[model]??model);}
export const catalogVersionLabel=(version:CatalogVersion)=>{const label=nativeVersionLabel(version.model);return [tr(version.edition),label===version.model&&version.edition?'':label].filter(Boolean).join(' · ');};
/**
 * Models whose picture files Load fetches by itself (strata_control/pictures.py): the Flash-Next GGUF families, which share the original's reader (Swift has its own file).
 * NVFP4 packs and Qwen3.6 keep the installer for their pictures. With these the Load of a model that is on this computer never runs the installer again just
 * because the picture files are missing.
 */
export const PICTURE_FETCH_FAMILIES = ['qwen', 'coder', 'unsloth', 'gyro', 'swift', 'hy4-flash', 'inkling-2-mini'];
export const fetchesPictures = (version: { sourceFamily?: string } | undefined, packaged: boolean | undefined) => !!version && packaged === true && PICTURE_FETCH_FAMILIES.includes(version.sourceFamily ?? '');
/** The one line that says how a version differs from its neighbours (the three NVFP4 versions); empty for the rest. */
export const versionNote=(model:string)=>{const nv=nvfp4Version(model);return nv?tr(nv.note):'';};
/** Whether a display group has a KV-format choice (the Models screen's "Context memory"): Qwen3.6-35B-A3B has none, its engine path ignores --kv and
 *  setup records no format for it, so the select is not shown there (#5919, owner's decision). Every other group keeps it. */
export const hasKvChoice=(group?:string)=>group!=='qwen3.6'&&group!==HY4_FAMILY&&group!==INKLING_FAMILY;
/** Tencent Hy4-Flash (setup.py HY4_FAMILIES, its own engine program): a display group of its own; its KV is fp16 and its builds (SQ-64, SQ-96, SQ-128; there is no SQ-32) are placed by the author's rank-cut rule (native-fit.ts hy4Fit). */
export const HY4_FAMILY='hy4-flash';
/** Thinking Machines Inkling-2-Mini (setup.py INKLING_FAMILIES, its own shape engine strata-inkling). */
export const INKLING_FAMILY='inkling-2-mini';
/** Display groups are distinct from the engine's installation family IDs. */
export function nativeCatalogModels(families:NativeFamily[]):CatalogModel[]{
 if(!families.length)return[];
 const groups:CatalogModel[]=[{family:'qwen3.8',title:'Qwen3.8-Flash-Next',about:'',experimental:false,images:false,sizes:[]},{family:'qwen3.6',title:'Qwen3.6-35B-A3B',about:'',experimental:false,images:false,sizes:[]}];
 for(const family of families){
  if(family.family===HY4_FAMILY||family.family===INKLING_FAMILY){groups.push({family:family.family,title:family.title,about:family.about??'',experimental:false,images:family.images,sizes:family.sizes.map(size=>({...size,sourceFamily:family.family,edition:'',images:family.images}))});continue;}
  if(family.feed){groups.push({family:family.family,title:family.title,about:family.about??'',experimental:false,images:family.images,available:family.available,unavailable_reason:family.unavailable_reason,feed:true,sizes:family.sizes.map(size=>({...size,sourceFamily:family.family,edition:'',images:family.images}))});continue;}
  const target=/qwen3[._-]?6|35b[-_]?a3b/i.test(family.title+' '+family.family)?groups[1]:['qwen','swift','coder','unsloth','gyro'].includes(family.family)||/qwen3[._-]?8|flash.next/i.test(family.title)?groups[0]:undefined;
  if(!target)continue;
  const edition=target===groups[0]?({swift:'Swift 1.5',coder:'Coder',unsloth:'Unsloth',gyro:'Gyro'} as Record<string,string>)[family.family]??'':'';
  target.images ||= family.images;
  target.sizes.push(...family.sizes.map(size=>({...size,sourceFamily:family.family,edition,images:family.images})));
 }
 return groups;
}

/** Where the first-run model step starts: a version already on this computer and ready (the recommended one if it is among them), else the catalog's recommendation. Null when neither is in the list (the caller then keeps its own fallback). */
export function startingChoice(families:CatalogModel[],readyIds:string[],pick?:{family?:string|null;model?:string|null}|null):{family:string;size:string}|null{
 const ready=new Set(readyIds);
 const recommended=pick?.family&&pick.model?families.flatMap(f=>f.sizes.filter(s=>s.model===pick.model&&(s.sourceFamily===pick.family||f.family===pick.family)).map(s=>({family:f.family,size:s.id}))):[];
 const have=recommended.find(r=>ready.has(r.size))??families.flatMap(f=>f.sizes.filter(s=>ready.has(s.id)).map(s=>({family:f.family,size:s.id})))[0];
 return have??recommended[0]??null;
}
