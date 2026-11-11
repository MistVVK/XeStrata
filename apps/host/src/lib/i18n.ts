import baseEnglish from '../locales/en.json';
import tune_en from '../locales/tune-en.json';
import tools_en from '../locales/tools-en.json';
import fix_en from '../locales/fix-en.json';
/* The strings of the 0.2.7 GitHub items (gh-<lang>.json, scripts/make-gh-locales.py). */
import gh_en from '../locales/gh-en.json';
/* The strings of the 0.2.10 interface changes (ui-<lang>.json, scripts/make-ui-locales.py). */
import ui_en from '../locales/ui-en.json';
/* The strings of Settings > Engine (engine-<lang>.json, scripts/make-engine-locales.py). */
import engine_en from '../locales/engine-en.json';
/* The strings of the model version list (pick-<lang>.json, scripts/make-pick-locales.py). */
import pick_en from '../locales/pick-en.json';
/* The strings of the Drafting choice and the SSD line (df2-<lang>.json, scripts/make-df2-locales.py). */
import df2_en from '../locales/df2-en.json';
/* The strings of the engine update beside a loaded model (upd-<lang>.json, scripts/make-update-locales.py). */
import upd_en from '../locales/upd-en.json';
import r3_en from '../locales/r3-en.json';
/* The picture files Load fetches by itself (pic-<lang>.json; the host words these, strata_control/pictures.py's sentences are the same). */
import pic_en from '../locales/pic-en.json';
/* The GUI feedback round 2 strings (gui-<lang>.json, scripts/make-gui-locales.py). */
import gui_en from '../locales/gui-en.json';
/* The notice for chats the host could not write (sav-<lang>.json, scripts/make-chatsave-locales.py). */
import sav_en from '../locales/sav-en.json';
/* The line shown while a download server has not answered yet (net-<lang>.json, scripts/make-net-locales.py; strata_control/netpolicy.py). */
import net_en from '../locales/net-en.json';

export const LANGUAGES=[{value:'en',label:'English'},{value:'zh',label:'中文（简体）'},{value:'ja',label:'日本語'},{value:'de',label:'Deutsch'},{value:'fr',label:'Français'},{value:'es',label:'Español'},{value:'ru',label:'Русский'},{value:'pt',label:'Português'},{value:'ko',label:'한국어'}] as const;
export type Language=typeof LANGUAGES[number]['value'];
/* The Tune model dialog's strings live in tune-<lang>.json (apps/host/scripts/make-tune-locales.py) so other interface work does not collide with them. */
/* The tool-server, picture and agent-result strings of the tools and images fixes live in tools-<lang>.json for the same reason. */
/* The strings that had no key in the files above (#3479, #3825) and the 0.2.4.3 fixes live in fix-<lang>.json (scripts/make-fix-locales.py). */
const english:Record<string,string>={...baseEnglish,...tune_en,...tools_en,...fix_en,...gh_en,...ui_en,...engine_en,...pick_en,...df2_en,...upd_en,...r3_en,...pic_en,...gui_en,...sav_en,...net_en};
/* English is the key of every string, so it stays in the main file. The other eight languages are one file each, fetched the first time a language is used
   (the saved language before the first screen is drawn, a language picked later when it is picked): most people read one of the nine. */
type Catalog=Record<string,string>;
const loaders:Record<Exclude<Language,'en'>,()=>Promise<{default:Catalog}>>={zh:()=>import('./catalog-zh'),ja:()=>import('./catalog-ja'),de:()=>import('./catalog-de'),fr:()=>import('./catalog-fr'),es:()=>import('./catalog-es'),ru:()=>import('./catalog-ru'),pt:()=>import('./catalog-pt'),ko:()=>import('./catalog-ko')};
const catalogs:Partial<Record<Language,Catalog>>={en:english};
const loading=new Map<Language,Promise<void>>();
export const catalogLoaded=(value:Language)=>!!catalogs[value];
/** Fetch the strings of a language (once). Resolves when they are in; a failed fetch also resolves, the text then stays English and the next call tries again. */
export function loadCatalog(value:Language):Promise<void>{
 if(catalogs[value])return Promise.resolve();
 let request=loading.get(value);
 if(!request){request=loaders[value as Exclude<Language,'en'>]().then(module=>{catalogs[value]=module.default;},()=>undefined).finally(()=>{loading.delete(value);});loading.set(value,request);}
 return request;
}
const entities:Record<string,string>={amp:'&',lt:'<',gt:'>',quot:'"',apos:"'",nbsp:'\u00a0'};
const decode=(text:string)=>text.replace(/&(#x[\da-f]+|#\d+|amp|lt|gt|quot|apos|nbsp);/gi,(whole,key:string)=>key[0]==='#'?String.fromCodePoint(key[1].toLowerCase()==='x'?parseInt(key.slice(2),16):parseInt(key.slice(1),10)):entities[key]??whole);
const ids=new Map(Object.entries(english).map(([id,value])=>[decode(value),id]));
let language:Language='en';
export const validLanguage=(value:unknown):value is Language=>LANGUAGES.some(item=>item.value===value);
export const currentLanguage=()=>language;
export const localeTag=()=>language==='zh'?'zh-CN':language;
/** Switch the language `tr` answers in. The strings of a language that is not loaded yet come in a moment later (English until then): wait on the result before drawing, or use `loadCatalog` first. */
export function setTranslationLanguage(value:Language):Promise<void>{language=value;return loadCatalog(value);}
/** Resolves when the current language's strings are in. */
export const translationsReady=()=>loadCatalog(language);
const escape=(value:string)=>value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
const patterns=Object.entries(english).filter(([,value])=>/\{\d+\}/.test(value)).map(([id,value])=>{
 const slots:number[]=[];let expression='',at=0;
 for(const match of value.matchAll(/\{(\d+)\}/g)){expression+=escape(value.slice(at,match.index))+'([\\s\\S]*?)';slots.push(Number(match[1]));at=match.index!+match[0].length;}
 expression+=escape(value.slice(at));
 return{id,slots,pattern:new RegExp('^'+expression+'$'),specificity:value.replace(/\{\d+\}/g,'').length};
}).sort((a,b)=>b.specificity-a.specificity);
const localizedTemplateSlots:Record<string,number[]>={s1242:[0],s2473:[1,2],s0825:[0,1],s2455:[0],s2456:[0],s7400:[1]};

/** Translate interface copy only. Non-string React children pass through unchanged. */
export function tr<T>(value:T,values?:readonly unknown[]):T{
 if(typeof value!=='string')return value;
 const leading=value.match(/^\s*/)?.[0]??'',trailing=value.match(/\s*$/)?.[0]??'',key=decode(value.trim());
 const dictionary=catalogs[language]??english;
 let id=ids.get(key),slots=values;
 if(!id&&language!=='en')for(const candidate of patterns){const match=candidate.pattern.exec(key);if(match){id=candidate.id;const found:unknown[]=[];candidate.slots.forEach((slot,index)=>{const captured=match[index+1];found[slot]=localizedTemplateSlots[candidate.id]?.includes(slot)?tr(captured):captured;});slots=found;break;}}
 let text=decode(id?(dictionary[id]??english[id as keyof typeof english]):key);
 if(slots)text=text.replace(/\{(\d+)\}/g,(_,index)=>String(slots![Number(index)]??''));
 if(!id&&!values)return decode(value) as T;
 return (leading+text+trailing) as T;
}
