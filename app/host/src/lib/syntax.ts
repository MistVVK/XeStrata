/** Bundled grammars: previews work offline and highlighted text stays plain React text. */
import Prism from 'prismjs';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-jsx';
import 'prismjs/components/prism-tsx';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-yaml';
import 'prismjs/components/prism-rust';
import 'prismjs/components/prism-go';
import 'prismjs/components/prism-java';
import 'prismjs/components/prism-c';
import 'prismjs/components/prism-cpp';
import 'prismjs/components/prism-csharp';
import 'prismjs/components/prism-sql';
Prism.manual = true;
import { codeLanguage } from './code-language';
export { codeLanguage };
export interface CodeToken {text:string;classes:string[]}
export function highlightedLines(text:string,language:string):CodeToken[][]{
 const lines:CodeToken[][]=[[]];
 function append(value:string,classes:string[]){const parts=value.split('\n');for(let i=0;i<parts.length;i++){if(i)lines.push([]);if(parts[i])lines.at(-1)!.push({text:parts[i],classes});}}
 function walk(tokens:Prism.TokenStream,classes:string[]=[]){
  if(typeof tokens==='string'){append(tokens,classes);return;}
  if(Array.isArray(tokens)){tokens.forEach(token=>walk(token,classes));return;}
  const aliases=typeof tokens.alias==='string'?[tokens.alias]:tokens.alias??[];
  walk(tokens.content,[...classes,tokens.type,...aliases]);
 }
 const grammar=Prism.languages[language];
 if(!grammar||text.length>120000){append(text,[]);return lines;}
 try{walk(Prism.tokenize(text,grammar));}catch{lines.splice(0,lines.length,[]);append(text,[]);}
 return lines;
}
