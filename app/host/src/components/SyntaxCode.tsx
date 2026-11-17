import { useEffect, useMemo, useState } from 'react';
import { codeLanguage } from '../lib/code-language';
import type { CodeToken } from '../lib/syntax';
/** The Prism grammars are a separate file (src/lib/syntax.ts), fetched after the first screen is up (App.tsx asks for it a few seconds in) or by the first code block, whichever comes first. Until it is here a block shows its text uncoloured. */
type Highlighter = typeof import('../lib/syntax');
let loaded: Highlighter | null = null;
let loading: Promise<Highlighter | null> | null = null;
export function loadHighlighter(): Promise<Highlighter | null> {
 loading ??= import('../lib/syntax').then(module => (loaded = module), () => { loading = null; return null; });
 return loading;
}
const plainLines = (text: string): CodeToken[][] => text.split('\n').map(part => (part ? [{ text: part, classes: [] }] : []));
export function SyntaxCode({text,language='',filename,lineNumbers=false,className=''}:{text:string;language?:string;filename?:string;lineNumbers?:boolean;className?:string}){
 const resolved=filename?codeLanguage(filename,true):codeLanguage(language);
 const [highlighter,setHighlighter]=useState<Highlighter|null>(loaded);
 useEffect(()=>{if(highlighter)return;let live=true;void loadHighlighter().then(module=>{if(live&&module)setHighlighter(module);});return()=>{live=false;};},[highlighter]);
 const lines=useMemo(()=>highlighter?highlighter.highlightedLines(text,resolved):plainLines(text),[highlighter,text,resolved]);
 return <pre className={`syntax-code ${lineNumbers?'code-preview':''} ${className}`} data-language={resolved}><code>{lines.map((line,i)=><span className="syntax-line" key={i}>{lineNumbers?<span className="syntax-gutter" aria-hidden="true">{i+1}</span>:null}<span className="syntax-text">{line.map((part,j)=>part.classes.length?<span className={part.classes.map(c=>`syntax-${c}`).join(' ')} key={j}>{part.text}</span>:part.text)}{!lineNumbers&&i<lines.length-1?'\n':''}</span></span>)}</code></pre>;
}
