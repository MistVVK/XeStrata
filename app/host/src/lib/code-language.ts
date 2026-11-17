/** The language name of a code fence or a file name. Kept apart from the highlighter (syntax.ts, which carries the Prism grammars) so a code block can name its language before the grammars have loaded. */
const aliases:Record<string,string>={ts:'typescript',tsx:'tsx',mts:'typescript',cts:'typescript',js:'javascript',jsx:'jsx',mjs:'javascript',cjs:'javascript',py:'python',pyw:'python',md:'markdown',mdx:'markdown',markdown:'markdown',html:'markup',htm:'markup',xml:'markup',svg:'markup',json:'json',css:'css',sh:'bash',shell:'bash',bash:'bash',zsh:'bash',yml:'yaml',yaml:'yaml',rs:'rust',go:'go',java:'java',c:'c',h:'c',cc:'cpp',cpp:'cpp',cxx:'cpp',hpp:'cpp',cs:'csharp',csharp:'csharp',sql:'sql',typescript:'typescript',javascript:'javascript',python:'python',rust:'rust'};
export function codeLanguage(value:string,filename=false){
 const input=filename?(value.split(/[?#]/)[0].split('.').at(-1)??''):value.trim().split(/\s/)[0].toLowerCase();
 return aliases[input.toLowerCase()]??'plain';
}
