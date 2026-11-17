import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..');
const repo=resolve(root,'..','..');
/* The app and the standalone server page (serve/web) each serve their own components.css. They are the same file: change both together. */
test('serve/web/components.css is byte-identical to the app kit components.css',()=>{
 const app=readFileSync(resolve(root,'src/styles/kit/components.css'));
 const web=readFileSync(resolve(repo,'terminal/serve/web/components.css'));
 assert.ok(app.equals(web),'src/styles/kit/components.css and serve/web/components.css differ; edit both with the same change');
});
/* A package.json script that runs a file under scripts/ must point at a file that exists (the removed sync:kit pointed at a design folder that is not in the repository). */
test('every npm script that runs a scripts/ file points at a file that exists',()=>{
 const scripts=JSON.parse(readFileSync(resolve(root,'package.json'),'utf8')).scripts;
 for(const [name,command] of Object.entries(scripts)){
  for(const m of command.matchAll(/\bnode\s+(scripts\/[\w./-]+)/g))assert.ok(existsSync(resolve(root,m[1])),`npm run ${name} runs ${m[1]}, which does not exist`);
 }
});
