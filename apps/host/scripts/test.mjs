#!/usr/bin/env node
/** Enumerate test files explicitly so npm test works without shell glob expansion. */
import {readdirSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..');
const files=readdirSync(resolve(root,'tests')).filter(name=>name.endsWith('.test.mjs')).sort().map(name=>resolve(root,'tests',name));
/* Every test file runs against scratch copies of the per-user places the host writes to (the owner's real profile is never touched). */
import {mkdtempSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
const scratch=mkdtempSync(join(tmpdir(),'strata-host-suite-profile-'));
const env={...process.env,STRATA_SETTINGS_FILE:join(scratch,'app-settings.json'),STRATA_COMPONENTS_DIR:join(scratch,'components'),LOCALAPPDATA:join(scratch,'Local'),APPDATA:join(scratch,'Roaming'),XDG_DATA_HOME:join(scratch,'xdg-data'),XDG_STATE_HOME:join(scratch,'xdg-state'),XDG_CONFIG_HOME:join(scratch,'xdg-config')};
/* Run from PowerShell 7 (GitHub's Windows runners), PSModulePath names PowerShell 7's module folders first and a Windows PowerShell 5.1 child
   that a test starts cannot load Get-Acl / Get-FileHash from them: keep only the folders that are not PowerShell 7's. */
if(process.platform==='win32'&&env.PSModulePath){const ps7=p=>{const l=p.toLowerCase().replace(/\//g,'\\');return l.includes('\\powershell\\')&&!l.includes('\\windowspowershell\\');};env.PSModulePath=env.PSModulePath.split(';').filter(p=>p&&!ps7(p)).join(';');}
/* A test that never finishes fails after STRATA_TEST_TIMEOUT_MS (default 10 minutes) instead of holding the whole run for ever; extra arguments (a reporter, --test-name-pattern) go to node --test. */
const timeout=Number(process.env.STRATA_TEST_TIMEOUT_MS)||600000;
const child=spawn(process.execPath,['--test',`--test-timeout=${timeout}`,...process.argv.slice(2),...files],{cwd:root,stdio:'inherit',env});
child.on('error',error=>{console.error(error.message);process.exitCode=1;});
child.on('exit',(code,signal)=>{process.exitCode=code??(signal?1:0);});
