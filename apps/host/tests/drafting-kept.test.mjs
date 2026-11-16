// A Drafting choice made on the Models screen is kept for the next start (0.2.15 release test T6-app, p1004, C2 app): the Load saved
// "drafting": "off" into the model's run config (prepareDrafter / rememberSetting), and the write at ready then put back the copy of the
// run config the Load had read before that, so the key was gone: after an app restart the page showed Auto and the next Load ran DFlash2.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const source = readFileSync(fileURLToPath(new URL('../server/native-control.mjs', import.meta.url)), 'utf8').replace(/\r\n/g, '\n');

test('the Load saves the Drafting choice before the engine starts', () => {
  const load = source.slice(source.indexOf('async function loadEngine'), source.indexOf('const fingerprint=createHash'));
  assert.match(load, /await prepareDrafter\(job,args,cfg,configFile,drafts,\{noSplit\}\)/);
  const prepare = source.slice(source.indexOf('async function prepareDrafter'), source.indexOf('function rememberSetting'));
  assert.match(prepare, /rememberSetting\(configFile,'drafting',asked===null\|\|asked==='auto'\?null:asked\)/);
});

test('the run-config write at ready starts from the file as it is now, so the saved choice is kept', () => {
  const at = source.indexOf("instance.status='ready';instance.phase='idle';await noteDrafting(job,instance,key);");
  assert.ok(at > 0, 'the ready branch of loadEngine');
  const ready = source.slice(at, source.indexOf('emit();ensurePoll();', at));
  assert.match(ready, /writeModelConfig\(record\(args\.model\)\.path,\{\.\.\.savedNow\(record\(args\.model\)\.path,config\),/,
    'the write at ready reads the run config again (it holds the Drafting choice the Load saved)');
  assert.doesNotMatch(ready, /writeModelConfig\(record\(args\.model\)\.path,\{\.\.\.config,/, 'not the copy read before the choice was saved');
  const helper = source.slice(source.indexOf('function savedNow('), source.indexOf('function writeModelConfig('));
  assert.match(helper, /checkedSettingsConfig\(readFileSync\(path,'utf8'\)\)/);
  assert.match(helper, /catch\{return fallback;\}/);
});
