// 0217 lane FIX-APP, #6244: changing one option on a model's settings screen rewrote its run config and dropped the drafting mode (dflash2) and
// turned STRATA_RESIDENT_PIN from "1" to "0". Saving the screen's own values back unchanged, or one changed, must leave what is not on that screen.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {nativeFixture} from './helpers/native-fixture.mjs';

const DFLASH = ['--draft','dflash2','--draft-model','D:/x/qwen3.8-flash-next-dflash2.q4_0-selq4.pack','--draft-fallback','none','--mtp-draft-vocab','D:/x/draft_vocab_latin.bin'];

function startConfig(f){
  const path=join(f.repo,'strata-iq2_xs.json');
  const config={...JSON.parse(readFileSync(path,'utf8')),drafting:'dflash2',args:['--max-context','32768',...DFLASH],args_written:['--max-context','32768',...DFLASH],
    env:{PATH:'/native/bin',STRATA_RESIDENT_PIN:'1',STRATA_PARTIAL_PIN:'1',STRATA_PARTIAL_PIN_GIB:'4'}};
  writeFileSync(path,JSON.stringify(config));
  return path;
}

test('saving the settings screen with one option changed keeps the drafting mode, its arguments and the other environment entries',async()=>{
  const f=nativeFixture();try{
    const path=startConfig(f);await f.control.refresh();
    const screen=f.control.settings('iq2_xs');
    f.control.save('iq2_xs',{values:{...screen.values,parallel:2}});
    const saved=JSON.parse(readFileSync(path,'utf8'));
    assert.equal(saved.drafting,'dflash2');
    assert.deepEqual(saved.args.slice(saved.args.indexOf('--draft')),DFLASH.slice(0));
    assert.equal(saved.env.STRATA_RESIDENT_PIN,'1');
    assert.equal(saved.env.STRATA_PARTIAL_PIN,'1');
    assert.equal(saved.env.STRATA_PARTIAL_PIN_GIB,'4');
    assert.equal(saved.parallel,2);
  }finally{await f.cleanup();}
});

test('saving the screen with nothing changed leaves the run config as it was',async()=>{
  const f=nativeFixture();try{
    const path=startConfig(f);const before=JSON.parse(readFileSync(path,'utf8'));await f.control.refresh();
    f.control.save('iq2_xs',{values:f.control.settings('iq2_xs').values});
    const saved=JSON.parse(readFileSync(path,'utf8'));
    assert.equal(saved.drafting,before.drafting);
    assert.deepEqual(saved.args,before.args);
    assert.deepEqual(saved.env,before.env);
  }finally{await f.cleanup();}
});
