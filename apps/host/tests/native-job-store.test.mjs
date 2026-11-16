import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { syncBuiltinESMExports } from 'node:module';
import { join } from 'node:path';
import { createJobStore } from '../server/native-job-store.mjs';
import { createNativeControl } from '../server/native-control.mjs';

const root = fs.mkdtempSync(join(import.meta.dirname, 'test-data-'));
test.after(() => { assert.equal(join(root, '..'), import.meta.dirname); fs.rmSync(root, { recursive: true, force: true }); });
const fault = () => Object.assign(new Error('simulated Windows open failure'), {code:'UNKNOWN',errno:-4094,syscall:'open'});
function fixture(ioPatch = {}) {
  const directory = fs.mkdtempSync(join(root, 'case-'));
  const path = join(directory, 'native-jobs.json');
  const queue = new Map(), warnings = [];
  let clock = 100000, next = 0;
  const store = createJobStore(path, {
    io: {...fs, ...ioPatch}, now:()=>clock, warn:x=>warnings.push(x),
    setTimer(fn, delay) { const handle={id:++next,unref(){}}; queue.set(handle,{fn,delay});return handle; },
    clearTimer(handle) { queue.delete(handle); },
  });
  return {store,path,queue,warnings,advance(ms){clock+=ms;},retry(){const [handle,item]=queue.entries().next().value;queue.delete(handle);item.fn();return item.delay;}};
}

test('unchanged jobs are not repeatedly written', () => {
  let writes=0;
  const f=fixture({writeFileSync(...args){writes++;return fs.writeFileSync(...args);}});
  f.store.save([{id:'one',state:'done'}]);
  f.store.save([{id:'one',state:'done'}]);
  assert.equal(writes,1);
  assert.deepEqual(JSON.parse(fs.readFileSync(f.path)),[{id:'one',state:'done'}]);
  f.store.close();
});

test('UNKNOWN open failure preserves the old JSON; retry saves newest state', () => {
  let blocked=true;
  const f=fixture({writeFileSync(...args){if(blocked)throw fault();return fs.writeFileSync(...args);}});
  fs.writeFileSync(f.path,'[{"id":"old"}]');
  assert.doesNotThrow(()=>f.store.save([{id:'new'}]));
  assert.deepEqual(JSON.parse(fs.readFileSync(f.path)),[{id:'old'}]);
  f.store.save([{id:'newest'}]);
  blocked=false;
  assert.equal(f.retry(),250);
  assert.deepEqual(JSON.parse(fs.readFileSync(f.path)),[{id:'newest'}]);
  assert.equal(f.warnings.length,2);
  assert.match(f.warnings[0],/UNKNOWN.*-4094.*inference continues/);
  assert.match(f.warnings[1],/recovered/);
  assert.equal(f.queue.size,0);
  f.store.close();
});

test('rename failure preserves old JSON and cleans only the temporary file', () => {
  const f=fixture({renameSync(){throw Object.assign(fault(),{code:'EPERM',syscall:'rename'});}});
  fs.writeFileSync(f.path,'[{"id":"old"}]');
  f.store.save([{id:'new'}]);
  assert.deepEqual(JSON.parse(fs.readFileSync(f.path)),[{id:'old'}]);
  assert.deepEqual(fs.readdirSync(join(f.path,'..')),['native-jobs.json']);
  f.store.close();
  assert.equal(f.queue.size,0);
});

test('persistent failures have bounded retries, cooldown and rate-limited warnings', () => {
  let writes=0;
  const f=fixture({writeFileSync(){writes++;throw fault();}});
  f.store.save([]);
  assert.equal(f.retry(),250);assert.equal(f.retry(),1000);assert.equal(f.retry(),3000);
  assert.equal(writes,4);assert.equal(f.queue.size,0);assert.equal(f.warnings.length,1);
  f.store.save([{id:'later'}]);assert.equal(writes,4);
  f.advance(30001);f.store.save([{id:'later'}]);assert.equal(writes,5);
  f.store.close();assert.equal(f.queue.size,0);
});

test('shutdown flushes the latest pending state and cancels retries', () => {
  let blocked=true;
  const f=fixture({writeFileSync(...args){if(blocked)throw fault();return fs.writeFileSync(...args);}});
  f.store.save([{id:'one'}]);f.store.save([{id:'two'}]);blocked=false;f.store.close();
  assert.deepEqual(JSON.parse(fs.readFileSync(f.path)),[{id:'two'}]);
  assert.equal(f.queue.size,0);
});

test('real controller polling survives the exact filesystem failure and still publishes state', async () => {
  const directory=fs.mkdtempSync(join(root,'controller-'));
  const original=fs.writeFileSync;
  let changes=0,controller;
  fs.writeFileSync=function(path,...args){if(String(path).startsWith(join(directory,'native-jobs.json')))throw fault();return original(path,...args);};
  syncBuiltinESMExports();
  try {
    controller=createNativeControl({directory,defaultRepo:directory,changed(){changes++;}});
    await assert.doesNotReject(controller.poll());
    assert.equal(changes,1);
  } finally {fs.writeFileSync=original;syncBuiltinESMExports();}
  await new Promise(resolve=>setTimeout(resolve,350));
  assert.deepEqual(JSON.parse(fs.readFileSync(join(directory,'native-jobs.json'))),[]);
  await controller.poll();assert.equal(changes,2);
  await controller.close();
});

test('an unexpected polling callback failure does not leave polling stuck or reject', async () => {
  const directory=fs.mkdtempSync(join(root,'boundary-'));
  let changes=0;
  const controller=createNativeControl({directory,defaultRepo:directory,changed(){if(++changes===1)throw new Error('simulated callback failure');}});
  await assert.doesNotReject(controller.poll());
  await assert.doesNotReject(controller.poll());
  assert.equal(changes,2);
  await controller.close();
});
