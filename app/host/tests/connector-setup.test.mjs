import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { connectorAsset, createConnectorInstaller, verifyConnector } from '../server/cloudflared-install.mjs';

test('automatic setup selects native executables and rejects unsupported platforms', () => {
  assert.equal(connectorAsset('darwin','arm64'),'cloudflared-darwin-arm64.tgz');
  assert.equal(connectorAsset('linux','x64'),'cloudflared-linux-amd64');
  assert.equal(connectorAsset('win32','x64'),'cloudflared-windows-amd64.exe');
  assert.throws(()=>connectorAsset('win32','arm64'),/not available/);
  assert.throws(()=>connectorAsset('unknown','x64'),/not available/);
});

test('corrupt or unverified downloads cannot become executable', () => {
  const bytes=Buffer.from('official connector fixture');
  const digest=`sha256:${createHash('sha256').update(bytes).digest('hex')}`;
  assert.doesNotThrow(()=>verifyConnector(bytes,digest));
  assert.throws(()=>verifyConnector(Buffer.from('tampered'),digest),/could not be verified/);
  assert.throws(()=>verifyConnector(bytes,null),/could not be verified/);
});

/** Waits until the fake download service has been asked; fails (instead of spinning for ever) when it never is. */
async function untilAsked(isAsked){
  const deadline=Date.now()+10000;
  while(!isAsked()){
    if(Date.now()>deadline)throw new Error('the installer never asked the download service (it found a connector, or stopped before the request)');
    await new Promise(resolve=>setTimeout(resolve,5));
  }
}

// The installer first looks for a connector that is already on the computer. The look is pinned to Strata's own empty folder
// (search:false, an empty environment): a cloudflared installed on the machine that runs the test (PATH, /usr/local/bin, ~/.local/bin)
// would otherwise be used, nothing would be downloaded, and the test would wait for ever for a request that never comes.
test('failed concurrent setup shares one download, cleans up and can be retried', {timeout:30000}, async () => {
  const directory=await mkdtemp(join(tmpdir(),'strata-connector-test-'));
  let requests=0, release;
  const installer=createConnectorInstaller({directory,platform:'linux',arch:'x64',env:{},home:directory,search:false,fetchImpl:async()=>{
    requests++;
    await new Promise(resolve=>{release=resolve;});
    return {ok:false};
  }});
  try {
    const first=installer.ensure(),second=installer.ensure();
    assert.equal(first,second);
    await untilAsked(()=>release);
    release();
    const results=await Promise.allSettled([first,second]);
    assert.ok(results.every(result=>result.status==='rejected'));
    assert.equal(requests,1);
    assert.deepEqual(await readdir(directory),[]);
    release=null;
    const retry=installer.ensure();
    await untilAsked(()=>release);
    release();
    await assert.rejects(retry,/download service/);
    assert.equal(requests,2);
    assert.deepEqual(await readdir(directory),[]);
  } finally {await rm(directory,{recursive:true,force:true});}
});
