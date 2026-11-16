import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { fork } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { findFreePort } from '../server/ports.mjs';

test('the LAN share routes Responses POSTs through the host API and its key check', { timeout: 30000 }, async t => {
  const home = await mkdtemp(join(tmpdir(), 'strata-share-responses-'));
  const port = await findFreePort(0);
  const host = fork('server/index.mjs', [], {
    cwd: new URL('..', import.meta.url),
    stdio: ['ignore', 'ignore', 'inherit', 'ipc'],
    windowsHide: true,
    env: {
      ...process.env,
      STRATA_HOME: home, STRATA_WORKSPACE: join(home, 'workspace'), STRATA_MODELS_DIR: join(home, 'models'),
      STRATA_PORT: String(port), STRATA_BIND: '127.0.0.1', STRATA_SHARE_PORT: '0', STRATA_TUNNEL_PORT: '0',
      STRATA_CONTROL_DIR: join(home, 'no-control'), STRATA_REAL_ENGINE: '0', STRATA_NO_ACQUIRE: '1',
    },
  });
  const exited = once(host, 'exit');
  t.after(async () => {
    if (host.exitCode === null && host.signalCode === null) host.kill();
    await exited;
    await rm(home, { recursive: true, force: true, maxRetries: 3 });
  });
  const [ready] = await Promise.race([
    once(host, 'message', { signal: AbortSignal.timeout(15000) }),
    exited.then(([code]) => { throw new Error(`Host exited during startup (${code})`); }),
  ]);
  assert.equal(ready.type, 'strata-ready');

  const hostUrl = `http://127.0.0.1:${port}`;
  const key = 'test-only-key';
  const json = { 'content-type': 'application/json' };
  const configured = await fetch(hostUrl + '/api/serving', { method: 'PATCH', headers: json, body: JSON.stringify({ apiKey: key }) });
  assert.equal(configured.status, 200);
  const started = await fetch(hostUrl + '/api/serving/share', { method: 'POST', headers: json, body: '{"enabled":true}' });
  assert.equal(started.status, 200);
  const share = new URL((await started.json()).sharing.url);
  share.hostname = '127.0.0.1';

  const url = share.href + '/responses';
  const post = { method: 'POST', body: '{"input":"Hello"}' };
  const headers = { ...json, authorization: `Bearer ${key}` };
  const response = await fetch(url, { ...post, headers });
  assert.equal(response.status, 503, 'the request reaches the Responses handler without a loaded model');
  assert.match((await response.json()).error, /Load a native Strata model/);
  assert.equal((await fetch(url, { ...post, headers: json })).status, 401);
  assert.equal((await fetch(url, { ...post, headers: { ...json, authorization: 'Bearer wrong-key' } })).status, 401);
  assert.equal((await fetch(url, { headers })).status, 404);
  assert.equal((await fetch(url + '/example', { ...post, headers })).status, 404);
  assert.equal((await fetch(share.origin + '/api/serving', { headers })).status, 404);
});
