// A link the host refuses to open (javascript:, data:, file:, vbscript:, a malformed address, a file outside the previews) is a bad
// request: 400 with the reason, not 500 (seen on the Strix tester build, 11-16: javascript: answered 500). Nothing is launched.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import net from 'node:net';
import { fileURLToPath } from 'node:url';
import { openArtifact } from '../server/artifacts.mjs';

const REFUSED = ['javascript:alert(1)', 'data:text/html,<b>x</b>', 'file:///C:/Windows/win.ini', 'vbscript:msgbox(1)', 'not a url', 'ftp://example.com/x'];

test('openArtifact marks every refused request as a 400 and launches nothing', async () => {
  const launched = [];
  const deps = { examplesDir: tmpdir(), previewsDir: tmpdir(), platform: 'win32', launch: async (...a) => { launched.push(a); } };
  for (const url of REFUSED) await assert.rejects(openArtifact({ action: 'open', url }, deps), e => e.status === 400, url);
  await assert.rejects(openArtifact({ action: 'reveal', url: 'https://example.com/' }, deps), e => e.status === 400);
  await assert.rejects(openArtifact({ action: 'run' }, deps), e => e.status === 400);
  await assert.rejects(openArtifact({ action: 'open', path: '/etc/passwd' }, deps), e => e.status === 400);
  await assert.rejects(openArtifact({ action: 'open', name: 'bad.app', content: 'x' }, deps), e => e.status === 400);
  assert.deepEqual(launched, []);
  await openArtifact({ action: 'open', url: 'https://example.com/' }, deps);
  assert.equal(launched.length, 1, 'an https link still opens');
});

const root = fileURLToPath(new URL('..', import.meta.url));
async function freePort() { const s = net.createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

test('the host answers a refused link with 400 and its reason', { timeout: 60_000 }, async t => {
  const home = await mkdtemp(join(tmpdir(), 'strata-host-open-'));
  const port = await freePort();
  const env = {
    ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port),
    STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()), STRATA_REAL_ENGINE: '1',
    STRATA_SETTINGS_FILE: join(home, 'app-settings.json'), STRATA_COMPONENTS_DIR: join(home, 'components'),
    LOCALAPPDATA: join(home, 'Local'), APPDATA: join(home, 'Roaming'), XDG_DATA_HOME: join(home, 'xdg-data'),
    XDG_STATE_HOME: join(home, 'xdg-state'), XDG_CONFIG_HOME: join(home, 'xdg-config'),
  };
  const host = spawn(process.execPath, ['server/index.mjs'], { cwd: root, stdio: 'ignore', env });
  const exit = new Promise(r => host.once('exit', r));
  t.after(async () => { host.kill('SIGKILL'); await exit; await rm(home, { recursive: true, force: true }).catch(() => {}); });
  let up = false;
  for (let i = 0; i < 150 && !up; i++) { try { up = (await fetch(`http://127.0.0.1:${port}/api/health`)).ok; } catch { await new Promise(r => setTimeout(r, 100)); } }
  assert.ok(up, 'host started');
  for (const url of ['javascript:alert(1)', 'data:text/html,x', 'file:///etc/passwd', 'vbscript:msgbox(1)']) {
    const r = await fetch(`http://127.0.0.1:${port}/api/artifacts/open`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ action: 'open', url }) });
    assert.equal(r.status, 400, url);
    assert.match((await r.json()).error, /Only http and https addresses can be opened/);
  }
});
