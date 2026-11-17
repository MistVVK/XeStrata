import './helpers/isolate-profile.mjs';   // a test run never writes the owner's real Strata profile
/**
 * The packaged app (STRATA_PACKAGED=1, STRATA_REAL_ENGINE=1, as the app shell starts the host) never starts a llama.cpp `llama-server`.
 * 0.2.14 did: a GGUF imported through /api/local-models was loaded through a llama-server named by STRATA_LLAMA_SERVER (POST /load), and at
 * the next start the host loaded it again by itself, also from a llama-server found on PATH, with nothing clicked. A fake llama-server that
 * writes a marker line stands in both places; neither the load route nor the start-up path may ever run it.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { delimiter, join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { ggufFixture } from './helpers/local-gguf.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const delay = ms => new Promise(r => setTimeout(r, ms));
async function freePort() { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

/** A llama-server look-alike: one marker line per start, answers /health like the real one, gone after 20 s. */
function fakeServer(dir, marker) {
  const js = join(dir, 'fake-llama-server.cjs');
  writeFileSync(js, `const fs=require('fs'),http=require('http');
fs.appendFileSync(${JSON.stringify(marker)}, JSON.stringify({pid:process.pid,argv:process.argv.slice(2)})+'\\n');
const i=process.argv.indexOf('--port');const port=i>0?Number(process.argv[i+1]):8080;
http.createServer((q,s)=>{s.setHeader('content-type','application/json');s.end(q.url.startsWith('/health')?'{"status":"ok"}':'{}');}).listen(port,'127.0.0.1');
setTimeout(()=>process.exit(0),20000);
`);
  const unix = join(dir, 'llama-server');
  writeFileSync(unix, `#!${process.execPath}\nrequire(${JSON.stringify(js)});\n`);
  chmodSync(unix, 0o755);
  writeFileSync(join(dir, 'llama-server.cmd'), `@"${process.execPath}" "${js}" %*\r\n`);
  return process.platform === 'win32' ? join(dir, 'llama-server.cmd') : unix;
}

async function scenario(t, where) {
  const home = mkdtempSync(join(tmpdir(), `strata-no-llama-${where}-`));
  t.after(() => rmSync(home, { recursive: true, force: true }));
  const bin = join(home, 'bin'), marker = join(home, 'marker.jsonl');
  for (const d of ['bin', 'src', 'ws', 'components']) mkdirSync(join(home, d), { recursive: true });
  const fake = fakeServer(bin, marker);
  const gguf = join(home, 'src', 'Small-Q4_K_M.gguf');
  ggufFixture(gguf, { arch: 'qwen3', name: 'no-llama fixture' });
  const port = await freePort();
  const env = { ...process.env, STRATA_PACKAGED: '1', STRATA_REAL_ENGINE: '1', NODE_ENV: 'production', STRATA_HOME: join(home, 'strata'),
    STRATA_MODELS_DIR: join(home, 'strata', 'models'), STRATA_WORKSPACE: join(home, 'ws'), STRATA_COMPONENTS_DIR: join(home, 'components'),
    STRATA_PORT: String(port), STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()) };
  delete env.STRATA_LLAMA_SERVER; delete env.STRATA_API_KEY;
  if (where === 'env') env.STRATA_LLAMA_SERVER = fake;
  else { const key = Object.keys(env).find(k => k.toUpperCase() === 'PATH') ?? 'PATH'; env[key] = bin + delimiter + (env[key] ?? ''); }
  const base = `http://127.0.0.1:${port}`;
  const call = async (path, method = 'GET', body) => {
    const r = await fetch(base + path, { method, headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
    return { status: r.status, body: await r.json().catch(() => null) };
  };
  const starts = () => existsSync(marker) ? readFileSync(marker, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)) : [];
  let host = null;
  const up = async () => {
    const log = openSync(join(home, 'host.log'), 'a');
    host = spawn(process.execPath, ['server/index.mjs'], { cwd: root, env, stdio: ['ignore', log, log] });
    for (let i = 0; i < 150; i++) { try { if ((await call('/api/health')).status === 200) return; } catch { /* not up yet */ } await delay(100); }
    assert.fail('the host did not start: ' + readFileSync(join(home, 'host.log'), 'utf8').slice(-600));
  };
  const down = async () => { if (!host) return; const h = host; host = null; h.kill(); await Promise.race([new Promise(r => h.once('exit', r)), delay(8000)]); };
  t.after(async () => { await down(); for (const s of starts()) { try { process.kill(s.pid); } catch { /* gone */ } } });

  await up();
  const imported = await call('/api/local-models', 'POST', { path: gguf });
  assert.equal(imported.status, 200, JSON.stringify(imported.body));
  const id = imported.body.variantId;
  const load = await call(`/api/models/${encodeURIComponent(id)}/load`, 'POST');
  await delay(1500);
  assert.equal(load.status, 409, JSON.stringify(load.body));
  assert.match(load.body?.error ?? '', /own engine/);
  assert.deepEqual(starts(), [], 'POST /load started the llama-server');
  assert.equal((await call('/api/engine/status')).body?.running, false);

  await down();
  await up();
  await delay(2500);   // (0.2.14 began its start-up load 0.7 s after the host was up)
  assert.deepEqual(starts(), [], 'the host start started the llama-server');
  const library = (await call('/api/models')).body;
  assert.equal(library?.loaded ?? null, null);
  assert.notEqual(library?.current, id, 'an imported GGUF that never loaded became the remembered model');
  assert.equal((await call('/api/engine/status')).body?.running, false);
}

test('the packaged host never starts a llama-server named by STRATA_LLAMA_SERVER (load route and host start)', { timeout: 90000 }, t => scenario(t, 'env'));
test('the packaged host never starts a llama-server found on PATH (load route and host start)', { timeout: 90000 }, t => scenario(t, 'path'));
