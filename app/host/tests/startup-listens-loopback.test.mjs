import test from 'node:test';
import assert from 'node:assert/strict';
import net from 'node:net';
import { spawn } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

// QA 0213: the host's start-up probed the share port on 0.0.0.0 (index.mjs), which makes Windows Defender Firewall show "has blocked some
// features of Node.js" on the first start of every new install path. Nothing the host does at start (or before sharing is turned on) may listen
// on an address other than the loopback address.
const cwd = fileURLToPath(new URL('..', import.meta.url));
const freePort = () => new Promise(resolve => { const s = net.createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => resolve(p)); }); });

test('every listen() of the host process, from its start until it answers, is on the loopback address', async t => {
  const home = realpathSync(mkdtempSync(join(tmpdir(), 'strata-loop-'))); t.after(() => rmSync(home, { recursive: true, force: true }));
  const log = join(home, 'listen-calls.jsonl');
  const hook = join(home, 'hook.cjs');
  writeFileSync(hook, `const fs = require('fs'); const net = require('net'); const dgram = require('dgram');
const note = (what, args) => { try { fs.appendFileSync(${JSON.stringify(log)}, JSON.stringify({ what, args: args.map(a => (typeof a === 'function' ? null : a)) }) + '\\n'); } catch {} };
const l = net.Server.prototype.listen; net.Server.prototype.listen = function (...a) { note('listen', a); return l.apply(this, a); };
const b = dgram.Socket.prototype.bind; dgram.Socket.prototype.bind = function (...a) { note('bind', a); return b.apply(this, a); };\n`);
  const port = await freePort();
  const env = { ...process.env, STRATA_HOME: home, STRATA_PORT: String(port), STRATA_MODELS_DIR: join(home, 'm'), NODE_OPTIONS: `--require ${hook}` };
  delete env.STRATA_SHARE_PORT; delete env.STRATA_TUNNEL_PORT; delete env.STRATA_BIND;
  const child = spawn(process.execPath, ['server/index.mjs'], { cwd, env });
  let err = ''; child.stderr.on('data', d => { err += d; });
  t.after(() => { child.kill(); });
  let up = false;
  for (let i = 0; i < 300 && !up; i++) { try { up = (await fetch(`http://127.0.0.1:${port}/api/health`)).ok; } catch { await new Promise(r => setTimeout(r, 100)); } }
  assert.ok(up, 'the host answers: ' + err.slice(-400));
  const calls = existsSync(log) ? readFileSync(log, 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l)) : [];
  assert.ok(calls.some(c => c.what === 'listen'), 'the hook saw the host listen');
  for (const c of calls) {
    const host = c.args.find(a => typeof a === 'string');
    // listen(port, host, cb): the host is the string argument; no string means "all addresses", which is what the firewall asks about
    assert.ok(host === '127.0.0.1' || host === '::1', `${c.what}(${JSON.stringify(c.args)}) is not on the loopback address`);
  }
});
