/**
 * QAGUI 0213 #2 (second part): while Strata Control starts slowly, /api/native must say "not known yet" (checking), not "engine not
 * connected, nothing installed", and a read that failed during the start is read again within seconds, not after 45 s.
 */
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { createNativeControl } from '../server/native-control.mjs';

const here = p => fileURLToPath(new URL(p, import.meta.url));
const result = await build({ entryPoints: [here('../src/lib/native.ts')], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent', alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' } });
const { installedListPending } = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);

function slowControl({ failFirst }) {
  const root = mkdtempSync(join(tmpdir(), 'strata-chk-'));
  const install = join(root, 'install'), data = join(root, 'data');
  mkdirSync(install, { recursive: true }); mkdirSync(data, { recursive: true });
  const configPath = join(install, 'strata-iq3_s.json');
  writeFileSync(configPath, JSON.stringify({ exe: 'component:strata-engine', args: [], model_name: 'm', port: 8080 }));
  let asked = 0;
  const control = {
    control: { dir: join(root, 'control'), python: join(root, 'control', 'python', 'python') }, version: '0.2.10',
    async installedModels() {
      asked++;
      if (asked <= failFirst) throw new Error('Strata Control is still starting');
      return { install_dir: install, models: [{ id: 'iq3_s', name: 'm', source_config: configPath, context_tokens: 8192, backend: 'cuda', component_status: 'present', missing: [], issue_codes: [] }] };
    },
    async engine() { return { usable: true, backend: 'cuda', components_dir: join(root, 'c'), engine: { version: '0.1.56', platform: 'linux-x86_64', backend: 'cuda', exe: join(root, 'e', 'strata'), root: join(root, 'e') } }; },
  };
  const catalog = { families: [{ family: 'qwen', title: 'Q', about: 'x', images: false, experimental: false, sizes: [{ id: 'iq3_s', model: 'IQ3_S', download_gb: 1, ram_needed_gb: 1, about: 'x', on_this_pc: 'fits', installed: true }] }], contexts: [8192], recommendation: { family: 'qwen', model: 'IQ3_S' }, context_note: '' };
  const spawnProcess = () => {
    const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter(); child.exitCode = null;
    child.kill = () => { child.exitCode = 0; return true; };
    child.stdin = { write(line) {
      const m = JSON.parse(line); if (!m.id) return;
      const result = m.method === 'initialize' ? { protocolVersion: '2024-11-05' }
        : m.method === 'tools/list' ? { tools: ['strata_models', 'strata_status'].map(name => ({ name })) }
        : m.params.name === 'strata_models' ? { structuredContent: catalog } : { structuredContent: { hardware: { ram_gb: 16, gpus: [] }, install: null, running: [], summary: 'x' } };
      queueMicrotask(() => child.stdout.emit('data', Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: m.id, result }) + '\n')));
    } };
    return child;
  };
  const native = createNativeControl({ control, directory: data, spawnProcess, fetcher: async () => Response.json({ status: 'ok' }), requests: () => [], modelsDirectory: () => join(root, 'models'), hardwareProbe: () => [] });
  return { native, asked: () => asked, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}

test('a read that fails while Control starts is "checking", not "nothing installed", and the page waits on it', async () => {
  const f = slowControl({ failFirst: 1 });
  try {
    const first = await f.native.refresh(false);
    assert.equal(first.available, false);
    assert.equal(first.checking, true, 'not known yet');
    assert.equal(installedListPending(first), true, 'the page shows Checking, no Connect engine, no Download & load');
    const again = await f.native.refresh(true);   // the next poll reads Control again (retry is soon, not after 45 s)
    assert.equal(again.available, true);
    assert.equal(again.checking, false);
    assert.equal(installedListPending(again), false);
    assert.equal(again.installed.models[0].ready, true, 'the downloaded model is shown as ready');
  } finally { f.cleanup(); }
});

test('before the first answer the state is checking; a plain poll after a failed start is not held for 45 s', async () => {
  const f = slowControl({ failFirst: 1 });
  try {
    assert.equal(f.native.state().checking, true);
    await f.native.refresh(false);
    await new Promise(r => setTimeout(r, 3100));
    const next = await f.native.refresh(false);   // an ordinary (not forced) poll, a few seconds later
    assert.equal(next.available, true);
    assert.equal(next.checking, false);
  } finally { f.cleanup(); }
});

test('a Control that keeps failing stops being "checking" so the real error shows', async () => {
  const f = slowControl({ failFirst: 99 });
  try {
    let s;
    for (let i = 0; i < 4; i++) s = await f.native.refresh(true);
    assert.equal(s.checking, false);
    assert.match(s.error, /still starting/);
    assert.equal(installedListPending(s), false);
  } finally { f.cleanup(); }
});
