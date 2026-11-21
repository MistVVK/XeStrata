import './helpers/isolate-profile.mjs';   // a test run never writes the owner's real Strata profile
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, realpathSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir as osTmpdir } from 'node:os';
import { bundledPython, createControl, findControl } from '../server/control-runtime.mjs';
import { createNativeControl } from '../server/native-control.mjs';

// ARCFIX (0.2.13): a Control-only feed update (`strata update`, or Settings > Updates > Strata) writes <components>/control/active.json.
// The packaged app must then run the model setup (tools/strata_mcp.py and the setup.py it starts) from THAT Control, not from the bundled one.
const isWin = process.platform === 'win32';
const tmp = () => realpathSync(osTmpdir());

function controlDir(parent, name, version) {
  const dir = join(parent, name);
  mkdirSync(join(dir, 'python', ...(isWin ? [] : ['bin'])), { recursive: true });
  mkdirSync(join(dir, 'strata_control'), { recursive: true });
  mkdirSync(join(dir, 'tools'), { recursive: true });
  mkdirSync(join(dir, 'serve'), { recursive: true });
  writeFileSync(bundledPython(dir), 'x');
  writeFileSync(join(dir, 'strata.py'), '# cli');
  writeFileSync(join(dir, 'tools', 'strata_mcp.py'), '# mcp');
  writeFileSync(join(dir, 'setup.py'), '# setup');
  writeFileSync(join(dir, 'serve', 'server.py'), 'port\nparallel\n/v1/responses');
  writeFileSync(join(dir, 'BUNDLE.json'), JSON.stringify({ version }));
  return dir;
}

function fixture() {
  const root = mkdtempSync(join(tmp(), 'strata-arcfix-'));
  const install = join(root, 'install');                          // the AppImage's / installer's folder: <install>/host + <install>/control
  mkdirSync(join(install, 'host'), { recursive: true });
  const bundled = controlDir(install, 'control', '0.1.59');
  const components = join(root, 'components');
  mkdirSync(join(components, 'control'), { recursive: true });
  const updated = controlDir(join(components, 'control'), '0.1.59.1', '0.1.59.1');
  writeFileSync(join(components, 'control', 'active.json'), JSON.stringify({ version: '0.1.59.1', folder: '0.1.59.1', sha256: 'x' }));
  return { root, install, bundled, components, updated, hostDir: join(install, 'host') };
}

test('a four-part Control (0.1.59.1) from the feed beats the bundled 0.1.59, whether the shell names the bundled one or it sits beside the host', () => {
  const f = fixture();
  try {
    // the Tauri shell sets STRATA_CONTROL_DIR to the bundled folder (app/desktop/src-tauri/src/main.rs apply_runtime_policy)
    const viaShell = findControl({ env: { STRATA_CONTROL_DIR: f.bundled }, hostDir: f.hostDir, components: f.components });
    assert.equal(viaShell.dir, f.updated);
    assert.equal(viaShell.bundle.version, '0.1.59.1');
    assert.equal(findControl({ env: {}, hostDir: f.hostDir, components: f.components }).dir, f.updated);
    // the same Control, the same version: the bundled one stays (a tie goes to the first), so a feed that offers 0.1.59 again changes nothing
    writeFileSync(join(f.updated, 'BUNDLE.json'), JSON.stringify({ version: '0.1.59' }));
    writeFileSync(join(f.components, 'control', 'active.json'), JSON.stringify({ version: '0.1.59', folder: '0.1.59.1' }));
    assert.equal(findControl({ env: { STRATA_CONTROL_DIR: f.bundled }, hostDir: f.hostDir, components: f.components }).dir, f.bundled);
  } finally { rmSync(f.root, { recursive: true, force: true }); }
});

test('the model setup of the packaged app runs from the updated Control: strata_mcp.py (and with it --src, the folder setup.py is started from) is the updated one', async () => {
  const f = fixture();
  try {
    const found = findControl({ env: { STRATA_CONTROL_DIR: f.bundled }, hostDir: f.hostDir, components: f.components });
    const spawned = [];
    const spawnProcess = (command, args, options) => {
      spawned.push({ command, args, options });
      const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter(); child.exitCode = null;
      child.kill = () => true;
      child.stdin = { write(line) {
        const message = JSON.parse(line); if (!message.id) return;
        const result = message.method === 'tools/list' ? { tools: [{ name: 'strata_status' }] } : { protocolVersion: '2024-11-05' };
        queueMicrotask(() => child.stdout.emit('data', Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: message.id, result }) + '\n')));
      } };
      return child;
    };
    const data = join(f.root, 'data');
    const control = { ...createControl({ control: found, run: async () => ({}) }),
      async installedModels() { return { install_dir: f.root, models: [] }; },
      async engine() { return { usable: true, backend: 'sycl', engine: { version: '0.1.59', platform: 'linux-x86_64', backend: 'sycl', exe: '/e/strata', root: '/e' } }; } };
    const native = createNativeControl({ control, directory: data, spawnProcess, fetcher: async () => Response.json({}) });
    try { await native.refresh(true); } finally { await native.close(); }
    const installer = spawned.find(s => String(s.args[0]).endsWith('strata_mcp.py'));
    assert.ok(installer, 'the host started the installer process');
    assert.equal(installer.command, bundledPython(f.updated));
    assert.equal(installer.args[0], join(f.updated, 'tools', 'strata_mcp.py'));
    assert.equal(installer.args[installer.args.indexOf('--src') + 1], f.updated);
    assert.equal(installer.options.cwd, f.updated);
    assert.equal(installer.options.env.STRATA_CONTROL_DIR, f.updated);
  } finally { rmSync(f.root, { recursive: true, force: true }); }
});
