// #6475: Configurable model server port in GUI Settings and stopping blind port 8080 probes.
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { validModelPort } from '../server/ports.mjs';
import { installerEnvironment } from '../server/control-runtime.mjs';
import { createNativeControl } from '../server/native-control.mjs';

const read = rel => readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');

test('#6475 validModelPort accepts 1024..65535 and rejects invalid/privileged values', () => {
  assert.equal(validModelPort(18500), true);
  assert.equal(validModelPort(1024), true);
  assert.equal(validModelPort(65535), true);
  assert.equal(validModelPort(8080), true);
  assert.equal(validModelPort(80), false);
  assert.equal(validModelPort(1023), false);
  assert.equal(validModelPort(65536), false);
  assert.equal(validModelPort(18500.5), false);
  assert.equal(validModelPort('18500'), false);
  assert.equal(validModelPort(null), false);
});

test('#6475 installerEnvironment suppresses default port 8080 probing unless explicitly configured to 8080', () => {
  const ctrl = { dir: '/fake/control' };
  const autoEnv = installerEnvironment(ctrl, { env: {}, installDir: '/fake/install', components: '/fake/components' });
  assert.equal(autoEnv.STRATA_NO_DEFAULT_PORT, '1', 'default desktop mode suppresses blind port 8080 probes');
  assert.equal(autoEnv.STRATA_MODEL_PORT, undefined);

  const customEnv = installerEnvironment(ctrl, { env: {}, installDir: '/fake/install', components: '/fake/components', modelPort: 19200 });
  assert.equal(customEnv.STRATA_MODEL_PORT, '19200');
  assert.equal(customEnv.STRATA_NO_DEFAULT_PORT, '1');

  const explicit8080 = installerEnvironment(ctrl, { env: {}, installDir: '/fake/install', components: '/fake/components', modelPort: 8080 });
  assert.equal(explicit8080.STRATA_MODEL_PORT, '8080');
  assert.equal(explicit8080.STRATA_NO_DEFAULT_PORT, undefined);
});

test('#6475 native-control persists modelPort, rejects bad ports, and passes port env to strata_mcp.py', async t => {
  const root = mkdtempSync(join(tmpdir(), 'strata-model-port-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const dataDir = join(root, 'data');
  const installDir = join(root, 'install');
  mkdirSync(installDir, { recursive: true });
  const spawnedEnvs = [];

  const spawnFakeMcp = (_cmd, _args, opts) => {
    spawnedEnvs.push(opts.env);
    const child = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.stdin = {
      write(chunk) {
        for (const line of String(chunk).split('\n').filter(Boolean)) {
          const msg = JSON.parse(line);
          if (msg.method === 'initialize') {
            queueMicrotask(() => child.stdout.emit('data', JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: { protocolVersion: '2024-11-05' } }) + '\n'));
          } else if (msg.method === 'tools/list') {
            queueMicrotask(() => child.stdout.emit('data', JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: { tools: [{ name: 'strata_status' }, { name: 'strata_models' }] } }) + '\n'));
          } else if (msg.method === 'tools/call') {
            const payload = msg.params.name === 'strata_models'
              ? { families: [], contexts: [32768], recommendation: { family: null, model: null } }
              : { summary: 'ok', installed: { models: [] }, hardware: { cpu: { cores: 8 }, ram_gb: 32, gpus: [] } };
            queueMicrotask(() => child.stdout.emit('data', JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: { structuredContent: payload } }) + '\n'));
          }
        }
      },
    };
    child.kill = () => {};
    return child;
  };

  const controlStub = {
    control: { dir: join(root, 'ctrl'), python: process.execPath, cli: join(root, 'ctrl', 'strata.py') },
    version: '0.2.19',
    installedModels: async () => ({ install_dir: installDir, models: [] }),
    engine: async () => ({ usable: true, backend: 'cuda', engine: { version: '0.1.65', platform: 'win', backend: 'cuda', root: join(root, 'eng') } }),
    killChildren: async () => 0,
  };

  const native = createNativeControl({ directory: dataDir, spawnProcess: spawnFakeMcp, control: controlStub });
  t.after(() => native.close());

  assert.deepEqual(native.modelPort(), { port: null, defaultPort: 18500, activePorts: [] });
  assert.throws(() => native.saveModelPort(80), /1024 and 65535/);
  assert.throws(() => native.saveModelPort(70000), /1024 and 65535/);

  await native.refresh(true);
  assert.equal(spawnedEnvs.at(-1).STRATA_NO_DEFAULT_PORT, '1');
  assert.equal(spawnedEnvs.at(-1).STRATA_MODEL_PORT, undefined);

  assert.deepEqual(native.saveModelPort(19450), { port: 19450, defaultPort: 18500, activePorts: [] });
  await native.refresh(true);
  assert.equal(spawnedEnvs.at(-1).STRATA_MODEL_PORT, '19450');
  assert.equal(spawnedEnvs.at(-1).STRATA_NO_DEFAULT_PORT, '1');

  // Persists across restart
  const native2 = createNativeControl({ directory: dataDir, spawnProcess: spawnFakeMcp, control: controlStub });
  t.after(() => native2.close());
  assert.equal(native2.modelPort().port, 19450);

  // Clearing returns to automatic
  assert.equal(native2.saveModelPort(null).port, null);
  assert.equal(JSON.parse(readFileSync(join(dataDir, 'native-host.json'), 'utf8')).modelPort, null);
});

test('#6475 Settings screen and host API expose Model server port', () => {
  assert.match(read('server/index.mjs'), /path==='\/api\/native\/model-port'&&method==='GET'/);
  assert.match(read('server/index.mjs'), /path==='\/api\/native\/model-port'&&method==='POST'/);
  assert.match(read('src/screens/Settings.tsx'), /label: 'Model server port'[\s\S]*?<ModelPortSetting \/>/);
  assert.match(read('src/components/ModelPortSetting.tsx'), /\/api\/native\/model-port/);
});
