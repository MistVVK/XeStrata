// #6469: --prompt-cache-spread support for desktop agent compaction on hybrid/Flash-Next models.
import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { NATIVE_FIELDS, validateNativeSettings, withPromptCacheSpread, withoutPromptCacheSpread } from '../server/native-settings.mjs';
import { checkRunConfig } from '../server/run-config-check.mjs';
import { createNativeControl } from '../server/native-control.mjs';

const read = rel => readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');

test('#6469 withPromptCacheSpread / withoutPromptCacheSpread toggle the flag cleanly without duplicates', () => {
  assert.deepEqual(withPromptCacheSpread(['--kv', 'int8']), ['--kv', 'int8', '--prompt-cache-spread']);
  assert.deepEqual(withPromptCacheSpread(['--kv', 'int8', '--prompt-cache-spread']), ['--kv', 'int8', '--prompt-cache-spread']);
  assert.deepEqual(withoutPromptCacheSpread(['--kv', 'int8', '--prompt-cache-spread']), ['--kv', 'int8']);
  assert.deepEqual(withoutPromptCacheSpread(['--kv', 'int8']), ['--kv', 'int8']);
});

test('#6469 prompt_cache_spread is a validated native setting known by run-config-check and hidden for Hy4/Inkling', () => {
  const field = NATIVE_FIELDS.find(f => f.id === 'prompt_cache_spread');
  assert.ok(field && field.type === 'boolean');
  assert.deepEqual(validateNativeSettings({ prompt_cache_spread: true }), { prompt_cache_spread: true });
  assert.deepEqual(validateNativeSettings({ prompt_cache_spread: false }), { prompt_cache_spread: false });
  assert.equal(checkRunConfig(JSON.stringify({ exe: 'e', tokenizer: 't', args: [], prompt_cache_spread: true })).issues.length, 0);
  const ui = read('src/components/NativeSettings.tsx');
  assert.match(ui, /HY4_HIDDEN_FIELDS = new Set\(\[[^\]]*'prompt_cache_spread'/);
  assert.match(ui, /INKLING_HIDDEN_FIELDS = new Set\(\[[^\]]*'prompt_cache_spread'/);
});

test('#6469 desktop packaged mode enables --prompt-cache-spread by default for Flash-Next/Gyro, honours off override, and skips Hy4/Inkling', async t => {
  const root = mkdtempSync(join(tmpdir(), 'strata-pcs-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const dataDir = join(root, 'data');
  const installDir = join(root, 'install');
  mkdirSync(dataDir, { recursive: true });
  mkdirSync(installDir, { recursive: true });
  writeFileSync(join(dataDir, 'native-host.json'), JSON.stringify({ installDir }));

  const gyroPath = join(installDir, 'strata-gyro_s.json');
  const hy4Path = join(installDir, 'strata-hy4.json');
  writeFileSync(gyroPath, JSON.stringify({ exe: 'component:strata-engine', tokenizer: 't', args: ['--kv', 'int8', '--prompt-cache', '6'] }));
  writeFileSync(hy4Path, JSON.stringify({ exe: 'component:strata-hy4', arch: 'hy4flash', tokenizer: 't', args: ['--kv', 'int8'] }));

  const launchedConfigs = [];
  const spawnFakeMcp = () => {
    const child = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.stdin = {
      write(chunk) {
        for (const line of String(chunk).split('\n').filter(Boolean)) {
          const msg = JSON.parse(line);
          if (msg.method === 'initialize') queueMicrotask(() => child.stdout.emit('data', JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: { protocolVersion: '2024-11-05' } }) + '\n'));
          else if (msg.method === 'tools/list') queueMicrotask(() => child.stdout.emit('data', JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: { tools: [{ name: 'strata_status' }, { name: 'strata_models' }] } }) + '\n'));
          else if (msg.method === 'tools/call') {
            const payload = msg.params.name === 'strata_models'
              ? { families: [], contexts: [131072], recommendation: { family: null, model: null } }
              : { summary: 'ok', installed: { models: [] }, hardware: { cpu: { cores: 8 }, ram_gb: 32, gpus: [{ index: 0, name: 'RX 9070 XT', vram_gb: 16 }] } };
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
    installedModels: async () => ({
      install_dir: installDir,
      models: [
        { id: 'gyro_s', name: 'Gyro-S', source_config: gyroPath, component_status: 'present', context_tokens: 131072, backend: 'hip' },
        { id: 'hy4', name: 'Hy4-Flash', source_config: hy4Path, component_status: 'present', context_tokens: 131072, backend: 'cuda' },
      ],
    }),
    engine: async () => ({ usable: true, backend: 'hip', engine: { version: '0.1.65', platform: 'win', backend: 'hip', root: join(root, 'eng') } }),
    saveProfile: async (_name, configPath) => { launchedConfigs.push(JSON.parse(readFileSync(configPath, 'utf8'))); return { ok: true }; },
    start: async () => ({ ok: true }),
    stop: async () => ({ ok: true }),
    status: async () => ({ ok: true, state: 'ready' }),
    killChildren: async () => 0,
  };

  const fetcher = async () => ({ ok: true, json: async () => ({ ok: true, live: { state: 'idle' } }) });
  const native = createNativeControl({ directory: dataDir, spawnProcess: spawnFakeMcp, control: controlStub, fetcher });
  t.after(() => native.close());
  await native.refresh(true);

  const finish = async id => {
    for (let i = 0; i < 300; i++) {
      const job = native.state().jobs.find(j => j.id === id);
      if (job.state !== 'running') return job;
      await new Promise(r => setTimeout(r, 10));
    }
    throw new Error('job did not finish');
  };

  assert.deepEqual(native.promptCacheSpread(), { enabled: true, customized: false });
  assert.equal(native.state().fields.find(f => f.id === 'prompt_cache_spread').supported, true);

  // 1. Default packaged load of Gyro-S includes --prompt-cache-spread
  const j1 = await finish(native.operation('load', { model: 'gyro_s' }).id);
  assert.equal(j1.state, 'done', j1.error);
  assert.ok(launchedConfigs.at(-1).args.includes('--prompt-cache-spread'), 'Gyro-S gets --prompt-cache-spread by default');

  // 2. Hy4-Flash never gets --prompt-cache-spread
  const j2 = await finish(native.operation('load', { model: 'hy4' }).id);
  assert.equal(j2.state, 'done', j2.error);
  assert.ok(!launchedConfigs.at(-1).args.includes('--prompt-cache-spread'), 'Hy4-Flash does not get --prompt-cache-spread');

  // 3. Turning the global setting off removes it from subsequent loads
  assert.deepEqual(native.savePromptCacheSpread(false), { enabled: false, customized: true });
  const j3 = await finish(native.operation('load', { model: 'gyro_s' }).id);
  assert.equal(j3.state, 'done', j3.error);
  assert.ok(!launchedConfigs.at(-1).args.includes('--prompt-cache-spread'), 'disabled global setting omits --prompt-cache-spread');

  // 4. Per-model override true still enables it even when global is off
  native.save('gyro_s', { values: { prompt_cache_spread: true } });
  const j4 = await finish(native.operation('load', { model: 'gyro_s' }).id);
  assert.equal(j4.state, 'done', j4.error);
  assert.ok(launchedConfigs.at(-1).args.includes('--prompt-cache-spread'), 'per-model prompt_cache_spread=true enables flag');
});

test('#6469 Settings screen and host API expose Spread prompt cache checkpoints', () => {
  assert.match(read('server/index.mjs'), /path==='\/api\/native\/prompt-cache-spread'&&method==='GET'/);
  assert.match(read('server/index.mjs'), /path==='\/api\/native\/prompt-cache-spread'&&method==='POST'/);
  assert.match(read('src/screens/Settings.tsx'), /label: 'Spread prompt cache checkpoints'[\s\S]*?<PromptCacheSpreadSetting \/>/);
  assert.match(read('src/components/PromptCacheSpreadSetting.tsx'), /\/api\/native\/prompt-cache-spread/);
});
