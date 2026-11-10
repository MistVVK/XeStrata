// The app's Load, headless, for the config snapshot.  node app_driver.mjs <world.json> <plain|models>
//
// The host's own createNativeControl runs the Load exactly as the app does.  Fake: the installer process (strata_mcp.py: catalog and hardware), Control's
// model list, engine package, profile save and start (they record what the host hands over).  Real: Control's `drafting plan` and `drafting fetch`, run
// through the Python shim on the same simulated machine, with the download replaced by a recorder.  No engine starts, nothing is downloaded, no network.
import { EventEmitter } from 'node:events';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const [worldFile, which] = process.argv.slice(2);
const spec = JSON.parse(readFileSync(worldFile, 'utf8'));
const root = spec.root;
const repo = process.env.STRATA_SNAPSHOT_REPO;
const python = process.env.STRATA_SNAPSHOT_PYTHON;
const here = dirname(fileURLToPath(import.meta.url));

// every per-user place the host or Control writes to is inside the scratch folder
Object.assign(process.env, {
  HOME: join(root, 'home'), USERPROFILE: join(root, 'home'), APPDATA: join(root, 'roaming'), LOCALAPPDATA: join(root, 'local'),
  XDG_CONFIG_HOME: join(root, 'xdg-config'), XDG_STATE_HOME: join(root, 'state'), XDG_DATA_HOME: join(root, 'xdg-data'),
  STRATA_SETTINGS_FILE: join(root, 'settings.json'), STRATA_COMPONENTS_DIR: join(root, 'components'), STRATA_INSTALL_DIR: join(root, 'install'),
  STRATA_PROFILE_DIR: join(root, 'config', 'profiles'), STRATA_RUNTIME_DIR: join(root, 'runtime'), STRATA_SNAPSHOT_WORLD: worldFile,
});
for (const k of ['STRATA_API_KEY', 'STRATA_ENGINE_BACKEND', 'STRATA_UPDATE_FEED', 'STRATA_NATIVE_REPO']) delete process.env[k];
mkdirSync(process.env.STRATA_COMPONENTS_DIR, { recursive: true });

const { createNativeControl } = await import(pathToFileURL(join(repo, 'apps/host/server/native-control.mjs')).href);
const { createControl } = await import(pathToFileURL(join(repo, 'apps/host/server/control-runtime.mjs')).href);

// ---- the world: the facts the Python side wrote into world-info.json (nothing is retyped here)
const info = JSON.parse(readFileSync(join(root, 'world-info.json'), 'utf8'));
const { tag, family, modelName, model, configPath, hardware, sizeId, backend } = info;

const calls = [];
const real = createControl({ control: { dir: repo, python, cli: join(here, 'control_shim.py') } });
const saved = [];
const pick = (o, keys) => { const out = {}; for (const k of keys) if (o && k in o) out[k] = o[k]; return out; };
const fake = {
  ...real,
  version: '0.2.12',
  async installedModels() {
    return { install_dir: join(root, 'install'), models: [{ id: tag, name: modelName, source_config: configPath, context_tokens: info.context, backend, component_status: 'present', missing: [], issue_codes: [] }] };
  },
  async engine() { return { usable: true, backend, components_dir: join(root, 'components'), engine: { version: '0.1.55', platform: info.platform, backend, exe: info.exe, root: dirname(info.exe) } }; },
  async saveProfile(name, path) { calls.push(['saveProfile', name]); saved.push(JSON.parse(readFileSync(path, 'utf8'))); return {}; },
  async start(name) { calls.push(['start', name]); return { ok: true, state: 'ready' }; },
  async stop() { return { ok: true }; },
  async status() { return { state: 'ready' }; },
};
const draftingPlan = real.draftingPlan.bind(real), draftingFetch = real.draftingFetch.bind(real);
let probe = null;
fake.draftingPlan = async (o = {}) => {
  if (!o.configPath) { probe = draftingPlan(o); return probe; }     // the host's one-off "does this engine serve DFlash2" probe: awaited before the Load (below), not part of the Load's record
  const r = await draftingPlan(o);
  calls.push(['draftingPlan', { mode: o.mode ?? null, noSplit: !!o.noSplit, result: pick(r, ['needed', 'choice', 'drafter', 'gb', 'line', 'note', 'split', 'mtp_needed', 'mtp_here']) }]);
  return r;
};
fake.draftingFetch = async (o = {}) => { const r = await draftingFetch(o); calls.push(['draftingFetch', { mode: o.mode ?? null, noSplit: !!o.noSplit }]); return r; };

const catalog = { families: [{ family, title: info.title, about: 'x', images: true, experimental: false, sizes: [
  { id: sizeId, model, download_gb: 75.8, ram_needed_gb: 60, about: 'x', on_this_pc: 'fits', installed: true, images: true, experimental: false }] }],
  contexts: [8192, 32768, 131072], recommendation: { family, model, context: info.context }, context_note: '' };

const spawnProcess = (command, argv) => {
  if (!String(argv[0]).endsWith('strata_mcp.py')) throw new Error('no process but the installer may be started in packaged mode: ' + command + ' ' + argv.join(' '));
  const child = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter(); child.exitCode = null;
  child.kill = () => { child.exitCode = 0; queueMicrotask(() => child.emit('exit', 0)); return true; };
  child.stdin = { write(line) {
    const message = JSON.parse(line); if (!message.id) return;
    let result;
    if (message.method === 'initialize') result = { protocolVersion: '2024-11-05' };
    else if (message.method === 'tools/list') result = { tools: ['strata_models', 'strata_status', 'strata_install', 'strata_remove'].map(name => ({ name })) };
    else if (message.params.name === 'strata_models') result = { structuredContent: catalog };
    else if (message.params.name === 'strata_status') result = { structuredContent: { hardware, recommendation: catalog.recommendation, install: null, running: [], summary: 'x' } };
    else result = { structuredContent: { summary: 'x', install: { running: false, exit_code: 0 } } };
    queueMicrotask(() => child.stdout.emit('data', Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: message.id, result }) + '\n')));
  } };
  return child;
};

const directory = join(root, 'host-data');
mkdirSync(directory, { recursive: true });
const native = createNativeControl({
  control: fake, directory, spawnProcess, fetcher: async () => Response.json({ status: 'ok', engine: {} }), requests: () => [], modelsDirectory: () => join(root, 'data', 'models'),
  hardwareProbe: () => hardware.gpus.map(g => ({ index: g.index, vram_gb: g.vram_gb })),
});

const until = async (job) => {
  for (let i = 0; i < 1500; i++) { const j = native.state().jobs.find(x => x.id === job.id); if (j && j.state !== 'running') return j; await new Promise(r => setTimeout(r, 20)); }
  throw new Error('the Load did not finish');
};
let exitCode = 0;
try {
  await native.refresh(true);
  native.state();
  for (let i = 0; i < 100 && !probe; i++) await new Promise(r => setTimeout(r, 20));    // the host asks once, at its first state; the Load must start after the answer
  if (probe) { await probe.catch(() => {}); await new Promise(r => setTimeout(r, 100)); }
  // STRATA_SNAPSHOT_APP_ARGS (JSON): more of what the Models screen passes (resourceProfile, vision), for the tests that need it; the golden runs leave it unset
  const extra = process.env.STRATA_SNAPSHOT_APP_ARGS ? JSON.parse(process.env.STRATA_SNAPSHOT_APP_ARGS) : {};
  const args = { ...(which === 'models' ? { model: tag, values: { drafting: 'auto' } } : { model: tag }), ...extra };
  const job = await until(native.operation('load', args));
  if (job.state !== 'done') { console.error('Load ended in state ' + job.state + ': ' + (job.error ?? '')); exitCode = 3; }
  const fetchLog = join(root, 'shim-fetches.jsonl');
  const fetches = existsSync(fetchLog) ? readFileSync(fetchLog, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)) : [];
  const result = {
    command: which === 'models' ? `Models screen: Load ${tag} with Drafting = Auto` : `plain Load ${tag}`,
    control_calls: calls.map(c => (c.length === 2 && typeof c[1] === 'object' ? { call: c[0], ...c[1] } : { call: c[0] })),
    fetches, config_file: saved.length ? saved[saved.length - 1] : null,
    job: { state: job.state, summary: job.summary ?? null, drafterNote: job.drafterNote ?? null },
  };
  console.log('SNAPSHOT-RESULT ' + JSON.stringify(result));
} catch (error) { console.error(error?.stack ?? String(error)); exitCode = 2; }
await native.close();
process.exit(exitCode);
