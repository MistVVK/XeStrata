/**
 * The App Host's bridge to Strata Control, the Python part of the product.
 *
 * In the installed app Control ships as a component next to the host (<install>/control): a pinned CPython with
 * hash-locked wheels, strata_control, strata_cli and the API server. The host never uses the user's Python, never runs
 * pip, git or a compiler, and never downloads anything to start a model: it asks Control, and Control starts the engine
 * under its owner runtime (loopback TCP + 256-bit token on Windows, a private socket elsewhere; a Job Object that ends
 * the server with its host; leases that stop a second owner).
 *
 * What lives here: where Control and the engine component folder are, a runner for the CLI (JSON in, JSON out, a
 * timeout, no shell), and the four operations the app needs: which engine this PC would use, which models are
 * installed, start a saved profile, stop it.
 */
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';

const HERE = dirname(fileURLToPath(import.meta.url));

export function bundledPython(dir, platform = process.platform) {
  return platform === 'win32' ? join(dir, 'python', 'python.exe') : join(dir, 'python', 'bin', 'python3');
}

/** The per-user component folder, the same rule as strata_control/engine_component.py components_dir(). */
export function componentsDirectory({ env = process.env, platform = process.platform, home = homedir() } = {}) {
  if (env.STRATA_COMPONENTS_DIR) return env.STRATA_COMPONENTS_DIR;
  if (platform === 'win32') return join(env.LOCALAPPDATA || join(home, 'AppData', 'Local'), 'Strata', 'components');
  return join(env.XDG_DATA_HOME || join(home, '.local', 'share'), 'strata', 'components');
}

/**
 * A Control that the component updater installed (`strata update` writes `<components>/control/active.json` after the new
 * bundle ran once). Null when there is none.
 */
export function updatedControlDirectory({ env = process.env, platform = process.platform, components = null } = {}) {
  try {
    const base = components ?? componentsDirectory({ env, platform });
    const pointer = JSON.parse(readFileSync(join(base, 'control', 'active.json'), 'utf8').replace(/^﻿/, ''));
    if (typeof pointer?.folder !== 'string' || !/^[\w.+-]+$/.test(pointer.folder)) return null;
    return { dir: join(base, 'control', pointer.folder), version: typeof pointer.version === 'string' ? pointer.version : null };
  } catch { return null; }
}

const versionKey = value => String(value ?? '').split('.').map(part => Number.parseInt(part, 10) || 0);
function newer(a, b) {
  for (let i = 0; i < Math.max(a.length, b.length); i++) if ((a[i] ?? 0) !== (b[i] ?? 0)) return (a[i] ?? 0) > (b[i] ?? 0);
  return false;
}

/**
 * Is there a Control component to use? STRATA_CONTROL_DIR names one (the Tauri shell sets it to the bundled one), the
 * `control` folder beside `host` is the installed app's, and `<components>/control/active.json` is one the component
 * updater installed. Each must be whole; the newest version wins, so an app update that ships a newer Control is never
 * held back by an older updated one, and an updated Control is never lost to an older bundled one. A tie goes to the first
 * in that order. Returns null in a source checkout (the host then keeps its older paths).
 */
export function findControl({ env = process.env, platform = process.platform, exists = existsSync, hostDir = resolve(HERE, '..'), components = null } = {}) {
  const updated = updatedControlDirectory({ env, platform, components });
  const candidates = [{ dir: env.STRATA_CONTROL_DIR }, { dir: join(hostDir, '..', 'control') }, ...(updated ? [updated] : [])]
    .filter(c => c.dir).map(c => ({ ...c, dir: resolve(c.dir) }));
  let best = null;
  for (const candidate of candidates) {
    const python = bundledPython(candidate.dir, platform);
    if (!(exists(python) && exists(join(candidate.dir, 'strata.py')) && exists(join(candidate.dir, 'strata_control')))) continue;
    let bundle = null;
    try { bundle = JSON.parse(readFileSync(join(candidate.dir, 'BUNDLE.json'), 'utf8').replace(/^﻿/, '')); } catch { /* optional */ }
    const found = { dir: candidate.dir, python, cli: join(candidate.dir, 'strata.py'), bundle, key: versionKey(bundle?.version ?? candidate.version) };
    if (!best || newer(found.key, best.key)) best = found;
  }
  if (!best) return null;
  const { key, ...control } = best;
  return control;
}

/**
 * The environment of the server the host starts from a source checkout (#3851): the key travels in the run configuration, so a
 * STRATA_API_KEY the host itself was started with is removed. Setting it to an empty string made serve/server.py stop with
 * "an API key was given but it is empty".
 */
export function sourceServerEnvironment(env = process.env) {
  const next = { ...env, PYTHONUNBUFFERED: '1', PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' };
  delete next.STRATA_API_KEY;
  return next;
}

/** The environment Control runs in: nothing from the user's Python, and acquiring forbidden. */
export function controlEnvironment(control, { env = process.env, components = componentsDirectory({ env }), acquire = false } = {}) {
  const next = { ...env };
  for (const key of ['PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP', 'VIRTUAL_ENV', 'STRATA_PYTHON', 'STRATA_NATIVE_PYTHON', 'STRATA_API_KEY']) delete next[key];
  const result = {
    ...next,
    PYTHONNOUSERSITE: '1', PYTHONUTF8: '1', PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8',
    STRATA_NO_ACQUIRE: '1', STRATA_COMPONENTS_DIR: components,
    // The app's own settings file, beside its components: Control and the installer never read or write the file an older
    // command-line Strata keeps in %APPDATA% / ~/.config (its list of Strata folders), so an old install is never adopted.
    ...(components ? { STRATA_SETTINGS_FILE: env.STRATA_SETTINGS_FILE || join(dirname(components), 'app-settings.json') } : {}),
    ...(control?.dir ? { STRATA_CONTROL_DIR: control.dir } : {}),
  };
  // Only the explicit update step (`strata update`) may acquire; every other Control process keeps the switch on.
  if (acquire) delete result.STRATA_NO_ACQUIRE;
  return result;
}

/**
 * The environment of the installer process (tools/strata_mcp.py): Control's, with acquiring still forbidden for the process
 * itself. Only the install job it starts when the user asks to download a model lifts that for its own children.
 */
export function installerEnvironment(control, { env = process.env, installDir, components = componentsDirectory({ env }) } = {}) {
  return { ...controlEnvironment(control, { env, components }), STRATA_CONTROL_INSTALL: '1', ...(installDir ? { STRATA_INSTALL_DIR: installDir } : {}) };
}

export class ControlError extends Error {
  constructor(message, { code = 'control_failed', exitCode = null, output = null } = {}) {
    super(message);
    this.name = 'ControlError';
    this.code = code;
    this.exitCode = exitCode;
    this.output = output;
  }
}

/**
 * Small Control programs that are a module of strata_control and not a `strata` command. They run on Control's interpreter with Control's folder first on the
 * path (the folder is the working directory), so they work whatever the interpreter does with the script folder.
 */
const BOOT = {
  pictures: 'import os,sys;sys.path.insert(0,os.getcwd());from strata_control import pictures;sys.exit(pictures.main(sys.argv[1:]))',
};

/** Run one CLI command of Control (or, with `boot`, one of its modules). Resolves { code, json, text, stderr }; rejects only if it cannot start or times out. */
export function runCli(control, args, { env = process.env, timeoutMs = 60_000, spawnProcess = spawn, components, acquire = false, onLine = null, signal = null, boot = null } = {}) {
  return new Promise((resolveRun, reject) => {
    const child = spawnProcess(control.python, boot ? ['-c', BOOT[boot], ...args] : [control.cli, ...args], {
      cwd: control.dir, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
      env: controlEnvironment(control, { env, acquire, ...(components ? { components } : {}) }),
    });
    if (signal) {
      if (signal.aborted) { try { child.kill(); } catch { /* gone */ } }
      else signal.addEventListener('abort', () => { try { child.kill(); } catch { /* gone */ } }, { once: true });
    }
    let out = '', err = '', done = false, pending = '';
    const finish = (fn, value) => { if (done) return; done = true; clearTimeout(alarm); fn(value); };
    const alarm = setTimeout(() => {
      try { child.kill(); } catch { /* already gone */ }
      finish(reject, new ControlError(`Strata Control did not answer within ${Math.round(timeoutMs / 1000)} s (${args.slice(0, 2).join(' ')}).`, { code: 'control_timeout' }));
    }, timeoutMs);
    child.stdout.on('data', c => {
      out += c; if (out.length > 4_000_000) child.kill();
      if (onLine) {                                       // JSON lines (--output jsonl): each complete line is one event
        pending += c;
        let at;
        while ((at = pending.indexOf('\n')) >= 0) {
          const line = pending.slice(0, at).trim(); pending = pending.slice(at + 1);
          if (!line.startsWith('{')) continue;
          try { onLine(JSON.parse(line)); } catch { /* not an event */ }
        }
      }
    });
    child.stderr.on('data', c => { err += c; if (err.length > 200_000) err = err.slice(-100_000); });
    child.on('error', e => finish(reject, new ControlError(`Strata Control could not start: ${e.message}`, { code: 'control_missing' })));
    child.on('close', code => {
      let json = null;
      const text = out.trim();
      if (onLine) {                                       // the last line is the result record
        const last = text.split(/\r?\n/).filter(line => line.startsWith('{')).pop();
        if (last) { try { json = JSON.parse(last); } catch { /* not JSON */ } }
      } else if (text) { try { json = JSON.parse(text.slice(text.indexOf('{'))); } catch { /* not JSON: caller reads text */ } }
      finish(resolveRun, { code, json, text, stderr: err.trim() });
    });
  });
}

/** Control's exit 7 means two things: an engine is serving, or another update holds the update lock (strata_control/updater.py update_lock). The lock is told by its text. */
export const UPDATE_LOCK_HELD = /another update is already running/i;
const failure = (result, fallback) => {
  const message = result.json?.error?.message ?? result.json?.error ?? result.json?.notice ?? (result.stderr.split(/\r?\n/).filter(Boolean).pop() || fallback);
  return new ControlError(message, { exitCode: result.code, output: result.json, ...(UPDATE_LOCK_HELD.test(String(message)) ? { code: 'update_busy' } : {}) });
};

/** Operations the app uses. `run` is injectable so the host's tests need no Python. */
export function createControl({ control = findControl(), env = process.env, run = null } = {}) {
  if (!control) return null;
  const call = run ?? ((args, options) => runCli(control, args, { env, ...options }));
  const feedArgs = feed => (feed ? ['--feed', feed] : []);
  return {
    control,
    version: control.bundle?.version ?? null,
    /**
     * Use the newest whole Control there is now (findControl again), after an update installed one: true when it changed. Every later call runs the new one.
     * The caller makes sure nothing of the old one is running (native-control.mjs switchControl).
     */
    reload({ find = findControl } = {}) {
      const next = find({ env });
      if (!next || resolve(next.dir) === resolve(control.dir)) return false;
      control = next;
      this.control = next;
      this.version = next.bundle?.version ?? null;
      return true;
    },
    /**
     * The explicit update step. These are the only calls that let Control reach the network: `acquire` removes
     * STRATA_NO_ACQUIRE for that one process. Starting, serving, listing and rolling back never do.
     */
    async updateCheck({ feed = env.STRATA_UPDATE_FEED ?? null, appVersion = env.STRATA_APP_VERSION ?? null } = {}) {
      const result = await call(['update', '--check', '--output', 'json', ...feedArgs(feed), ...(appVersion ? ['--app-version', appVersion] : [])], { timeoutMs: 90_000, acquire: true });
      if (result.json?.ok) return result.json;
      throw failure(result, 'The update check failed.');
    },
    async update({ feed = env.STRATA_UPDATE_FEED ?? null, engine = true, control: withControl = true, appVersion = env.STRATA_APP_VERSION ?? null, force = false, stopModel = false, onEvent = null, signal = null } = {}) {
      const args = ['update', '--output', 'jsonl', ...feedArgs(feed), ...(engine ? [] : ['--no-engine']), ...(withControl ? [] : ['--no-control']),
        ...(appVersion ? ['--app-version', appVersion] : []), ...(force ? ['--force'] : []), ...(stopModel ? ['--stop-model'] : [])];
      const result = await call(args, { timeoutMs: 3 * 3600_000, acquire: true, onLine: onEvent ?? (() => {}), signal });
      if (result.json?.ok) return result.json;
      const failed = result.json?.results?.find?.(r => r.status === 'failed');
      if (failed) throw Object.assign(new ControlError(failed.error, { exitCode: failed.exit_code ?? result.code, output: result.json }), { code: result.code === 7 && !UPDATE_LOCK_HELD.test(failed.error ?? '') ? 'engine_serving' : result.code === 7 ? 'update_busy' : 'update_failed' });
      throw failure(result, 'The update failed.');
    },
    /** Install an engine package the person downloaded and chose (the offline path): no network; Control checks the package's own signature. */
    async installEngineFile(path, { onEvent = null, signal = null } = {}) {
      const result = await call(['components', 'install-file', path, '--output', 'jsonl'], { timeoutMs: 3600_000, onLine: onEvent ?? (() => {}), signal });
      if (result.json?.ok) return { ok: true, results: [result.json] };
      throw failure(result, 'This file could not be installed.');
    },
    async rollbackEngine() {
      const result = await call(['components', 'rollback', '--output', 'json'], { timeoutMs: 120_000 });
      if (result.json?.ok) return result.json;
      throw failure(result, 'The engine could not be rolled back.');
    },
    /** The engines installed on this computer, by family (`strata components engines`): the one list the Terminal prints too. Reads files only. */
    async engines() {
      const result = await call(['components', 'engines', '--output', 'json'], { timeoutMs: 60_000 });
      if (result.json?.ok) return result.json;
      throw failure(result, 'The installed engines could not be read.');
    },
    /** Switch to an installed engine (`strata components use`): nothing is downloaded. `root` is only for a DeepSeek engine that lives in more than one place. */
    async useEngine({ family, folder }) {
      const result = await call(['components', 'use', folder, '--family', family, '--output', 'json'], { timeoutMs: 600_000 });
      if (result.json?.ok) return result.json;
      throw failure(result, 'The engine could not be switched.');
    },
    /** Delete an installed engine (`strata components remove`); Control refuses the one in use. */
    async removeEngine({ family, folder }) {
      const result = await call(['components', 'remove', folder, '--family', family, '--output', 'json'], { timeoutMs: 600_000 });
      if (result.json?.ok) return result.json;
      throw failure(result, 'The engine could not be deleted.');
    },
    async engine({ backend = null } = {}) {
      const result = await call(['components', 'engine', '--output', 'json', ...(backend ? ['--backend', backend] : [])], { timeoutMs: 30_000 });
      if (result.json) return result.json;                  // exit 5 still carries the report and the notice
      throw failure(result, 'The engine check failed.');
    },
    async installedModels({ installDir = null } = {}) {
      const result = await call(['models', 'list', '--installed', ...(installDir ? ['--install-dir', installDir] : []), '--output', 'json'], { timeoutMs: 60_000 });
      if (result.code !== 0 || !result.json) throw failure(result, 'The installed models could not be read.');
      return result.json;
    },
    /** What a local GGUF folder or file holds, without touching it: the model it is, and why it cannot be used when it cannot. */
    async importPlan(path) {
      const result = await call(['models', 'import', path, '--plan', '--output', 'json'], { timeoutMs: 120_000 });
      if (result.json && result.code === 0) return result.json.plan ?? result.json;
      // A set that cannot be used is still a checked set: Control answers (exit 5, ok:false) with the plan and the reasons, and the page says them in
      // words ("Strata can't run these files yet"). Only a check that could not run at all is a failure.
      if (result.json?.plan && Array.isArray(result.json.plan.issues)) return result.json.plan;
      throw failure(result, 'The files could not be checked.');
    },
    /** What picking `to` as the models folder means (sizes, free space, which choices are possible, why not): nothing is changed. */
    async planModelsMove(from, to) {
      const result = await call(['models', 'move', to, ...(from ? ['--from', from] : []), '--plan', '--output', 'json'], { timeoutMs: 120_000 });
      if (result.json?.plan) return result.json.plan;
      throw failure(result, 'The folder could not be checked.');
    },
    /**
     * Move the model files to `to` (copied, checked, then the old copies removed). `onEvent` gets the progress lines; a file at `cancelFile` stops it
     * safely. Resolves with the summary (done / stopped, what moved, what is left); a failure is a ControlError whose `output` carries the details.
     */
    async moveModels(from, to, { installDir = null, cancelFile = null, onEvent = null } = {}) {
      const args = ['models', 'move', to, ...(from ? ['--from', from] : []), ...(installDir ? ['--install-dir', installDir] : []), ...(cancelFile ? ['--cancel-file', cancelFile] : []), '--output', 'jsonl'];
      const result = await call(args, { timeoutMs: 24 * 3600_000, onLine: onEvent ?? (() => {}) });
      if (result.json?.ok && result.json.event === 'result') return result.json;
      const error = result.json?.error;
      throw Object.assign(new ControlError(error?.message ?? (result.stderr.split(/\r?\n/).filter(Boolean).pop() || 'The models could not be moved.'), { exitCode: result.code, output: result.json }), { code: error?.code ?? 'move_failed', details: error?.details ?? '' });
    },
    /** A move of the models folder that the last run left unfinished (`strata models move --status`): null, or {state, from, to, bytes}. */
    async interruptedModelsMove(dataFolder) {
      const result = await call(['models', 'move', '--status', ...(dataFolder ? ['--from', dataFolder] : []), '--output', 'json'], { timeoutMs: 30_000 });
      if (result.json?.ok) return result.json.interrupted ?? null;
      throw failure(result, 'The models folder could not be checked.');
    },
    /** "Finish it": the same move again, from where it stopped. Same events and result as moveModels. */
    async resumeModelsMove(dataFolder, { installDir = null, cancelFile = null, onEvent = null } = {}) {
      const args = ['models', 'move', '--resume', ...(dataFolder ? ['--from', dataFolder] : []), ...(installDir ? ['--install-dir', installDir] : []), ...(cancelFile ? ['--cancel-file', cancelFile] : []), '--output', 'jsonl'];
      const result = await call(args, { timeoutMs: 24 * 3600_000, onLine: onEvent ?? (() => {}) });
      if (result.json?.ok && result.json.event === 'result') return result.json;
      const error = result.json?.error;
      throw Object.assign(new ControlError(error?.message ?? (result.stderr.split(/\r?\n/).filter(Boolean).pop() || 'The models could not be moved.'), { exitCode: result.code, output: result.json }), { code: error?.code ?? 'move_failed', details: error?.details ?? '' });
    },
    /** "Leave it": keep the old folder, remove the partial copies in the new one. */
    async leaveModelsMove(dataFolder, { installDir = null } = {}) {
      const result = await call(['models', 'move', '--leave', ...(dataFolder ? ['--from', dataFolder] : []), ...(installDir ? ['--install-dir', installDir] : []), '--output', 'json'], { timeoutMs: 24 * 3600_000 });
      if (result.json?.ok) return result.json;
      throw Object.assign(failure(result, 'The partial copies could not be removed.'), { code: result.json?.error?.code ?? 'leave_failed', details: result.json?.error?.details ?? '' });
    },
    /**
     * The drafter for a model's run config and the drafting choice (`strata drafting plan`): { needed, tuned_needed, gb, drafter_gb, tuned, file, line, note, drafter, split, mtp_needed, mtp_gb }. `model` (with `family`) names a model that has no run config yet. Control decides
     * (the table of strata_control/drafting.py and the all-cards split engine's rule, strata_control/split_engine.py: the same answer as the Terminal's); the app never reads
     * it. Nothing is downloaded. `noSplit` asks as if the split engine will not take the start (only after the draft layer for the split could not be downloaded).
     */
    async draftingPlan({ configPath = null, mode = null, family = null, model = null, dataDir = null, noSplit = false, unloading = false } = {}) {
      const result = await call(['drafting', 'plan', ...(configPath ? ['--config', configPath] : []), ...(mode ? ['--mode', mode] : []), ...(family ? ['--family', family] : []), ...(model && !configPath ? ['--model', model] : []), ...(dataDir ? ['--data-dir', dataDir] : []), ...(noSplit ? ['--no-split'] : []), ...(unloading ? ['--engines-unloading'] : []), '--output', 'json'], { timeoutMs: 30_000 });
      if (result.json?.ok) return result.json;
      throw failure(result, 'The drafter could not be checked.');
    },
    /** Fetch and pack the drafter when the choice would use it and it is not here (`strata drafting fetch --if-needed`); `onEvent` gets {step, done, total}. */
    async draftingFetch({ configPath, mode = null, noSplit = false, unloading = false, throttle = false, onEvent = null, signal = null } = {}) {
      const args = ['drafting', 'fetch', '--config', configPath, ...(mode ? ['--mode', mode] : []), ...(noSplit ? ['--no-split'] : []), ...(unloading ? ['--engines-unloading'] : []), '--if-needed', ...(throttle ? ['--throttle'] : []), '--output', 'jsonl'];
      const result = await call(args, { timeoutMs: 3 * 3600_000, acquire: true, onLine: onEvent ?? (() => {}), signal });
      if (result.json?.ok) return result.json;
      throw Object.assign(new ControlError(result.json?.error?.message ?? (result.stderr.split(/\r?\n/).filter(Boolean).pop() || 'The DFlash2 drafter could not be downloaded.'), { exitCode: result.code, output: result.json }), { code: 'drafting_failed' });
    },
    /**
     * The all-cards split engine for the run config the app is about to start (`strata split-engine apply`, the step `strata launch` does itself): Control checks the rule, the
     * engine package and the draft layer on this computer and writes the split engine's run config to `outPath` when the answer is yes. Resolves { split, line, warn, reasons, cards, written };
     * `configPath` is never changed, nothing is downloaded.
     */
    async splitEngine({ configPath, outPath } = {}) {
      const result = await call(['split-engine', 'apply', '--config', configPath, '--out', outPath, '--output', 'json'], { timeoutMs: 60_000 });
      if (result.json?.ok) return result.json;
      throw failure(result, 'The all-cards split engine could not be checked.');
    },
    /**
     * The picture files of an installed model that has none (strata_control/pictures.py): found or downloaded once (about 0.9 GB, shared), and the run
     * config's `vision` section worked out for this engine. Nothing else of the model is touched and nothing is written to its run config: the caller does.
     * Resolves { mmproj, vision, fetched, note, data_dir }; a failure is a ControlError whose `code` is offline | disk | failed | checksum | unsupported | no_reader.
     * `onEvent` gets {step, done, total}. This is an explicit step that may reach the network (`acquire`), only for the Load the person pressed.
     */
    async picturesFetch({ configPath, engineDir, vision = 'yes', dataDir = null, family = null, hardware = null, onEvent = null, signal = null } = {}) {
      const args = ['fetch', '--config', configPath, '--engine-dir', engineDir, '--vision', vision, ...(dataDir ? ['--data-dir', dataDir] : []), ...(family ? ['--family', family] : []),
        ...(hardware?.vram_gb != null ? ['--vram-gb', String(hardware.vram_gb)] : []), ...(hardware?.make ? ['--make', String(hardware.make)] : []), ...(hardware?.arch != null ? ['--arch', String(hardware.arch)] : [])];
      const result = await call(args, { timeoutMs: 3 * 3600_000, acquire: true, boot: 'pictures', onLine: onEvent ?? (() => {}), signal });
      if (result.json?.ok) return result.json;
      throw Object.assign(new ControlError(result.json?.error?.message ?? (result.stderr.split(/\r?\n/).filter(Boolean).pop() || 'The picture files could not be downloaded.'), { exitCode: result.code, output: result.json }), { code: result.json?.error?.code ?? 'failed' });
    },
    async saveProfile(name, configPath) {
      const listed = await call(['profile', 'list', '--output', 'json'], { timeoutMs: 20_000 });
      const known = (listed.json?.profiles ?? []).some(p => p.id === name);
      const result = await call(known
        ? ['profile', 'use-config', name, '--from-config', configPath, '--output', 'json']
        : ['profile', 'create', name, '--from-config', configPath, '--output', 'json'], { timeoutMs: 30_000 });
      if (result.code !== 0) throw failure(result, 'The model profile could not be saved.');
      return result.json;
    },
    // The app owns the engines it starts: `--die-with-pid` makes the owner stop its engine (drained) when this host is
    // gone, however the host ended (a crash, a kill, the shell's death). A Terminal's `runtime start` has no such tie.
    async start(name, { waitSeconds = 300, dieWithPid = process.pid, drainSeconds = 15 } = {}) {
      const result = await call(['runtime', 'start', '--profile', name, '--wait', String(waitSeconds), '--shutdown-timeout', String(drainSeconds), ...(dieWithPid ? ['--die-with-pid', String(dieWithPid)] : []), '--output', 'json'], { timeoutMs: (waitSeconds + 60) * 1000 });
      if (result.code !== 0 || !result.json?.ok) throw failure(result, 'The engine did not start.');
      return result.json;
    },
    /** Give the user back Strata files in their own folders that a run as administrator locked (`strata repair-access`). Never throws. */
    async repairAccess() {
      try {
        const result = await call(['repair-access', '--output', 'json'], { timeoutMs: 60_000 });
        return result.json ?? { ok: false, error: result.stderr };
      } catch (error) {
        return { ok: false, error: String(error?.message ?? error) };
      }
    },
    /** End servers and engines Strata started whose owner is gone (each is re-checked by PID, start time and command line). */
    async cleanup() {
      const result = await call(['runtime', 'cleanup', '--output', 'json'], { timeoutMs: 120_000 });
      return result.json ?? { ok: false, error: result.stderr };
    },
    async status(name) {
      const result = await call(['runtime', 'status', '--profile', name, '--output', 'json'], { timeoutMs: 30_000 });
      return result.json ?? { ok: false, state: 'unknown', error: result.stderr };
    },
    async stop(name, { waitSeconds = 120 } = {}) {
      const result = await call(['runtime', 'stop', '--profile', name, '--wait', String(waitSeconds), '--output', 'json'], { timeoutMs: (waitSeconds + 60) * 1000 });
      // an instance that is not running is already stopped
      if (result.code !== 0 && !/not running|no owner|stopped|absent|no managed/i.test(JSON.stringify(result.json ?? result.stderr))) throw failure(result, 'The engine did not stop.');
      return result.json;
    },
  };
}
