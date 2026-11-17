// The 0.2.7 GitHub items done in f-027-gh-app: the Power card's readings (#3628), the finished-work sound (#3586), approval
// notifications and the tray choice (#3477), the pinned port (#3402). Hermetic: scratch folders and fake fetch/spawn only.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { EventEmitter } from 'node:events';
import { build } from 'esbuild';
import { pickTemperatures, readLinuxTemperatures, parseSsdLine, createWindowsSsdReader } from '../server/temperature-telemetry.mjs';
import { addEnergy, energyWh } from '../server/energy-meter.mjs';
import { cleanPrefs, readShellPrefs, writeShellPrefs, parsePortNotice, createShellNotifier, validPort, DEFAULTS } from '../server/shell-prefs.mjs';
import { parseNvidiaReadings } from '../server/device-probes.mjs';

const compiled = await build({ stdin: { contents: "export * from './src/lib/attention'; export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const A = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
for (const l of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he']) await A.loadCatalog(l);  // the non-English catalogs load on first use (src/lib/i18n.ts loadCatalog); load them before the first tr()

test('temperatures: the hottest processor and drive sensor, nothing invented', () => {
  assert.deepEqual(pickTemperatures([{ name: 'coretemp', temps: [51000, 56000] }, { name: 'nvme', temps: [41850] }, { name: 'amdgpu', temps: [70000] }]), { cpuTempC: 56, ssdTempC: 42 });
  assert.deepEqual(pickTemperatures([{ name: 'nvme', temps: [-1, 0, 200000] }]), { cpuTempC: null, ssdTempC: null });
  assert.deepEqual(pickTemperatures([]), { cpuTempC: null, ssdTempC: null });
});

test('Linux hwmon folders are read through the injected file functions', async () => {
  const files = { '/h/hwmon0/name': 'k10temp\n', '/h/hwmon0/temp1_input': '48500\n', '/h/hwmon1/name': 'nvme\n', '/h/hwmon1/temp1_input': '39000\n', '/h/hwmon2/name': 'acpitz\n' };
  const dirs = { '/h': ['hwmon0', 'hwmon1', 'hwmon2'], '/h/hwmon0': ['name', 'temp1_input'], '/h/hwmon1': ['name', 'temp1_input'], '/h/hwmon2': ['name'] };
  const read = async path => { if (!(path in files)) throw new Error('ENOENT'); return files[path]; };
  const list = async path => { if (!(path in dirs)) throw new Error('ENOENT'); return dirs[path]; };
  assert.deepEqual(await readLinuxTemperatures({ base: '/h', read, list }), { cpuTempC: 49, ssdTempC: 39 });
  assert.deepEqual(await readLinuxTemperatures({ base: '/missing', read, list }), { cpuTempC: null, ssdTempC: null });
});

test('the Windows SSD loop gives a value only for a plausible line and goes stale', () => {
  assert.equal(parseSsdLine('ssd 53'), 53);
  assert.equal(parseSsdLine('ssd -'), null);
  assert.equal(parseSsdLine('ssd 400'), null);
  assert.equal(parseSsdLine('something else'), undefined);
  let clock = 1000;
  const child = Object.assign(new EventEmitter(), { stdout: Object.assign(new EventEmitter(), { unref() {} }), unref() {}, kill() {} });
  const reader = createWindowsSsdReader({ spawnFn: () => child, now: () => clock });
  assert.equal(reader.read(), null, 'no line yet');
  child.stdout.emit('data', 'ssd 47\r\n');
  assert.equal(reader.read(), 47);
  clock += 10 * 60 * 1000;
  assert.equal(reader.read(), null, 'an old value is not shown');
});

test('GPU energy adds up the held power over the time between ticks, at most 5 s of a gap', () => {
  const before = energyWh();
  addEnergy(360, 10_000);                       // first tick: nothing to add yet
  assert.equal(energyWh(), before);
  addEnergy(360, 11_000);                       // 360 W for 1 s = 0.1 Wh
  assert.ok(Math.abs(energyWh() - before - 0.1) < 1e-9);
  addEnergy(360, 3_600_000);                    // a sleep of an hour counts as 5 s only
  assert.ok(Math.abs(energyWh() - before - 0.6) < 1e-9);
  addEnergy(null, 3_601_000);                   // no reading: nothing added
  assert.ok(Math.abs(energyWh() - before - 0.6) < 1e-9);
});

test('nvidia-smi fan and clock are read, an unsupported field is no reading', () => {
  const [a] = parseNvidiaReadings('GPU-1, 41, 5, 1024, 30.5, 200, 4, 4, 16, 35, 1800');
  assert.equal(a.fanPct, 35); assert.equal(a.clockMhz, 1800);
  const [b] = parseNvidiaReadings('GPU-1, 41, 5, 1024, 30.5, 200, 4, 4, 16, [N/A], [N/A]');
  assert.equal(b.fanPct, null); assert.equal(b.clockMhz, null);
  const [c] = parseNvidiaReadings('GPU-1, 41, 5, 1024');
  assert.equal(c.fanPct, null);
});

test('the readings strip lists only what this PC can read and the Settings has the price', () => {
  const monitor = readFileSync(new URL('../src/screens/Monitor.tsx', import.meta.url), 'utf8');
  for (const label of ['Energy since start', 'Cost since start', 'GPU fan', 'GPU clock', 'CPU temperature', 'SSD temperature']) assert.ok(monitor.includes(`'${label}'`), label);
  assert.match(monitor, /\.filter\(\(line\): line is/);
  assert.match(monitor, /readingsStrip\.length \? <section className="st-card monitor-readings"/, 'the whole strip is left out when no reading is available');
  assert.doesNotMatch(monitor, /lines=\{/, 'the Power card keeps its old height');
  const settings = readFileSync(new URL('../src/screens/Settings.tsx', import.meta.url), 'utf8');
  assert.match(settings, /label: 'Electricity price'/);
});

test('shell prefs: defaults, a valid port, automatic, a refused port, the tray, the language', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-shell-prefs-'));
  try {
    const file = join(dir, 'sub', 'shell-prefs.json');
    assert.deepEqual(readShellPrefs(file), DEFAULTS);
    assert.deepEqual(writeShellPrefs({ port: 47900 }, file), { port: 47900, tray: false, language: 'en' });
    assert.deepEqual(writeShellPrefs({ tray: true, language: 'de' }, file), { port: 47900, tray: true, language: 'de' });
    assert.throws(() => writeShellPrefs({ port: 80 }, file), /between 1024 and 65535/);
    assert.throws(() => writeShellPrefs({ port: 70000 }, file), /between 1024 and 65535/);
    assert.equal(readShellPrefs(file).port, 47900, 'a refused port changes nothing');
    assert.equal(writeShellPrefs({ port: null }, file).port, 0);
    assert.equal(writeShellPrefs({ language: 'xx' }, file).language, 'de', 'an unknown language is ignored');
    writeFileSync(file, '{ not json');
    assert.deepEqual(readShellPrefs(file), DEFAULTS);
    assert.deepEqual(cleanPrefs({ port: '47831', tray: 'yes', language: 'ko' }), { port: 0, tray: false, language: 'ko' });
    assert.ok(validPort(1024) && validPort(65535) && !validPort(1023) && !validPort(65536) && !validPort(47.5));
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('the busy-port notice and the shell notifier come only from the shell', async () => {
  assert.deepEqual(parsePortNotice('busy:47831:51234'), { wanted: 47831, used: 51234 });
  assert.equal(parsePortNotice('busy:x:1'), null);
  assert.equal(parsePortNotice(undefined), null);
  assert.equal(createShellNotifier({ env: {} }), null, 'a browser or a source checkout has no shell');
  assert.equal(createShellNotifier({ env: { STRATA_SHELL_BRIDGE: 'http://example.com:1', STRATA_SHELL_TOKEN: 't' } }), null, 'only a loopback bridge');
  const seen = [];
  const notifier = createShellNotifier({ env: { STRATA_SHELL_BRIDGE: 'http://127.0.0.1:5555', STRATA_SHELL_TOKEN: 'secret' }, fetcher: async (url, init) => { seen.push([url, init]); return { ok: true, json: async () => ({ shown: true }) }; } });
  assert.deepEqual(await notifier.notify({ title: 'Strata needs your answer', body: 'a & b?' }), { shown: true });
  assert.equal(seen[0][0], 'http://127.0.0.1:5555/shell/notify?title=Strata%20needs%20your%20answer&body=a%20%26%20b%3F');
  assert.equal(seen[0][1].headers.authorization, 'Bearer secret');
  assert.equal(seen[0][1].method, 'POST');
});

test('the finished-work sound is opt-in, quiet for quick answers in front, and never twice in a row', async () => {
  const base = { enabled: true, durationMs: 20_000, focused: true, now: 1_000_000 };
  assert.equal(A.shouldPlayDone({ ...base, enabled: false }), false);
  assert.equal(A.shouldPlayDone({ ...base, durationMs: 1000 }), false, 'a quick answer with the window in front');
  assert.equal(A.shouldPlayDone({ ...base, durationMs: 1000, focused: false }), true, 'the window is in the background');
  assert.equal(A.shouldPlayDone(base), true, 'work that took a while');
  const played = [];
  assert.equal(A.playDoneSound(url => ({ volume: 0, play() { played.push(url); return Promise.resolve(); } })), true);
  assert.deepEqual(played, ['/strata-done-sound.wav']);
  assert.equal(A.shouldPlayDone({ ...base, now: Date.now() + 100 }), false, 'a second chime within two seconds is skipped');
  assert.equal(A.playDoneSound(() => { throw new Error('no audio'); }), false, 'a missing audio device is not an error');
  const wav = readFileSync(new URL('../public/strata-done-sound.wav', import.meta.url));
  assert.equal(wav.toString('latin1', 0, 4), 'RIFF');
  assert.equal(wav.toString('latin1', 8, 12), 'WAVE');
});

test('approval notifications: shell first, the page second, nothing when off or when Strata is in front', async () => {
  const notice = { title: 'T', body: 'B' };
  const seen = [];
  assert.equal(await A.showNotice(notice, { post: async () => ({ shown: true }), page: () => { seen.push('page'); return true; } }), true);
  assert.deepEqual(seen, []);
  assert.equal(await A.showNotice(notice, { post: async () => ({ shown: false }), page: () => { seen.push('page'); return true; } }), true);
  assert.equal(await A.showNotice(notice, { post: async () => { throw new Error('no shell'); }, page: () => { seen.push('page2'); return false; } }), false);
  assert.deepEqual(seen, ['page', 'page2']);
  assert.equal(await A.approvalNeeded(false, 'write_file', false), false);
  assert.equal(await A.approvalNeeded(true, 'write_file', true), false, 'the person is looking at Strata');
  A.setTranslationLanguage('de');
  assert.match(A.approvalNotice('write_file').body, /write_file/);
  assert.notEqual(A.approvalNotice('write_file').title, 'Strata needs your answer');
  A.setTranslationLanguage('en');
  assert.equal(A.approvalNotice('').body, 'The assistant wants to use a tool. Open Strata to approve or deny.');
});

test('every new string exists in all eight languages with the same slots, and does not collide with another catalog', async () => {
  const languages = ['en', 'zh', 'ja', 'de', 'fr', 'es', 'ru', 'pt', 'ko', 'he'];
  const read = async name => JSON.parse(readFileSync(new URL(`../src/locales/${name}.json`, import.meta.url), 'utf8'));
  const withFrench = languages;
  const gh = Object.fromEntries(await Promise.all(withFrench.map(async l => [l, await read(`gh-${l}`)])));
  const slots = text => [...text.matchAll(/\{\d+\}/g)].map(m => m[0]).sort();
  const others = new Set();
  for (const name of ['en', 'tune-en', 'tools-en', 'fix-en']) for (const v of Object.values(await read(name))) others.add(v);
  for (const l of withFrench) {
    assert.deepEqual(Object.keys(gh[l]), Object.keys(gh.en), l);
    for (const [id, source] of Object.entries(gh.en)) {
      assert.ok(gh[l][id].trim(), `${l}:${id} is empty`);
      assert.deepEqual(slots(gh[l][id]), slots(source), `${l}:${id}`);
      if (l !== 'en') assert.notEqual(gh[l][id], source, `${l}:${id} is still English`);
    }
  }
  for (const [id, text] of Object.entries(gh.en)) assert.ok(!others.has(text), `already in another catalog: ${id} ${text}`);
  const busyId = Object.keys(gh.en).find(id => gh.en[id].startsWith('Port {0} is used by another program'));
  for (const l of languages) {
    A.setTranslationLanguage(l);
    assert.equal(A.tr('Energy since start'), gh[l].gh01, l);
    assert.equal(A.tr('Port {0} is used by another program, so Strata started on port {1}. Chats and settings saved on port {0} are not shown until Strata can use it again. Close the other program and restart Strata.', [47831, 51234]),
      gh[l][busyId].replaceAll('{0}', '47831').replaceAll('{1}', '51234'), l);
  }
  A.setTranslationLanguage('en');
});

test('the interface size scales the whole page, up to 200 percent, and a saved value is checked (#3970)', () => {
  const store = readFileSync(new URL('../src/lib/store.ts', import.meta.url), 'utf8');
  assert.match(store, /setProperty\('zoom', state\.uiScale === 1 \? '' : String\(state\.uiScale\)\)/);
  assert.match(store, /setProperty\('--st-ui-scale', '1'\)/, 'the token stays at 1 so nothing is scaled twice');
  assert.match(store, /UI_SCALES = \[0\.9, 1, 1\.1, 1\.25, 1\.5, 1\.75, 2\]/);
  assert.match(readFileSync(new URL('../src/lib/default-sizes.ts', import.meta.url), 'utf8'), /UI_SCALE_CHOICES = \[0\.9, 1, 1\.1, 1\.25, 1\.5, 1\.75, 2\]/);
  assert.match(store, /savedSizes\(parsed, CONTENT_FS_MIN, CONTENT_FS_MAX\)/, 'a saved value is checked');
  const settings = readFileSync(new URL('../src/screens/Settings.tsx', import.meta.url), 'utf8');
  assert.match(settings, /Makes all text, buttons and menus in Strata larger or smaller\./);
});

test('the port, tray, sound and notification rows are in Settings, and only the desktop app shows the port and the tray', () => {
  const settings = readFileSync(new URL('../src/screens/Settings.tsx', import.meta.url), 'utf8');
  assert.match(settings, /label: 'Sound when work finishes'/);
  assert.match(settings, /label: 'Notify me when Strata needs an answer'/);
  assert.match(settings, /\.\.\.\(shell\?\.tray \? \[\{ section: 'General', label: 'Minimise to tray'/);
  assert.match(settings, /\.\.\.\(shell\?\.desktop \? \[\{ section: 'General', label: 'App port'/);
  const store = readFileSync(new URL('../src/lib/store.ts', import.meta.url), 'utf8');
  assert.match(store, /frame\.code === 'waiting-approval'/);
  assert.match(store, /workFinished\(state\.doneSound/);
  assert.match(store, /announcePortNotice/);
});

test('the host route for the shell: a browser has no shell, the port choice is saved and checked, a notification is not shown', { timeout: 40000 }, async t => {
  const { spawn } = await import('node:child_process');
  const { once } = await import('node:events');
  const { createServer } = await import('node:net');
  const { mkdirSync } = await import('node:fs');
  const free = async () => { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; };
  const home = mkdtempSync(join(tmpdir(), 'strata-shell-route-'));
  let child, exited;
  t.after(async () => { if (child && child.exitCode === null) { child.kill(); await exited; } rmSync(home, { recursive: true, force: true, maxRetries: 3 }); });
  mkdirSync(join(home, 'data'));
  const port = await free(), base = `http://127.0.0.1:${port}`;
  const env = { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_WORKSPACE: join(home, 'work'), STRATA_SETTINGS_FILE: join(home, 'settings.json'), STRATA_COMPONENTS_DIR: join(home, 'components'), STRATA_CONTROL_DIR: join(home, 'no-control'), STRATA_HARNESS_DIR: join(home, 'no-harness'), STRATA_NATIVE_REPO: join(home, 'no-repo'), STRATA_PORT: String(port), STRATA_SHARE_PORT: String(await free()), STRATA_TUNNEL_PORT: String(await free()), STRATA_BIND: '127.0.0.1', STRATA_REAL_ENGINE: '0', STRATA_AGENT: 'off', STRATA_NO_ACQUIRE: '1', STRATA_PORT_NOTICE: 'busy:47831:' + port, STRATA_TEST_LOCALE: 'de-DE', LOCALAPPDATA: join(home, 'Local'), APPDATA: join(home, 'Roaming'), XDG_DATA_HOME: join(home, 'xdg-data'), XDG_STATE_HOME: join(home, 'xdg-state'), XDG_CONFIG_HOME: join(home, 'xdg-config') };
  delete env.STRATA_SHELL_BRIDGE; delete env.STRATA_SHELL_TOKEN;
  child = spawn(process.execPath, ['server/index.mjs'], { cwd: join(import.meta.dirname, '..'), windowsHide: true, stdio: 'ignore', env });
  exited = once(child, 'exit');
  let up = false;
  for (let n = 0; n < 150 && !up; n++) { try { up = (await fetch(base + '/api/health')).ok; } catch { /* not yet */ } if (!up) await new Promise(r => setTimeout(r, 100)); }
  assert.ok(up, 'the fixture host started');
  const get = async path => (await fetch(base + path)).json();
  const post = (path, body) => fetch(base + path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const first = await get('/api/shell');
  assert.equal(first.desktop, false); assert.equal(first.tray, false); assert.equal(first.port, port); assert.equal(first.defaultPort, 47831);
  assert.equal(first.pinnedPort, null); assert.equal(first.minimizeToTray, false);
  assert.deepEqual(first.portNotice, { wanted: 47831, used: port });
  assert.deepEqual(await get('/api/locale'), { locale: 'de-DE', region: 'DE', currency: null, source: 'test' }, 'the host names the regional format; the page maps it to a currency');
  const saved = await post('/api/shell/prefs', { port: 47900, tray: true, language: 'de' });
  assert.equal(saved.status, 200);
  const after = await saved.json();
  assert.equal(after.pinnedPort, 47900); assert.equal(after.minimizeToTray, true);
  assert.deepEqual(JSON.parse(readFileSync(join(home, 'shell-prefs.json'), 'utf8')), { port: 47900, tray: true, language: 'de' });
  const refused = await post('/api/shell/prefs', { port: 80 });
  assert.equal(refused.status, 400);
  assert.match((await refused.json()).error, /between 1024 and 65535/);
  assert.equal((await get('/api/shell')).pinnedPort, 47900, 'a refused port changes nothing');
  assert.equal((await (await post('/api/shell/prefs', { port: null })).json()).pinnedPort, null);
  assert.deepEqual(await (await post('/api/shell/notify', { title: 'T', body: 'B' })).json(), { shown: false });
});
