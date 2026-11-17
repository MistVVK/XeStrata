/** The engine inside the installer: installed when missing or older, never over a newer or equal one, never without a matching manifest. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, readdirSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { installBundledEngine, removeVerifiedBundle, cleanupAfterInstall, compareVersions, wrongMakeNotice } from '../server/bundled-engine.mjs';

// the make of card is read from this computer unless a test names it; the existing tests are written for an NVIDIA PC
process.env.STRATA_ENGINE_BACKEND = 'cuda';
const sha = buffer => createHash('sha256').update(buffer).digest('hex');
function makePackage(root, version, { body = 'engine-' + version, backend = 'cuda', platform = 'windows-x86_64', variant = 'cuda13', minDriver = null } = {}) {
  const name = `${version}-${platform}-${backend}-${variant}`;
  const dir = join(root, name);
  mkdirSync(dir, { recursive: true });
  const files = { 'strata.exe': Buffer.from(body), 'strata_kq.dll': Buffer.from('kq-' + body), 'BUILD.json': Buffer.from('{}') };
  for (const [file, data] of Object.entries(files)) writeFileSync(join(dir, file), data);
  writeFileSync(join(dir, 'engine-component.json'), JSON.stringify({
    schema_version: 2, id: 'strata-engine', name: 'strata-engine', version, platform, backend, archs: ['sm_86'], ...(minDriver ? { min_driver: { nvidia: minDriver } } : {}), build_id: 'x', entrypoint: 'strata.exe',
    files: Object.entries(files).map(([path, data]) => ({ path, sha256: sha(data), bytes: data.length })), signature: null }));
  return name;
}
const setup = () => {
  const base = mkdtempSync(join(tmpdir(), 'st-bundled-engine-'));
  const bundle = join(base, 'engine-bundle'), components = join(base, 'components');
  mkdirSync(bundle); mkdirSync(components);
  return { base, bundle, components, engine: join(components, 'engine'), cleanup: () => rmSync(base, { recursive: true, force: true }) };
};

test('versions compare by number, not text', () => {
  assert.equal(compareVersions('0.1.45', '0.1.9'), 1);
  assert.equal(compareVersions('0.1.45', '0.1.45.1'), -1);
  assert.equal(compareVersions('0.1.45', '0.1.45'), 0);
});

test('no engine installed: the bundled one is installed, hash-checked, with nothing left over', async () => {
  const t = setup();
  try {
    const name = makePackage(t.bundle, '0.1.45');
    const result = await installBundledEngine({ bundleDir: t.bundle, components: t.components });
    assert.equal(result.action, 'installed');
    assert.equal(readFileSync(join(t.engine, name, 'strata.exe'), 'utf8'), 'engine-0.1.45');
    assert.ok(existsSync(join(t.engine, name, 'engine-component.json')));
    assert.deepEqual(readdirSync(t.engine), [name], 'no temporary folder remains');
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components })).action, 'kept', 'a second start does nothing');
  } finally { t.cleanup(); }
});

test('an older engine is installed: the bundled one is added beside it', async () => {
  const t = setup();
  try {
    const old = makePackage(t.engine, '0.1.44');
    const name = makePackage(t.bundle, '0.1.45');
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components })).action, 'installed');
    assert.deepEqual(readdirSync(t.engine).sort(), [old, name].sort());
  } finally { t.cleanup(); }
});

test('a newer or equal engine is never overwritten or touched', async () => {
  const t = setup();
  try {
    const newer = makePackage(t.engine, '0.1.46', { body: 'mine' });
    makePackage(t.bundle, '0.1.45');
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components })).action, 'kept');
    assert.deepEqual(readdirSync(t.engine), [newer]);
    assert.equal(readFileSync(join(t.engine, newer, 'strata.exe'), 'utf8'), 'mine');
    // same version, other build: the installed one stays
    const same = makePackage(t.engine, '0.1.45', { body: 'installed-build' });
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components })).action, 'kept');
    assert.equal(readFileSync(join(t.engine, same, 'strata.exe'), 'utf8'), 'installed-build');
  } finally { t.cleanup(); }
});

test('a damaged copy of the same package is put right; an engine for another backend does not count', async () => {
  const t = setup();
  try {
    const name = makePackage(t.bundle, '0.1.45');
    makePackage(t.engine, '0.1.45');
    writeFileSync(join(t.engine, name, 'strata.exe'), 'cut');          // wrong size
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components })).action, 'repaired');
    assert.equal(readFileSync(join(t.engine, name, 'strata.exe'), 'utf8'), 'engine-0.1.45');
    const t2 = setup();
    try {
      makePackage(t2.engine, '0.1.50', { backend: 'cpu' });
      makePackage(t2.bundle, '0.1.45');
      assert.equal((await installBundledEngine({ bundleDir: t2.bundle, components: t2.components })).action, 'installed');
    } finally { t2.cleanup(); }
  } finally { t.cleanup(); }
});

test('a bundle that does not match its manifest is refused and installs nothing', async () => {
  const t = setup();
  try {
    const name = makePackage(t.bundle, '0.1.45');
    writeFileSync(join(t.bundle, name, 'strata_kq.dll'), 'kq-engine-0.1.4X');   // same size, other bytes
    const result = await installBundledEngine({ bundleDir: t.bundle, components: t.components });
    assert.equal(result.action, 'refused');
    assert.equal(existsSync(join(t.engine, name)), false);
    assert.equal((await installBundledEngine({ bundleDir: join(t.base, 'nope'), components: t.components })).action, 'none');
  } finally { t.cleanup(); }
});

test('CUDA 13 and CUDA 12 builds of one version are different engines: both are installed when the driver runs both', async () => {
  const t = setup();
  try {
    const n13 = makePackage(t.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda13', minDriver: '580.65', body: 'thirteen' });
    const n12 = makePackage(t.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda12', minDriver: '575.51', body: 'twelve' });
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components, driver: '580.178.04' })).action, 'installed');
    assert.deepEqual(readdirSync(t.engine).sort(), [n12, n13].sort());
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components, driver: '580.178.04' })).action, 'kept', 'a second start does nothing');
  } finally { t.cleanup(); }
});

test('an older driver gets only the build it can run; a driver below both gets the lowest build so the app can name the driver', async () => {
  const t = setup(), u = setup(), v = setup();
  try {
    const n13 = makePackage(t.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda13', minDriver: '580.65' });
    const n12 = makePackage(t.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda12', minDriver: '575.51' });
    await installBundledEngine({ bundleDir: t.bundle, components: t.components, driver: '575.57.08' });
    assert.deepEqual(readdirSync(t.engine), [n12], 'driver 575 runs CUDA 12 only');
    for (const x of [u]) { makePackage(x.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda13', minDriver: '580.65' }); makePackage(x.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda12', minDriver: '575.51' }); }
    await installBundledEngine({ bundleDir: u.bundle, components: u.components, driver: '550.120' });
    assert.deepEqual(readdirSync(u.engine), [n12], 'driver 550 runs neither: the CUDA 12 build is installed so Control can say 575 is needed');
    // no driver at all (a PC without NVIDIA, or nvidia-smi cannot be read): nothing is held back
    makePackage(v.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda13', minDriver: '580.65' }); makePackage(v.bundle, '0.1.45', { platform: 'linux-x86_64', variant: 'cuda12', minDriver: '575.51' });
    await installBundledEngine({ bundleDir: v.bundle, components: v.components, driver: null });
    assert.deepEqual(readdirSync(v.engine).sort(), [n12, n13].sort());
  } finally { t.cleanup(); u.cleanup(); v.cleanup(); }
});

// The app update (latest.json) is built without engine-bundle (tauri.update.conf.json), so one update fits every installer variant.
test('an engine-less app update (no engine-bundle folder, or one holding only .gitignore) leaves the installed engines exactly as they are', async () => {
  const t = setup();
  try {
    const amd = makePackage(t.engine, '0.1.45.3', { platform: 'windows-x86_64', backend: 'hip', variant: 'rx9070' });
    const before = readFileSync(join(t.engine, amd, 'engine-component.json'), 'utf8');
    assert.equal((await installBundledEngine({ bundleDir: join(t.base, 'no-such-engine-bundle'), components: t.components })).action, 'none');
    writeFileSync(join(t.bundle, '.gitignore'), '*\n!.gitignore\n');
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components, platform: 'windows-x86_64' })).action, 'none');
    assert.deepEqual(readdirSync(t.engine), [amd]);
    assert.equal(readFileSync(join(t.engine, amd, 'engine-component.json'), 'utf8'), before);
  } finally { t.cleanup(); }
});

test('why the update carries no engine: a CUDA installer on an AMD PC would copy its CUDA engine beside the AMD one (no NVIDIA driver to filter by)', async () => {
  const t = setup();
  try {
    const amd = makePackage(t.engine, '0.1.45.3', { platform: 'windows-x86_64', backend: 'hip', variant: 'rx9070' });
    const cuda = makePackage(t.bundle, '0.1.45.3', { platform: 'windows-x86_64', variant: 'cuda13', minDriver: '580.65' });
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components, driver: null })).action, 'installed');
    assert.deepEqual(readdirSync(t.engine).sort(), [amd, cuda].sort(), 'the AMD engine stays; a useless CUDA engine is added');
  } finally { t.cleanup(); }
});

test('an AMD build on an NVIDIA PC is not unpacked and the sentence names the right download', async () => {
  const t = setup();
  try {
    const name = makePackage(t.bundle, '0.2.0', { platform: 'linux-x86_64', backend: 'hip', variant: 'rx9070' });
    const manifest = JSON.parse(readFileSync(join(t.bundle, name, 'engine-component.json'), 'utf8'));
    manifest.archs = ['gfx1201'];
    writeFileSync(join(t.bundle, name, 'engine-component.json'), JSON.stringify(manifest));
    const result = await installBundledEngine({ bundleDir: t.bundle, components: t.components, backend: 'cuda' });
    assert.equal(result.action, 'wrong-make');
    assert.equal(result.notice, 'This Strata is for AMD RX 9000 cards. Your PC has an NVIDIA card: download the NVIDIA version from https://github.com/Niko1221/Strata/releases');
    assert.equal(existsSync(t.engine), false, 'nothing was copied');
    // the same build on an AMD PC is unpacked; an unknown card is not a reason to refuse
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components, backend: 'hip' })).action, 'installed');
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components, backend: null })).action, 'kept');
  } finally { t.cleanup(); }
});

test('wrongMakeNotice names the make, not a list of reasons', () => {
  assert.equal(wrongMakeNotice({ backend: 'cuda', archs: ['sm_86'] }, 'hip'), 'This Strata is for NVIDIA cards. Your PC has an AMD card: download the AMD version from https://github.com/Niko1221/Strata/releases');
});

test('Windows: the display adapters decide the make, and a leftover nvidia-smi does not outvote them (#3688)', async () => {
  const { gpuBackendHere, windowsDisplayVendors } = await import('../server/bundled-engine.mjs');
  assert.equal(gpuBackendHere({}, { vendors: new Set(['1002']) }), 'hip');
  assert.equal(gpuBackendHere({}, { vendors: new Set(['10de', '1002']) }), 'cuda');
  assert.equal(gpuBackendHere({}, { vendors: new Set(['8086']) }), 'sycl');
  assert.equal(gpuBackendHere({ STRATA_ENGINE_BACKEND: 'cuda' }, { vendors: new Set(['1002']) }), 'cuda', 'an explicit choice wins');
  const out = 'MatchingDeviceId    REG_SZ    PCI/VEN_1002&DEV_7551&SUBSYS_0' + String.fromCharCode(10) + 'MatchingDeviceId    REG_SZ    PCI/VEN_8086&DEV_46A6';
  assert.deepEqual([...windowsDisplayVendors({ platform: 'win32', run: () => out })].sort(), ['1002', '8086']);
  assert.equal(windowsDisplayVendors({ platform: 'linux', run: () => { throw new Error('must not run'); } }).size, 0);
  assert.equal(windowsDisplayVendors({ platform: 'win32', run: () => { throw new Error('no registry'); } }).size, 0);
});

test('after an update, the engine bundle of the old install folder is removed only when the same engine is complete in the store', async () => {
  const t = setup();
  try {
    const name = makePackage(t.bundle, '0.1.48');
    writeFileSync(join(t.bundle, '.gitignore'), 'ignore-all');
    const lines = [];
    // 1. nothing in the store yet: kept, and one line says why
    let r = await removeVerifiedBundle({ bundleDir: t.bundle, components: t.components, log: l => lines.push(l) });
    assert.equal(r.removed, false);
    assert.equal(lines.length, 1);
    assert.match(lines[0], /kept in the install folder.*not \(or not completely\) in the engine store/);
    assert.ok(existsSync(join(t.bundle, name)));
    // 2. the engine is copied (the app's own first start does this)
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components })).action, 'installed');
    // 3. a store file with the right size and the wrong content: kept (the hash is checked, not only the size)
    const target = join(t.engine, name, 'strata.exe');
    const good = readFileSync(target);
    writeFileSync(target, Buffer.alloc(good.length, 'x'));
    lines.length = 0;
    r = await removeVerifiedBundle({ bundleDir: t.bundle, components: t.components, log: l => lines.push(l) });
    assert.equal(r.removed, false);
    assert.equal(lines.length, 1, 'exactly one line');
    assert.ok(existsSync(join(t.bundle, name)));
    // 4. a store file with the wrong size: kept
    writeFileSync(target, Buffer.concat([good, Buffer.from('extra')]));
    assert.equal((await removeVerifiedBundle({ bundleDir: t.bundle, components: t.components })).removed, false);
    // 5. the store copy is whole: the bundle goes, its .gitignore stays, and the store is untouched
    writeFileSync(target, good);
    lines.length = 0;
    r = await removeVerifiedBundle({ bundleDir: t.bundle, components: t.components, log: l => lines.push(l) });
    assert.equal(r.removed, true);
    assert.equal(existsSync(join(t.bundle, name)), false);
    assert.ok(existsSync(join(t.bundle, '.gitignore')));
    assert.equal(readFileSync(target, 'utf8'), 'engine-0.1.48');
    assert.match(lines.at(-1), /was removed; the same engine is complete in the engine store/);
    // 6. nothing left to do
    assert.equal((await removeVerifiedBundle({ bundleDir: t.bundle, components: t.components })).removed, false);
    // 7. a different version in the store does not count
    const other = makePackage(t.bundle, '0.1.49');
    assert.equal((await removeVerifiedBundle({ bundleDir: t.bundle, components: t.components })).removed, false);
    assert.ok(existsSync(join(t.bundle, other)));
  } finally { t.cleanup(); }
});

test('the bundle cleanup lines reach the output of the host (the desktop log copies it), for "removed" and for "kept"', async () => {
  const t = setup();
  const seen = [], real = console.log;
  try {
    const name = makePackage(t.bundle, '0.1.48');
    console.log = line => { seen.push(String(line)); };
    // kept: the engine is not in the store yet
    await cleanupAfterInstall(Promise.resolve({ action: 'kept' }), { bundleDir: t.bundle, components: t.components });
    assert.equal(seen.length, 1);
    assert.match(seen[0], /^engine bundle: kept in the install folder/);
    // removed: the engine is in the store and whole
    const installed = await installBundledEngine({ bundleDir: t.bundle, components: t.components });
    assert.equal(installed.action, 'installed');
    seen.length = 0;
    await cleanupAfterInstall(Promise.resolve({ action: 'kept' }), { bundleDir: t.bundle, components: t.components });
    assert.equal(seen.length, 1);
    assert.match(seen[0], /^engine bundle: the copy that came with the installer \(.*\) was removed/);
    assert.equal(existsSync(join(t.bundle, name)), false);
    // nothing to say when the engine step did nothing (no bundle) or was refused
    seen.length = 0;
    await cleanupAfterInstall(Promise.resolve({ action: 'none' }), { bundleDir: t.bundle, components: t.components });
    await cleanupAfterInstall(Promise.resolve({ action: 'wrong-make' }), { bundleDir: t.bundle, components: t.components });
    await cleanupAfterInstall(Promise.reject(new Error('boom')), { bundleDir: t.bundle, components: t.components });
    assert.deepEqual(seen, []);
  } finally { console.log = real; t.cleanup(); }
});

test('macOS: the engine bundle inside a signed .app stays (deleting it breaks the signature); outside an .app, or on another system, it is removed as before', async () => {
  const t = setup();
  const platform = Object.getOwnPropertyDescriptor(process, 'platform');
  const as = value => Object.defineProperty(process, 'platform', { value, configurable: true });
  try {
    const app = join(t.base, 'Strata.app', 'Contents', 'Resources', 'engine-bundle');
    mkdirSync(app, { recursive: true });
    const inApp = makePackage(app, '0.1.48');
    const outside = makePackage(t.bundle, '0.1.48');
    // the engine is in the store (the app's own first start copies it), so the bundle would be removable
    assert.equal((await installBundledEngine({ bundleDir: t.bundle, components: t.components })).action, 'installed');
    const lines = [];
    // 1. macOS, bundle inside Strata.app: kept, with one line that says why
    as('darwin');
    let r = await removeVerifiedBundle({ bundleDir: app, components: t.components, log: l => lines.push(l) });
    assert.equal(r.removed, false);
    assert.ok(existsSync(join(app, inApp, 'strata.exe')), 'the signed app keeps its engine files');
    assert.equal(lines.length, 1);
    assert.match(lines[0], /kept in the install folder.*macOS app bundles must remain unchanged after signing/);
    // 2. macOS, bundle outside an .app: removed as before
    r = await removeVerifiedBundle({ bundleDir: t.bundle, components: t.components, log: () => {} });
    assert.equal(r.removed, true);
    assert.equal(existsSync(join(t.bundle, outside)), false);
    // 3. another system, bundle inside a folder called .app: removed as before (the guard is for macOS only)
    as('linux');
    r = await removeVerifiedBundle({ bundleDir: app, components: t.components, log: () => {} });
    assert.equal(r.removed, true);
    assert.equal(existsSync(join(app, inApp)), false);
  } finally {
    Object.defineProperty(process, 'platform', platform);
    t.cleanup();
  }
});

test('a card taken out of the PC does not decide the make; two present makes accept a build for either (#3892)', async () => {
  const { gpuBackendsHere, windowsDisplayVendors } = await import('../server/bundled-engine.mjs');
  const nl = String.fromCharCode(10);
  const registry = 'MatchingDeviceId    REG_SZ    PCI/VEN_10DE&DEV_2882' + nl + 'MatchingDeviceId    REG_SZ    PCI/VEN_1002&DEV_7550';
  // the NVIDIA card was swapped for a Radeon: Windows still lists its driver entry, but only the AMD card is present
  assert.deepEqual([...windowsDisplayVendors({ platform: 'win32', run: () => registry, present: () => new Set(['1002']) })], ['1002']);
  assert.equal(gpuBackendsHere({}, { vendors: new Set(['1002']) })[0], 'hip');
  // both cards present: both makes count, NVIDIA first
  assert.deepEqual([...windowsDisplayVendors({ platform: 'win32', run: () => registry, present: () => new Set(['1002', '10de']) })].sort(), ['1002', '10de']);
  assert.deepEqual(gpuBackendsHere({}, { vendors: new Set(['10de', '1002']) }), ['cuda', 'hip']);
  // present cannot be read: the registry answer stands
  assert.equal(windowsDisplayVendors({ platform: 'win32', run: () => registry, present: () => new Set() }).size, 2);
  // the second source is only asked when two makes are listed
  assert.equal(windowsDisplayVendors({ platform: 'win32', run: () => 'MatchingDeviceId    REG_SZ    PCI/VEN_1002&DEV_7550', present: () => { throw new Error('must not run'); } }).size, 1);
  assert.deepEqual(gpuBackendsHere({ STRATA_ENGINE_BACKEND: 'hip' }, { vendors: new Set(['10de', '1002']) }), ['hip']);

  const t = setup();
  try {
    const name = makePackage(t.bundle, '0.2.0', { platform: 'windows-x86_64', backend: 'hip', variant: 'rx9070' });
    const manifest = JSON.parse(readFileSync(join(t.bundle, name, 'engine-component.json'), 'utf8'));
    manifest.archs = ['gfx1201'];
    writeFileSync(join(t.bundle, name, 'engine-component.json'), JSON.stringify(manifest));
    // an AMD build on a PC with an NVIDIA desktop card and an AMD model card is unpacked, not refused
    const result = await installBundledEngine({ bundleDir: t.bundle, components: t.components, backend: ['cuda', 'hip'] });
    assert.equal(result.action, 'installed');
  } finally { t.cleanup(); }
});
