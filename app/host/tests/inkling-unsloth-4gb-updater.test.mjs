/**
 * Track 7A GUI & Host tests:
 * - Inkling-2-Mini Unsloth GGUF catalog grouping & labeling (native SQ-20/15/10 + direct-download Unsloth UD-* GGUFs)
 * - 4 GB GPU unlock in native-fit.ts (runCards) and gpu-roles.ts (recommend)
 * - Issue #6573 Windows install path mismatch & custom launcher profile warning in updates.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { checkComponentUpdates, formatInstallPathWarning } from '../server/updates.mjs';

const bundle = async file => {
  const built = await build({
    entryPoints: [fileURLToPath(new URL(file, import.meta.url))],
    bundle: true,
    write: false,
    platform: 'node',
    format: 'esm',
    loader: { '.json': 'json' },
  });
  return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
};

const catalog = await bundle('../src/lib/native-catalog.ts');
const fit = await bundle('../src/lib/native-fit.ts');
const gpuRoles = await bundle('../src/lib/gpu-roles.ts');

test('Inkling-2-Mini catalog exposes native .strata builds and Unsloth GGUF builds cleanly', () => {
  const families = [
    {
      family: 'inkling-2-mini',
      title: 'Inkling-2-Mini',
      about: 'Inkling-2-Mini',
      images: true,
      sizes: [
        { id: 'inkling-2-mini-sq-20', model: 'SQ-20', download_gb: 20.0, format: 'strata' },
        { id: 'inkling-2-mini-sq-15', model: 'SQ-15', download_gb: 15.0, format: 'strata' },
        { id: 'inkling-2-mini-sq-10', model: 'SQ-10', download_gb: 11.1, format: 'strata' },
        { id: 'inkling-2-mini-ud-iq1_s', model: 'UD-IQ1_S', download_gb: 8.8, format: 'gguf' },
        { id: 'inkling-2-mini-ud-q2_k_xl', model: 'UD-Q2_K_XL', download_gb: 10.2, format: 'gguf' },
        { id: 'inkling-2-mini-ud-q4_k_xl', model: 'UD-Q4_K_XL', download_gb: 18.7, format: 'gguf' },
      ],
    },
  ];
  const groups = catalog.nativeCatalogModels(families);
  const ink = groups.find(g => g.family === 'inkling-2-mini');
  assert.ok(ink);
  const sq10 = ink.sizes.find(s => s.model === 'SQ-10');
  const udq2 = ink.sizes.find(s => s.model === 'UD-Q2_K_XL');
  assert.equal(sq10.edition, '');
  assert.equal(udq2.edition, 'Unsloth');
  assert.equal(catalog.catalogVersionLabel(sq10), 'Compact');
  assert.equal(catalog.catalogVersionLabel(udq2), 'Unsloth · UD-Q2_K_XL');
});

test('4 GB GPUs reporting 3.8-4.0 GiB are admitted by runCards and gpu-roles', () => {
  const gtx1650 = { index: 0, name: 'NVIDIA GeForce GTX 1650', vram_gb: 3.9, usable: true, vendor: 'nvidia' };
  const igpu = { index: 1, name: 'Intel UHD Graphics', vram_gb: 0, integrated: true, usable: true, vendor: 'nvidia' };
  const cards = fit.runCards([gtx1650, igpu], 'nvidia');
  assert.equal(cards.length, 1);
  assert.equal(cards[0].name, 'NVIDIA GeForce GTX 1650');

  const rec = gpuRoles.recommend([gtx1650], 'nvidia');
  assert.equal(rec.main, 0);
  assert.equal(rec.roles[0], 'main');
});

test('Issue #6573: checkComponentUpdates surfaces Windows install path mismatch and custom profile env warning', async () => {
  const warning = formatInstallPathWarning({
    installDir: 'C:\\Strata\\App',
    registeredInstallDir: 'D:\\Strata',
    installPathMismatch: true,
    customProfileEnv: ['STRATA_DESKTOP_DATA_DIR', 'STRATA_MODELS_DIR'],
  });
  assert.match(warning, /C:\\Strata\\App/);
  assert.match(warning, /D:\\Strata/);
  assert.match(warning, /STRATA_DESKTOP_DATA_DIR, STRATA_MODELS_DIR/);

  assert.equal(formatInstallPathWarning({ installPathMismatch: false, customProfileEnv: [] }), null);

  const report = await checkComponentUpdates({
    control: {
      version: '0.2.17',
      updateCheck: async () => ({
        feed: 'https://updates.example/feed.json',
        engine: { status: 'current', installed: { version: '0.1.65' } },
        control: { status: 'current', installed: '0.2.17' },
        app: {},
      }),
    },
    appUpdater: {
      check: async () => ({
        available: true,
        version: '0.2.18',
        current: '0.2.17',
        notes: 'Bug fixes',
        installDir: 'C:\\Strata\\App',
        registeredInstallDir: 'D:\\Strata',
        installPathMismatch: true,
        customProfileEnv: ['STRATA_DESKTOP_DATA_DIR'],
      }),
    },
    appVersion: '0.2.17',
  });
  assert.equal(report.strata.status, 'available');
  assert.match(report.strata.installPathWarning, /C:\\Strata\\App/);
  assert.match(report.strata.installPathWarning, /D:\\Strata/);
  assert.match(report.strata.installPathWarning, /STRATA_DESKTOP_DATA_DIR/);
});
