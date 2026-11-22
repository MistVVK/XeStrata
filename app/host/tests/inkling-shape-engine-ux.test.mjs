/**
 * Unit tests for packaged Desktop GUI behavior when only a shape engine
 * (`strata-inkling` or `strata-hy4`) is installed without the main Qwen `strata-engine`:
 *  1. `engineNeededFromStatus()` suppresses the 490 MB Qwen `Welcome to Strata` first-run modal.
 *  2. `engineMissing(native, family)` hides the yellow `"didn't recognise an engine"` banner for
 *     the installed shape engine's family while keeping it visible for Qwen/Swift/Coder families.
 *  3. Host routes (`/api/control/status`, `/api/native/local/preview`) and `native-control.mjs`
 *     forward shape engines and allow `.gguf` / `.strata` local Inkling-2-Mini imports without
 *     triggering a Qwen engine download.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundle = async file => {
  const uiStub = {
    name: 'ui-stub',
    setup(b) {
      b.onResolve({ filter: /^react(-dom)?(\/.*)?$|^\.\.?\/(Brand|ui|FirstRunLanguage|lib\/(i18n|client|download-eta|first-run-card|connect-only|store|router|format))$/ }, args => ({ path: args.path, namespace: 'stub' }));
      b.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({
        contents: 'export const tr = s => s; export const useState = () => [null, () => {}]; export const useEffect = () => {}; export const useRef = () => ({ current: null }); export const useCallback = fn => fn; export const useMemo = fn => fn(); export const Brand = () => null; export const Bar = () => null; export const Btn = () => null; export const Icon = () => null; export const FirstRunLanguage = () => null; export const api = { get: async () => ({}) }; export const useDownloadEta = () => ""; export const megabytes = x => x; export const firstRunCard = () => null; export const connectOnly = async () => {}; export const completeSetupStep = async () => {}; export const openSetup = () => {}; export const setSettingsSection = () => {}; export const navigate = () => {}; export const jsx = () => null; export const jsxs = () => null; export const Fragment = Symbol("F"); export default {};',
        loader: 'js',
      }));
    },
  };
  const built = await build({
    entryPoints: [fileURLToPath(new URL(file, import.meta.url))],
    bundle: true,
    write: false,
    platform: 'node',
    format: 'esm',
    jsx: 'automatic',
    plugins: [uiStub],
    loader: { '.json': 'json' },
  });
  return import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
};

const firstRun = await bundle('../src/components/EngineFirstRun.tsx');
const warning = await bundle('../src/components/EngineWarning.tsx');

test('engineNeededFromStatus suppresses first-run modal when a shape engine is usable', () => {
  // No main engine and no shape engines -> needs first-run download card
  assert.equal(firstRun.engineNeededFromStatus({
    engine: { usable: false, notice: 'No engine' },
    shapeEngines: [],
    anyUsableEngine: false,
    engineSetup: false,
  }), true);

  // Still unpacking bundled engine -> null (unknown)
  assert.equal(firstRun.engineNeededFromStatus({
    engine: { usable: false },
    engineSetup: true,
  }), null);

  // Only strata-inkling installed -> DO NOT show 490 MB Qwen modal
  assert.equal(firstRun.engineNeededFromStatus({
    engine: { usable: false, notice: 'No Qwen engine', shapes: [{ shape: 'inkling2', usable: true }] },
    shapeEngines: [{ shape: 'inkling2', component: 'strata-inkling', usable: true }],
    anyUsableEngine: true,
    engineSetup: false,
  }), false);

  // Only strata-hy4 installed -> DO NOT show 490 MB Qwen modal
  assert.equal(firstRun.engineNeededFromStatus({
    engine: { usable: false, shapes: [{ shape: 'hy4', usable: true }] },
    shapeEngines: [{ shape: 'hy4', component: 'strata-hy4', usable: true }],
    anyUsableEngine: true,
    engineSetup: false,
  }), false);

  // Shape engine present in engine.shapes even if older host lacks top-level anyUsableEngine
  assert.equal(firstRun.engineNeededFromStatus({
    engine: { usable: false, shapes: [{ shape: 'inkling2', usable: true }] },
    engineSetup: false,
  }), false);

  // Unusable shape engine still prompts for an engine
  assert.equal(firstRun.engineNeededFromStatus({
    engine: { usable: false, shapes: [{ shape: 'inkling2', usable: false }] },
    shapeEngines: [{ shape: 'inkling2', usable: false }],
    anyUsableEngine: false,
    engineSetup: false,
  }), true);
});

test('engineMissing hides yellow banner for installed shape engine family and keeps it for Qwen', () => {
  const inklingOnly = {
    engine: {
      usable: false,
      notice: 'No Qwen engine installed',
      shapes: [{ shape: 'inkling2', component: 'strata-inkling', usable: true, version: '0.1.65', backend: 'cuda' }],
    },
    installed: { models: [] },
  };
  assert.equal(warning.shapeEngineUsable(inklingOnly, 'inkling-2-mini'), true);
  assert.equal(warning.engineMissing(inklingOnly, 'inkling-2-mini'), false);
  // Selecting Qwen3.8 or Qwen3.6 or Hy4-Flash (without hy4 engine) MUST still show the warning
  assert.equal(warning.engineMissing(inklingOnly, 'qwen3.8'), true);
  assert.equal(warning.engineMissing(inklingOnly, 'qwen3.6'), true);
  assert.equal(warning.engineMissing(inklingOnly, 'hy4-flash'), true);
  assert.equal(warning.engineMissing(inklingOnly, undefined), true);

  const hy4Only = {
    engine: {
      usable: false,
      shapes: [{ shape: 'hy4', component: 'strata-hy4', usable: true, version: '0.1.65', backend: 'cuda' }],
    },
    installed: { models: [] },
  };
  assert.equal(warning.engineMissing(hy4Only, 'hy4-flash'), false);
  assert.equal(warning.engineMissing(hy4Only, 'inkling-2-mini'), true);
  assert.equal(warning.engineMissing(hy4Only, 'qwen3.8'), true);

  // Ready installed Inkling model also suppresses the Inkling warning
  const readyInkling = {
    engine: { usable: false, shapes: [] },
    installed: { models: [{ model: 'inkling-2-mini-sq-10', ready: true }] },
  };
  assert.equal(warning.engineMissing(readyInkling, 'inkling-2-mini'), false);
  assert.equal(warning.engineMissing(readyInkling, 'qwen3.8'), true);

  // Main engine usable -> never missing for any family
  const mainUsable = { engine: { usable: true, shapes: [] }, installed: { models: [] } };
  assert.equal(warning.engineMissing(mainUsable, 'qwen3.8'), false);
  assert.equal(warning.engineMissing(mainUsable, 'inkling-2-mini'), false);
});

test('host server wires shapeEngines, allows .strata local preview, and skips Qwen engineStep for Inkling', () => {
  const indexSrc = readFileSync(new URL('../server/index.mjs', import.meta.url), 'utf8');
  assert.match(indexSrc, /fileType:\/\\\.\(gguf\|strata\)\$\/i/, '/api/native/local/preview accepts .gguf and .strata');
  assert.match(indexSrc, /anyUsableEngine/, '/api/control/status exposes anyUsableEngine');
  assert.match(indexSrc, /shapeEngines/, '/api/control/status exposes shapeEngines');

  const nativeSrc = readFileSync(new URL('../server/native-control.mjs', import.meta.url), 'utf8');
  assert.match(nativeSrc, /!isHy4Family\(args\.family\)&&!isInklingFamily\(args\.family\)/, 'Inkling install skips Qwen engineStep and keeps default vision');
  assert.match(nativeSrc, /shapes:Array\.isArray\(engine\.shapes\)/, 'strata_status forwards shape engines to UI');
});
