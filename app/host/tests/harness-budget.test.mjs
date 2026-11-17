import test from 'node:test';
import assert from 'node:assert/strict';
import { linkSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { runInNewContext } from 'node:vm';
import { HARNESS_BUDGET_PATCH, patchStagedHarnessBudget } from '../../desktop/scripts/harness-budget.mjs';

// Exact MIT-licensed pi-ai 1.1.0 module; arithmetic fixture, not native tokenization.
const original = readFileSync(new URL('./fixtures/pi-ai-1.1.0-simple-options.js.fixture', import.meta.url), 'utf8').replaceAll('\r\n', '\n');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
function staged(t, { source = original, manifest = { name: '@earendil-works/pi-ai', version: '1.1.0' } } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'strata-harness-budget-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const file = join(root, HARNESS_BUDGET_PATCH.file), packageFile = join(root, 'node_modules', '@earendil-works', 'pi-ai', 'package.json');
  mkdirSync(join(file, '..'), { recursive: true });
  writeFileSync(file, source); writeFileSync(packageFile, JSON.stringify(manifest));
  return { root, file, packageFile };
}
function patched(t) {
  const fixture = staged(t), provenance = patchStagedHarnessBudget(fixture.root);
  return { ...fixture, provenance, clamp: clampOf(readFileSync(fixture.file, 'utf8')) };
}
function clampOf(source) {
  const code = source.replace(/^import .*;\n/gm, '').replace(/^export /gm, '');
  return runInNewContext(`${code}\nclampMaxTokensToContext;`, {
    estimateContextTokens: context => ({ tokens: context.tokens }),
    clampThinkingLevel: (_model, level) => level,
  });
}

test('staged pi-ai leaves output room at 8K Context', t => {
  assert.equal(clampOf(original)({ contextWindow: 8192 }, { tokens: 6545 }, 4096), 1, 'The Original Regression');
  const { clamp } = patched(t);
  assert.equal(clamp({ contextWindow: 8192 }, { tokens: 6545 }, 4096), 1135);
});

test('small-window safety leaves answer room after a tool result', t => {
  const { clamp } = patched(t);
  assert.equal(clamp({ contextWindow: 8192 }, { tokens: 7750 }, 4096), 221);
  assert.equal(clamp({ contextWindow: 8192 }, { tokens: 8000 }, 4096), 96);
  assert.equal(clamp({ contextWindow: 8192 }, { tokens: 8000 }, 32), 32);
});

test('the staged correction retains caller caps and the no-room floor', t => {
  const { clamp } = patched(t);
  for (const cap of [1, 128]) assert.equal(clamp({ contextWindow: 8192 }, { tokens: 6545 }, cap), cap);
  for (const tokens of [8191, 8192, 9000]) assert.equal(clamp({ contextWindow: 8192 }, { tokens }, 4096), 1);
});

test('32K scales safety to 2048; 64K and larger keep the original calculation', t => {
  const { clamp } = patched(t), old = clampOf(original);
  assert.equal(clamp({ contextWindow: 32768 }, { tokens: 6545 }, 32768), 24175);
  assert.equal(clamp({ contextWindow: 65536 }, { tokens: 6545 }, 65536), 54895);
  assert.equal(clamp({ contextWindow: 131072 }, { tokens: 6545 }, 131072), 120431);
  for (const window of [65536, 131072, 262144]) {
    for (const tokens of [0, 6545, window - 4096, window]) {
      for (const cap of [1, 128, 32768, window])
        assert.equal(clamp({ contextWindow: window }, { tokens }, cap), old({ contextWindow: window }, { tokens }, cap));
    }
  }
});

test('unknown and nonpositive context paths retain their previous behavior', t => {
  const { clamp } = patched(t), old = clampOf(original);
  for (const window of [0, -1, null, undefined, NaN]) {
    for (const cap of [0, 1, 128, 4096])
      assert.equal(clamp({ contextWindow: window }, { tokens: 6545 }, cap), old({ contextWindow: window }, { tokens: 6545 }, cap));
  }
});

test('patch provenance records its identity, exact dependency and before/after hashes', t => {
  const { file, packageFile, provenance } = patched(t);
  assert.equal(sha256(original), HARNESS_BUDGET_PATCH.beforeSha256);
  assert.deepEqual(provenance, { id: 'strata-pi-ai-context-safety-v2', package: '@earendil-works/pi-ai', version: '1.1.0',
    file: HARNESS_BUDGET_PATCH.file, beforeSha256: sha256(original), afterSha256: sha256(readFileSync(file)) });
  assert.notEqual(provenance.beforeSha256, provenance.afterSha256);
  assert.deepEqual(JSON.parse(readFileSync(packageFile, 'utf8')), { name: '@earendil-works/pi-ai', version: '1.1.0' });
  const changed = original.replace('const available = model.contextWindow - estimateContextTokens(context).tokens - CONTEXT_SAFETY_TOKENS;',
    `const remaining = model.contextWindow - estimateContextTokens(context).tokens;
    const safety = model.contextWindow >= 65536 ? CONTEXT_SAFETY_TOKENS
        : Math.min(CONTEXT_SAFETY_TOKENS, Math.floor(model.contextWindow / 16), Math.max(0, Math.floor(remaining / 2)));
    const available = remaining - safety;`);
  assert.equal(readFileSync(file, 'utf8'), changed, 'Only The Known Clamp Changes');
});

test('malformed or unexpected dependencies fail before writing', t => {
  const cases = [
    { manifest: { name: '@earendil-works/pi-ai', version: '1.2.0' } },
    { manifest: { name: 'another-package', version: '1.1.0' } },
    { manifest: null },
    { source: original + '\n' },
    { source: original.replace('CONTEXT_SAFETY_TOKENS = 4096', 'CONTEXT_SAFETY_TOKENS = 1024') },
  ];
  for (const value of cases) {
    const f = staged(t, value), before = readFileSync(f.file), manifest = readFileSync(f.packageFile);
    assert.throws(() => patchStagedHarnessBudget(f.root), /Harness output-budget patch:/);
    assert.deepEqual(readFileSync(f.file), before); assert.deepEqual(readFileSync(f.packageFile), manifest);
  }
  const f = staged(t); writeFileSync(f.packageFile, '{bad json');
  assert.throws(() => patchStagedHarnessBudget(f.root), /cannot read a standalone staged pi-ai dependency/);
  assert.equal(readFileSync(f.file, 'utf8'), original); assert.equal(readFileSync(f.packageFile, 'utf8'), '{bad json');
});

test('a repeated patch rejects already-changed bytes without writing again', t => {
  const f = patched(t), before = readFileSync(f.file);
  assert.throws(() => patchStagedHarnessBudget(f.root), /unexpected simple-options.js SHA-256/);
  assert.deepEqual(readFileSync(f.file), before);
});

test('staging refuses a module hardlinked to another copy', t => {
  const f = staged(t), other = join(f.root, 'source-module.js'); linkSync(f.file, other);
  assert.throws(() => patchStagedHarnessBudget(f.root), /expected an unlinked staged module/);
  assert.equal(readFileSync(other, 'utf8'), original); assert.equal(readFileSync(f.file, 'utf8'), original);
});

test('patching a disposable copy leaves the installed original hash unchanged', {
  skip: !process.env.STRATA_TEST_PI_AI_DIR && 'STRATA_TEST_PI_AI_DIR is not set (the Harness AI package folder)',
}, t => {
  const installed = join(process.env.STRATA_TEST_PI_AI_DIR, 'dist', 'api', 'simple-options.js');
  const before = sha256(readFileSync(installed));
  assert.equal(before, HARNESS_BUDGET_PATCH.beforeSha256);
  const f = staged(t, { source: readFileSync(installed) }); patchStagedHarnessBudget(f.root);
  assert.equal(sha256(readFileSync(installed)), before);
});
