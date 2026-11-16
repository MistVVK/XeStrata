import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { ggufFixture } from './helpers/local-gguf.mjs';
import { inspectLocalModel } from '../server/local-models.mjs';
import { CONTEXT_TOKENS } from '../server/catalog.mjs';
import { modelReservation } from '../server/runtime-models.mjs';

test('a 1K or 2K GGUF survives inspection and reservation without increasing its context', t => {
  const root = mkdtempSync(join(tmpdir(), 'strata-small-context-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const hardware = { memory: { totalGb: 32 }, gpus: [] };
  for (const context of [1024, 2048]) {
    const file = join(root, `model-${context}.gguf`);
    ggufFixture(file, { context });
    const { variant } = inspectLocalModel(file);
    const label = `${context / 1024}k`;
    assert.equal(variant.contextMax, context);
    assert.equal(CONTEXT_TOKENS[label], context);
    const config = { beginner: { contextLength: label }, expert: [] };
    assert.equal(modelReservation(variant, config, hardware, []).context, context);
    assert.throws(() => modelReservation(variant, { ...config, beginner: { contextLength: '4k' } }, hardware, []), /supported conversation context/);
  }
});
