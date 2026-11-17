/** Strata's output-budget correction for the staged pi-ai copy. */
import { createHash } from 'node:crypto';
import { lstatSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

export const HARNESS_BUDGET_PATCH = Object.freeze({
  id: 'strata-pi-ai-context-safety-v2',
  package: '@earendil-works/pi-ai',
  version: '1.1.0',
  file: 'node_modules/@earendil-works/pi-ai/dist/api/simple-options.js',
  beforeSha256: '9b3cc0ae6daa86fdc050bc0cb0f3f986401e64ccddbde6757183c9f8ad411be2',
});

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const oldClamp = 'const available = model.contextWindow - estimateContextTokens(context).tokens - CONTEXT_SAFETY_TOKENS;';
// Small windows also need room after tool results; a fixed margin must not
// consume all remaining estimated space. Keep the old 64K+ calculation.
const newClamp = `const remaining = model.contextWindow - estimateContextTokens(context).tokens;
    const safety = model.contextWindow >= 65536 ? CONTEXT_SAFETY_TOKENS
        : Math.min(CONTEXT_SAFETY_TOKENS, Math.floor(model.contextWindow / 16), Math.max(0, Math.floor(remaining / 2)));
    const available = remaining - safety;`;

/**
 * Patch only a flattened, disposable staged Harness. Returns the BUNDLE.json
 * provenance entry; the caller records it when writing the bundle manifest.
 * A repeat application or any unknown package/source fails before writing.
 */
export function patchStagedHarnessBudget(stagedRoot) {
  const fail = message => { throw new Error(`Harness output-budget patch: ${message}`); };
  const root = realpathSync(resolve(stagedRoot));
  const packageDir = join(root, 'node_modules', '@earendil-works', 'pi-ai');
  const target = join(root, HARNESS_BUDGET_PATCH.file);
  let manifest, before;
  try {
    manifest = JSON.parse(readFileSync(join(packageDir, 'package.json'), 'utf8'));
    // Flattening must have removed links; never write through a shared source file.
    if (realpathSync(target) !== target || !lstatSync(target).isFile() || lstatSync(target).nlink !== 1)
      fail('expected an unlinked staged module');
    before = readFileSync(target);
  } catch (error) {
    fail(`cannot read a standalone staged pi-ai dependency (${error.message})`);
  }
  if (manifest?.name !== HARNESS_BUDGET_PATCH.package || manifest?.version !== HARNESS_BUDGET_PATCH.version)
    fail('expected @earendil-works/pi-ai 1.1.0; review the patch before staging a different dependency');
  if (sha256(before) !== HARNESS_BUDGET_PATCH.beforeSha256)
    fail('unexpected simple-options.js SHA-256; review the dependency source before staging');
  const source = before.toString('utf8');
  if (source.split(oldClamp).length !== 2) fail('expected exactly one known context clamp');
  const after = Buffer.from(source.replace(oldClamp, newClamp));
  writeFileSync(target, after);
  return { ...HARNESS_BUDGET_PATCH, afterSha256: sha256(after) };
}
