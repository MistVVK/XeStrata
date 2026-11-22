import { fileURLToPath } from 'node:url';
import { prepareStagedHarnessBudget } from './harness-budget.mjs';

const staged = fileURLToPath(new URL('../src-tauri/resources/harness/', import.meta.url));
const patch = prepareStagedHarnessBudget(staged);
if (patch) console.log(`Verified ${patch.id}: ${patch.afterSha256}`);
