import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('the card list preselects the card the loaded model runs on, and the sidebar card says the version and the card', async () => {
  const models = await readFile(new URL('../src/components/NativeModels.tsx', import.meta.url), 'utf8');
  assert.ok(models.indexOf('const loadedSelected = ') < models.indexOf('const loadedRoles = '));
  assert.match(models, /fromLegacy\(loadedSelected\.gpus, roleCards, 'as_given'\)/);
  // the order: the person's pick, then the cards a running model uses, then the roles its run config holds (#6117), then the recommendation
  assert.match(models, /roles \? normalizeRoles\(roles, roleCards\) : loadedRoles \?\? savedRoles \?\? recommended\.roles/);
  const side = await readFile(new URL('../src/app/Sidebar.tsx', import.meta.url), 'utf8');
  assert.match(side, /nativeSize\?\.model \?\? nativeCurrent\.model\.toUpperCase\(\)/);
  assert.match(side, /`GPU \$\{g \+ 1\}`/);
  assert.ok(!side.includes('nativeCurrent?.apiModelId'));
});
