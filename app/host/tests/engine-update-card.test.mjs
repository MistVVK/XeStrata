import test from 'node:test';
import { readFileSync as read } from 'node:fs';
import assert from 'node:assert/strict';

const src = name => read(new URL(`../${name}`, import.meta.url), 'utf8');

test('the host passes the kept update check to the app without asking the network', () => {
  const host = src('server/native-control.mjs');
  assert.match(host, /updateAvailable:engine\.update_available\?\?null/, 'the snapshot carries what `components engine` kept');
  assert.match(src('src/lib/native.ts'), /updateAvailable\?:\{engine\?:string\|null;app\?:string\|null\}\|null/);
});

test('one card, one button, only when an engine is offered, and it runs the existing Update engine job', () => {
  const card = src('src/components/EngineUpdateCard.tsx');
  assert.match(card, /offer && offer\.engine\)/);
  assert.match(card, /nativeAction\('operations', \{ kind: 'update-engine' \}\)/);
  assert.equal((card.match(/<Btn /g) ?? []).length, 1, 'one button');
  assert.doesNotMatch(card, /fetch\(|api\.get|api\.post|XMLHttpRequest/, 'showing the card reads no network: only the kept state');
  assert.match(card, /disabled=\{busy \|\| pending \|\| running\}/, 'not while a job runs or a model is loaded');
  assert.doesNotMatch(card, /process\.env|STRATA_|environment variable/i, 'no environment variables in the words');
});

test('the card is on the Models screen once and nowhere else', () => {
  const models = src('src/components/NativeModels.tsx');
  assert.equal((models.match(/<EngineUpdateCard /g) ?? []).length, 1);
  assert.match(models, /<EngineUpdateCard native=\{native\} busy=\{!!active\}\/>/);
  assert.doesNotMatch(src('src/components/GuidedSetup.tsx'), /EngineUpdateCard/);
});

test('an updated engine is reported as an engine, not as "Strata is updated"', () => {
  const host = src('server/native-control.mjs');
  assert.match(host, /isEngine=c=>c==='engine'\|\|\(typeof c==='string'&&c\.startsWith\('strata-'\)&&c!=='strata-control'\)/);
  assert.match(host, /isEngine\(x\.component\)\?'The engine is updated and ready\.'/);
});
