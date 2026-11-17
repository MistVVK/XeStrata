// #5529: setup without a local engine. The first-run engine card gets "Only connect to Strata on another computer" (in every state) with a muted
// line under it. The click downloads nothing: setup ends dismissed (as "Go straight to the app"), and Settings > Models opens with a Custom
// model API connection already started. The engine card comes back while setup is dismissed but not complete, and the Models page has a way to it.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const here = new URL('..', import.meta.url);
const src = rel => readFileSync(fileURLToPath(new URL(rel, here)), 'utf8').replace(/\r\n/g, '\n');
const built = await build({ entryPoints: [fileURLToPath(new URL('src/lib/connect-only.ts', here))], bundle: true, write: false, platform: 'node', format: 'esm' });
const M = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

test('a request for the custom connection is taken once', () => {
  assert.equal(M.takeCustomConnectionRequest(), false);
  M.requestCustomConnection();
  assert.equal(M.takeCustomConnectionRequest(), true);
  assert.equal(M.takeCustomConnectionRequest(), false, 'the next opening of Models settings starts nothing');
});

test('the click ends setup dismissed, opens Settings > Models and asks for the custom connection; no engine download', async () => {
  const calls = [];
  const deps = {
    completeSetupStep: async (step, choices, dismissed) => { calls.push(['completeSetupStep', step, choices, dismissed]); },
    openSetup: open => calls.push(['openSetup', open]),
    setSettingsSection: section => { const pending = M.takeCustomConnectionRequest(); if (pending) M.requestCustomConnection(); calls.push(['setSettingsSection', section, 'request pending:', pending]); },
    openSettings: () => calls.push(['openSettings']),
  };
  await M.connectOnly(deps);
  assert.deepEqual(calls, [
    ['completeSetupStep', 1, { onboarding: { step: 1, manual: true } }, true],
    ['openSetup', false],
    ['setSettingsSection', 'Models', 'request pending:', true],
    ['openSettings'],
  ]);
  assert.equal(calls.some(c => /download|update-engine/.test(JSON.stringify(c))), false, 'nothing is downloaded');
  assert.equal(M.takeCustomConnectionRequest(), true, 'the request waits for the Models settings');
  assert.equal(M.takeCustomConnectionRequest(), false);
});

test('when setup cannot be saved, nothing opens and nothing is asked for', async () => {
  const calls = [];
  const deps = { completeSetupStep: async () => { throw new Error('host is gone'); }, openSetup: () => calls.push('openSetup'), setSettingsSection: () => calls.push('section'), openSettings: () => calls.push('open') };
  await assert.rejects(() => M.connectOnly(deps), /host is gone/);
  assert.deepEqual(calls, []);
  assert.equal(M.takeCustomConnectionRequest(), false);
});

test('the link and its muted line are in the footer every card state shares', () => {
  const s = src('src/components/EngineFirstRun.tsx');
  const footer = s.slice(s.indexOf('<footer className="onboard-footer onboard-footer--welcome">{actions}'));
  assert.match(footer, /\{actions\}<button type="button" className="onboard-continue-anyway" disabled=\{busy \|\| running\} onClick=\{\(\) => void connectOnlyClick\(\)\}>\{tr\("Only connect to Strata on another computer"\)\}<\/button><small>\{tr\("Nothing is downloaded\. You can add the engine later from the Models page\."\)\}<\/small><\/footer>/);
  // the footer follows the branch that sets `actions` for offer, offline, error (the list could not be read), wrong make and the rest (none, installed, unavailable)
  const branch = s.slice(s.indexOf("if (info.status === 'offer' && info.offer)"), s.indexOf('<footer className="onboard-footer onboard-footer--welcome">{actions}'));
  assert.equal((branch.match(/actions = /g) ?? []).length, 5, 'every state sets its actions and so gets the footer');
  assert.match(branch, /info\.status === 'offline'/);
  assert.match(branch, /info.status === 'error'/);
  assert.match(branch, /info\.status === 'wrong-make'/);
  // the handler uses connectOnly with the store and the router, and never starts the engine download
  const handler = s.slice(s.indexOf('async function connectOnlyClick()'), s.indexOf("const running = job?.state === 'running';"));
  assert.match(handler, /connectOnly\(\{ completeSetupStep, openSetup, setSettingsSection, openSettings: \(\) => navigate\('chat', undefined, \{ settings: '1' \}\) \}\)/);
  assert.doesNotMatch(handler, /update-engine|native\/operations|from-file/);
});

test('Models settings start the custom connection when asked', () => {
  const s = src('src/components/SettingsModels.tsx');
  assert.match(s, /import \{ takeCustomConnectionRequest \} from '\.\.\/lib\/connect-only';/);
  assert.match(s, /useEffect\(\(\) => \{ if \(takeCustomConnectionRequest\(\)\) \{ reset\(\); setDraft\(\{ kind: 'custom', provider: null, name: '', baseUrl: '', protocol: 'openai-chat', apiKey: '', models: \[\] \}\); \} \}, \[\]\);/);
});

test('the engine card comes back while setup is dismissed but not complete, and the Models page leads to it', () => {
  const setup = src('src/screens/Setup.tsx');
  assert.match(setup, /if \(!engineStepDone && !setup\?\.complete && \(needsEngine \|\| cardShown\)\)\s*\n\s*return <EngineFirstRun/);
  assert.doesNotMatch(setup.slice(setup.indexOf('if (!engineStepDone')), /dismissed/, 'the card does not look at "dismissed"');
  const warning = src('src/components/EngineWarning.tsx');
  assert.match(warning, /onSetup \? <p><Btn size="sm" onClick=\{onSetup\}>\{tr\("Run guided setup"\)\}<\/Btn><\/p> : null/);
  assert.match(src('src/components/NativeModels.tsx'), /<EngineWarning native=\{native\} onSetup=\{\(\) => go\('setup'\)\}\/>/);
  // the first-run card reads Control's local answer, so the card shows again after a restart while no engine is installed
  assert.match(src('src/components/EngineFirstRun.tsx'), /setNeeded\(!!s\.engine && s\.engine\.usable === false\)/);
});

test('the two sentences are in the guic catalog in all ten languages', () => {
  const dir = new URL('src/locales/', here);
  const langs = readdirSync(fileURLToPath(dir)).filter(f => /^guic-[a-z]{2}\.json$/.test(f));
  assert.equal(langs.length, 10);
  const en = JSON.parse(readFileSync(fileURLToPath(new URL('guic-en.json', dir)), 'utf8'));
  const values = Object.values(en);
  for (const text of ['Only connect to Strata on another computer', 'Nothing is downloaded. You can add the engine later from the Models page.']) {
    const key = Object.keys(en).find(k => en[k] === text);
    assert.ok(key, text);
    for (const f of langs) assert.ok(JSON.parse(readFileSync(fileURLToPath(new URL(f, dir)), 'utf8'))[key], `${f} has ${key}`);
  }
  assert.ok(values.length >= 24);
});
