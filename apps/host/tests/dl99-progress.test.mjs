import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createEngineStep } from '../server/engine-step.mjs';

/* DL99 (a Qwen3.6 UD-Q4_K_XL download that "stays at 99%"): what the setup wizard says and counts after the last byte. Replayed
   from the install job's own progress as the app host saw it on a real run (Boston 2026-11-15, phaseA-tests/dl99/out1). */
const compiled = await build({
  stdin: { contents: "export * from './src/lib/download-eta'; export * from './src/lib/onboarding-download'; export * from './src/lib/i18n';", resolveDir: process.cwd() },
  bundle: true, write: false, platform: 'node', format: 'esm',
});
const lib = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
lib.setTranslationLanguage('en');

const engineDone = { kind: 'engine', name: 'Engine', state: 'done', done_gb: 0.67, total_gb: 0.67 };
const model = (state, done) => ({ kind: 'model', name: 'Model files', state, done_gb: done, total_gb: 22.854, files_done: state === 'done' ? 1 : 0, files_total: 1 });
const reader = (state, done) => ({ kind: 'vision', name: 'Picture reader', state, done_gb: done, total_gb: 0.899, files_done: state === 'done' ? 1 : 0, files_total: 1 });

test('DL99: the wizard says what setup is doing (its own sentence), not "Downloading the picture reader" while the model file is checked', () => {
  const checking = { running: true, step: '5/7: downloading Qwen3.6-35B-A3B UD-Q4_K_XL', phase: 'verify',
    label: 'Checking that the model files are complete (a few minutes for a big model; nothing is downloaded)',
    download: { file: 'checking files', done_gb: 4.13, total_gb: 22.85, remaining_gb: 18.72, percent: 18 },
    parts: [model('done', 22.854), reader('waiting', 0), engineDone] };
  const progress = lib.installProgress(checking);
  assert.equal(lib.guidedStatusText(checking, progress), 'Checking that the model files are complete (a few minutes for a big model; nothing is downloaded)');
  const tokenizer = { running: true, step: '6/7: preparing the model for Strata', phase: 'tokenizer', label: 'Preparing the tokenizer', parts: [engineDone] };
  assert.equal(lib.guidedStatusText(tokenizer, lib.installProgress(tokenizer)), 'Preparing the tokenizer');
  // no sentence from setup yet: the part being fetched, as before
  const early = { running: true, parts: [model('downloading', 3.1), reader('waiting', 0)] };
  assert.equal(lib.guidedStatusText(early, lib.installProgress(early)), 'Downloading model files…');
  assert.equal(lib.guidedStatusText({ running: true }, undefined), 'Preparing your model…');
  // a server that has not answered yet keeps its own waiting line first
  const waiting = { running: true, phase: 'waiting', label: 'Waiting for huggingface.co to answer (25 s so far)', parts: [model('downloading', 0)] };
  assert.equal(lib.guidedStatusText(waiting, lib.installProgress(waiting)), 'Waiting for huggingface.co to answer (25 s so far)');
});

test('DL99: the model check shows its own figure, not a frozen 96%; the counter does not call it "left to download"', () => {
  const step = createEngineStep({ control: null });
  const checking = { running: true, step: '5/7: downloading', label: 'Checking…',
    download: { file: 'checking files', done_gb: 4.13, total_gb: 22.85, remaining_gb: 18.72, percent: 18 },
    parts: [model('done', 22.854), reader('waiting', 0)] };
  const p = lib.installProgress(step.merge(checking));
  assert.equal(p.percent, 18);
  assert.ok(Math.abs(p.done_gb - 4.13) < 0.01 && Math.abs(p.total_gb - 22.85) < 0.01, JSON.stringify(p));
  const counter = lib.downloadCounter(23.8, p, true);
  assert.notEqual(counter.label, 'GB left to download');
});

test('DL99: once every file is in, the bar says 100%, not 99% (and never 0.7 / 0.7 GB of the engine alone)', () => {
  const done = { running: true, step: '6/7: preparing the model for Strata', label: 'Preparing the tokenizer',
    parts: [model('done', 22.854), reader('done', 0.899), engineDone] };
  const p = lib.installProgress(done);
  assert.equal(p.percent, 100);
  assert.ok(p.total_gb > 24, JSON.stringify(p));
  const still = lib.installProgress({ parts: [model('done', 22.854), reader('downloading', 0.6), engineDone] });
  assert.ok(still.percent < 100 && still.percent >= 96, JSON.stringify(still));
  const almost = lib.installProgress({ parts: [model('downloading', 22.85), reader('done', 0.899)] });
  assert.equal(almost.percent, 99, 'a part still downloading stays below 100');
});

const NOT_READY = "Your model is downloaded, but the engine for this PC isn't ready yet. Open Settings > About to install it.";

test('DL99: an install that ended with every file in, for a model the missing engine keeps from being ready, says so on the wizard page', async () => {
  const { readFileSync } = await import('node:fs');
  // when: the job is done, the model is not ready, the engine for this PC is missing or can't be used
  assert.equal(lib.engineNotReadyAfterInstall('done', false, false), true);
  assert.equal(lib.engineNotReadyAfterInstall('done', undefined, false), true, 'a model the list does not show as ready yet');
  assert.equal(lib.engineNotReadyAfterInstall('done', true, false), false, 'a ready model moves on to the next step');
  assert.equal(lib.engineNotReadyAfterInstall('done', false, true), false, 'a usable engine: not this case');
  assert.equal(lib.engineNotReadyAfterInstall('done', false, undefined), false, 'nothing known about the engine yet');
  assert.equal(lib.engineNotReadyAfterInstall('running', false, false), false, 'not while the job runs');
  assert.equal(lib.engineNotReadyAfterInstall('error', false, false), false, 'a failed job shows its own error');
  // the wizard shows the owner's sentence (11-15) for that case, through the translation catalog
  const page = readFileSync(new URL('../src/components/GuidedSetup.tsx', import.meta.url), 'utf8');
  assert.ok(page.includes(`engineNotReadyAfterInstall(installJob?.state, installed?.ready, native?.engine?.usable) ? `), 'hooked to the install job, the model and the engine');
  assert.ok(page.includes(`tr("${NOT_READY}")`));
  assert.equal(lib.tr(NOT_READY), NOT_READY);
  // all nine languages have it (scripts/make-engine-locales.py)
  const en = JSON.parse(readFileSync(new URL('../src/locales/engine-en.json', import.meta.url), 'utf8'));
  const key = Object.keys(en).find(k => en[k] === NOT_READY);
  assert.ok(key, 'in the English catalog');
  for (const l of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr']) {
    const t = JSON.parse(readFileSync(new URL(`../src/locales/engine-${l}.json`, import.meta.url), 'utf8'))[key];
    assert.ok(t && t !== NOT_READY && t.includes('>'), `${l}: ${t}`);
  }
  await lib.loadCatalog('de');
  lib.setTranslationLanguage('de');
  assert.match(lib.tr(NOT_READY), /^Dein Modell ist heruntergeladen/);
  lib.setTranslationLanguage('en');
});
