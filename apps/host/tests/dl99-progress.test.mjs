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
