// FIX-APP-0216 A1 (GUI-VIS-1117 Run B): a version card on the first-run "Your model" step said "58.5 GB Download size" although the files were already in the model folder;
// "Already on this computer" came one step later (Prepare). The card now uses the plan's answer (a look at the files' sizes) for the selected version.
import './helpers/isolate-profile.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const { code } = await transform(src('lib/onboarding-download.ts'), { loader: 'ts', format: 'esm' });
const { filesHere, versionCardSize } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('a version whose files are all here says so; part of them here shows what is left; no answer shows the download size', () => {
  const v = { download_gb: 58.5 };
  assert.deepEqual(versionCardSize(v, filesHere({ files_complete: true, partly_downloaded: true, download_left_gb: 0 })), { kind: 'here', text: 'Already on this computer' });
  assert.deepEqual(versionCardSize(v, filesHere({ files_complete: false, partly_downloaded: true, download_left_gb: 12.5 })), { kind: 'left', value: 12.5, caption: 'Left to download' });
  for (const answer of [undefined, null, filesHere(null), filesHere({ files_complete: false, partly_downloaded: false, download_left_gb: 58.5 }),
    filesHere({ files_complete: null, partly_downloaded: true }), filesHere({ files_complete: false, partly_downloaded: true, download_left_gb: 0 }),
    filesHere({ files_complete: false, partly_downloaded: true, download_left_gb: 70 })])
    assert.deepEqual(versionCardSize(v, answer), { kind: 'download', value: 58.5 }, JSON.stringify(answer));
});

test('the page asks for the plan on the "Your model" step and the card draws the answer', () => {
  const w = src('components/GuidedSetup.tsx');
  assert.match(w, /if \(step !== 2 \|\| !native\?\.available \|\| folderOpen \|\| !hereAsk\.length\)/);
  // the selected version and each card, one plan call after the other, with that version's own family, model and picture choice
  assert.match(w, /const hereAsk = \[version, \.\.\.cards\]\.filter\(/);
  assert.match(w, /for \(const v of hereAsk\) \{\s*if \(cancelled\)\s*return;\s*try \{\s*const value = await api\.post<Record<string, any>>\('\/api\/native\/plan', \{ \.\.\.values, family: v\.sourceFamily, model: v\.model, vision: v\.images \? 'yes' : 'no' \}\);\s*if \(!cancelled\)\s*setHereByVersion/);
  assert.ok(w.indexOf('const cards = ') < w.indexOf('const hereAsk = '), 'the cards exist before the effect reads them');
  assert.match(w, /cardSize\(v\)\.kind === 'here' \? <><Icon name="check" size=\{15\}\/>\{tr\(cardSize\(v\)\.text\)\}<\/>/);
  assert.match(w, /cardSize\(v\)\.kind === 'left' \? <>\{tr\("Left to download"\)\}<\/>/);
  assert.match(w, /<span className="onboard-version-size" data-here=\{cardSize\(v\)\.kind === 'here' \? 'true' : undefined\}>/);
  assert.match(src('styles/onboarding.css'), /\.onboard-version-size\[data-here\]\{font-size:14px;white-space:normal/);
  // the old caption stays for a version that is neither here nor partly here
  assert.match(w, /<>\{tr\("Download size"\)\}<\/>/);
});
