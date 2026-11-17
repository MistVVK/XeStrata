// First-run walkthrough (GUI-VIS-1117): the Prepare page said "Downloading the model... 111.3 GB to download / Download model" for files already in the model folder;
// the file check showed the model's figures under the Engine row; the tuning card had a gap under its artwork; "30d / 7d" stayed English in every other language.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const { code } = await transform(src('lib/onboarding-download.ts'), { loader: 'ts', format: 'esm' });
const { downloadCounter, filesHere, prepareTitle } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const LANGS = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];

const complete = filesHere({ files_complete: true, partly_downloaded: true, download_left_gb: 0 });
const partly = filesHere({ files_complete: false, partly_downloaded: true, download_left_gb: 61.2 });
const partlyUnknown = filesHere({ files_complete: null, partly_downloaded: true });
const none = filesHere({ files_complete: false, partly_downloaded: false, download_left_gb: 111.3 });

test('filesHere reads Control\'s plan and trusts only a plain true', () => {
  assert.deepEqual(complete, { complete: true, partly: true, left: 0 });
  assert.deepEqual(partly, { complete: false, partly: true, left: 61.2 });
  assert.deepEqual(partlyUnknown, { complete: false, partly: true, left: null });
  assert.deepEqual(none, { complete: false, partly: false, left: 111.3 });
  for (const plan of [null, undefined, {}, { files_complete: 'yes', partly_downloaded: 1, download_left_gb: '5' }, { download_left_gb: -3 }, { download_left_gb: NaN }])
    assert.deepEqual(filesHere(plan), { complete: false, partly: false, left: null }, JSON.stringify(plan));
});

test('the counter: files here say so with no GB; part of the model says what is left; nothing here keeps the size', () => {
  assert.deepEqual(downloadCounter(111.3, undefined, false, complete), { value: undefined, label: 'Already on this computer', key: 'here', plain: true });
  assert.deepEqual(downloadCounter(111.3, undefined, false, partly), { value: 61.2, label: 'GB left to download', key: 'left', note: 'Part of the model is already here' });
  assert.deepEqual(downloadCounter(111.3, undefined, false, partlyUnknown), { value: undefined, label: 'Part of the model is already here', key: 'part', plain: true });
  assert.deepEqual(downloadCounter(111.3, undefined, false, none), { value: 111.3, label: 'GB to download', key: 'size' });
  assert.deepEqual(downloadCounter(111.3, undefined, false), { value: 111.3, label: 'GB to download', key: 'size' }, 'no plan yet: as before');
  // a running job keeps its own figures, whatever was here when it started
  assert.equal(downloadCounter(111.3, { remaining_gb: 40 }, true, complete).value, 40);
  assert.equal(downloadCounter(111.3, { file: 'checking files', percent: 16 }, true, complete).label, 'Preparing files…');
});

test('the title: checking the files is not a download; files that are here are only prepared', () => {
  assert.equal(prepareTitle(true, { file: 'checking files', percent: 16, done_gb: 8.4, total_gb: 49.4 }, complete), 'Checking the model files...');
  assert.equal(prepareTitle(true, { file: 'checking files' }, none), 'Checking the model files...');
  assert.equal(prepareTitle(false, undefined, complete), 'Preparing the model...');
  assert.equal(prepareTitle(true, undefined, complete), 'Preparing the model...');
  assert.equal(prepareTitle(false, undefined, partly), 'Downloading the model...');
  assert.equal(prepareTitle(false, undefined, none), 'Downloading the model...');
  assert.equal(prepareTitle(true, { file: 'shard-2', percent: 40 }, none), 'Downloading the model...');
  assert.equal(prepareTitle(false, undefined, undefined), 'Downloading the model...');
});

test('the wizard uses them: title, counter, hint and button; no download wording when nothing is to download', () => {
  const w = src('components/GuidedSetup.tsx');
  assert.ok(w.includes('const here = filesHere(plan?.plan), counter = downloadCounter(plan?.plan?.download_gb ?? version?.download_gb, progress, preparing, here);'));
  assert.ok(w.includes("step === 3 ? prepareTitle(preparing, progress, here) :"));
  assert.ok(w.includes("here.complete ? 'Strata checks the files first.' : 'Downloads start only when you’re ready.'"));
  assert.ok(w.includes("here.complete ? 'Continue' : stopped ? 'Resume download' : 'Download model'"));
  assert.ok(w.includes('{tr(!preparing && !here.complete ? <Icon name="download" size={16}/> : null)}'));
  assert.doesNotMatch(w, /step === 3 \? 'Downloading the model\.\.\.' :/);
});

test('the bar and its figures come before the part rows (the figures used to sit under the Engine row)', () => {
  const w = src('components/GuidedSetup.tsx');
  const bar = w.indexOf('<div className="onboard-progress"'), meta = w.indexOf('<div className="onboard-progress-meta">'), parts = w.indexOf('<DownloadParts parts=');
  assert.ok(bar > 0 && meta > bar && parts > meta, `bar ${bar}, meta ${meta}, parts ${parts}`);
  assert.equal(w.split('<DownloadParts parts=').length - 1, 1);
  assert.match(src('styles/onboarding.css'), /\.onboard-progress-meta\+\.native-parts\{margin-top:12px\}/);
});

test('the tuning card\'s second row takes the spare height, so the artwork has no gap under it', () => {
  const css = src('styles/onboarding.css');
  assert.ok(css.includes('.onboard-tune-stage:has(.onboard-tuning-status){display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto 1fr;gap:0 20px}'));
  assert.ok(css.includes('.onboard-tune-stage:has(.onboard-tuning-status)>.onboard-tour{grid-column:2;grid-row:1/3}'));
});

test('the new words and the Usage periods are in the catalog in all ten languages', () => {
  const cat = Object.fromEntries(LANGS.map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/guic-${l}.json`, import.meta.url), 'utf8'))]));
  for (const text of ['Preparing the model...', 'Checking the model files...', 'Already on this computer', 'Part of the model is already here', 'Strata checks the files first.', '30d', '7d']) {
    const key = Object.keys(cat.en).find(k => cat.en[k] === text);
    assert.ok(key, `${text} is in guic-en.json`);
    for (const l of LANGS) assert.ok(cat[l][key] && cat[l][key].trim(), `${l}: ${text}`);
    assert.equal(/[^  ][;:?!]| [;:?!]/.test(cat.fr[key]), false, `French: ${text}`);
  }
  const key = t => Object.keys(cat.en).find(k => cat.en[k] === t);
  assert.notEqual(cat.ja[key('30d')], '30d'); assert.notEqual(cat.he[key('7d')], '7d');
  assert.match(cat.de[key('Strata checks the files first.')], /^Strata prüft/);
  assert.equal(cat.fr[key('30d')], '30 j');
  // the Usage dashboard passes these two words through tr()
  assert.ok(src('components/UsageDashboard.tsx').includes("tr(v === 'all' ? 'All' : v)"));
});
