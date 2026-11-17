/** GUI feedback round 2 (11-02), items 1 to 8: the owner's own words are in C:\Users\AI-Server\Desktop\gui-feedback2.md. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync as read } from 'node:fs';
import { openingVision, shownVision, steadyNoReader } from '../src/lib/pictures-choice.ts';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const src = name => read(new URL(`../${name}`, import.meta.url), 'utf8');
async function bundle(file) {
  const result = await build({ entryPoints: [fileURLToPath(new URL(`../src/lib/${file}`, import.meta.url))], bundle: true, format: 'esm', platform: 'node', write: false, loader: { '.json': 'json' } });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}
globalThis.window ??= globalThis; globalThis.localStorage ??= { getItem: () => null, setItem: () => {} };
const { COMPACT_FILES_FROM, deliveryFolder, dirOf, rootOf } = await bundle('deliverables.ts');
const LANGS = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];

test('1: Read pictures is a plain checkmark: no explanation, no download button; Load does the rest', () => {
  const models = src('src/components/NativeModels.tsx');
  assert.match(models, /<strong>\{tr\("Read pictures"\)\}<\/strong>/);
  for (const gone of ['Understand photos, screenshots and charts', 'Turn it off to save memory', 'On by default', 'Download picture-reading files', 'Picture reading needs a few more files'])
    assert.ok(!models.includes(gone), `${gone} is gone from the page`);
  assert.doesNotMatch(models, /native-vision-missing/);
  // a failed or offline picture download is still said, in one line under the checkbox, once the model is loaded without them
  assert.match(models, /vision !== 'no' && !noReader && !active && loadedSelected\?\.picturesNote \? <p className="small muted native-vision-note" role="status">/);
  // Load still takes the picture step itself (the host side is pinned in load-pictures.test.mjs)
  assert.match(models, /picturesByLoad/);
});

test('2: the tick of a checkbox is white in light mode and keeps the dark ink in dark mode', () => {
  const css = src('src/styles/liquid.css');
  assert.match(css, /:root \{ --st-check-ink: #ffffff; \}/);
  assert.match(css, /\[data-theme='dark'\] \{ --st-check-ink: var\(--st-accent-ink\); \}/);
  assert.match(css, /input\[type='checkbox'\]:checked::after,\s*\.native-image-reading input\[type='checkbox'\]::after,\s*\.native-gpu-options input\[type='checkbox'\]::after \{\s*border-color: var\(--st-check-ink\);/);
});

test('3: the progress card names the step and nothing else', () => {
  const models = src('src/components/NativeModels.tsx'), css = src('src/styles/app.css');
  const card = models.slice(models.indexOf('export function NativeProgress'));
  for (const gone of ['The buttons below wait until this finishes.', 'Still working for {0}', 'native-wait-note', 'section-toolbar']) assert.ok(!models.includes(gone) || !card.includes(gone), gone);
  assert.ok(!models.includes('The buttons below wait until this finishes.'));
  assert.match(models, /<button type="button" className="native-progress__stop"[\s\S]*?\}>\{tr\(stopping \? 'Stopping…' : 'Stop'\)\}<\/button>/, 'Stop is a text button');
  assert.match(models, /aria-live="polite"/);
  // only transform and opacity move
  const frames = [...css.matchAll(/@keyframes (native-ring|native-breathe|native-glide) \{[^\n]*\n/g)].map(m => m[0]);
  assert.equal(frames.length, 3);
  for (const fr of frames) assert.doesNotMatch(fr, /\b(width|height|top|left|margin|padding)\s*:/);
  assert.match(css, /\.native-progress__bar\[data-indeterminate\] > i \{[^}]*animation: native-glide/);
});

test('4: what the box shows is the person\'s choice; the host\'s reports only open it', () => {
  const ready = { visionPrepared: true, visionEnabled: true, visionDevice: 'gpu', images: 'yes' };
  assert.equal(openingVision(undefined), 'yes');
  assert.equal(openingVision({ images: 'off', visionPrepared: false }), 'yes', 'a version with no picture files opens on');
  assert.equal(openingVision({ visionEnabled: false, visionPrepared: false }), 'yes', 'an old "off" saved while the files were missing opens on');
  assert.equal(openingVision({ visionEnabled: false, visionPrepared: true }), 'no', 'off by hand, with the files there');
  assert.equal(openingVision(ready), 'gpu');
  // the report changes while Load runs (the run file gets its picture section, the choice is saved, the instance appears): the choice stays
  for (const report of [undefined, { images: 'off', visionPrepared: false }, { visionEnabled: false, visionPrepared: true }, ready, { visionEnabled: false, visionPrepared: false }])
    for (const picked of ['yes', 'no', 'cpu']) assert.equal(shownVision(picked, report), picked);
  assert.equal(shownVision(undefined, ready), 'gpu');
  // the probe for the engine's picture reader can drop out while a job runs: the answer from before the job is kept
  let last = steadyNoReader(false, false, undefined, 'iq3_s');
  assert.equal(last.value, false);
  last = steadyNoReader(true, true, last, 'iq3_s');
  assert.equal(last.value, false, 'a job is running: the dropout is not shown');
  last = steadyNoReader(true, false, last, 'iq3_s');
  assert.equal(last.value, true, 'the job is over: the real answer shows');
  assert.equal(steadyNoReader(true, true, { id: 'other', value: false }, 'iq3_s').value, true, 'another version never inherits it');
  // the page uses all three
  const models = src('src/components/NativeModels.tsx');
  assert.match(models, /noReaderSeen\.current = steadyNoReader\(/);
  assert.match(models, /const known = visionPicks\.current\.get\(selectedSize\.id\), value = shownVision\(known, installed\);/);
  assert.match(models, /if \(known === undefined\)\s*visionPicks\.current\.set\(selectedSize\.id, value\);/, 'the first value is remembered: later reports of the host never change it');
  assert.doesNotMatch(models, /installed\.images, installed\.visionEnabled, installed\.visionDevice/, 'the host\'s picture fields no longer re-run the sync');
  assert.doesNotMatch(models, /installed\?\.visionEnabled, installed\?\.visionDevice/);
  // the host keeps the last known reader answer when its probe fails
  assert.match(src('server/native-control.mjs'), /catch\{visionReader=snapshot\.visionReader===true;\}/);
  // a locked box stays clearly on
  assert.match(src('src/styles/liquid.css'), /\.native-image-reading input\[type='checkbox'\]:disabled:checked \{\s*opacity: 0\.8;/);
});

test('5: no separator under Read pictures', () => {
  const css = src('src/styles/annotation-fixes.css');
  const line = css.split('\n').find(l => l.startsWith('.native-model-actions { display:flex;'));
  assert.ok(line && !/border-top/.test(line), line);
});

test('6: the running-chat ring is before the name, an SVG circle, and turns with the existing keyframes', () => {
  const bar = src('src/app/Sidebar.tsx'), css = src('src/styles/liquid.css');
  assert.ok(bar.indexOf('navsession__working') < bar.indexOf('navsession__title'));
  assert.match(css, /\.navsession__arc \{[^}]*stroke-linecap: round;[^}]*stroke-dasharray/);
  assert.match(css, /shape-rendering: geometricPrecision/);
  assert.doesNotMatch(css, /\.navsession__working \{[^}]*position: absolute/, 'in the row\'s flow, not in the right gutter');
});

test('7: the chat\'s time and token count have no fixed widths', () => {
  const css = src('src/styles/app.css');
  const rules = css.split('\n').filter(l => l.startsWith('.live-status'));
  assert.ok(rules.length >= 4);
  for (const r of rules) assert.doesNotMatch(r, /grid-template-columns/);
  assert.match(css, /\.live-status \{ display: flex; flex-wrap: wrap;/);
  assert.match(css, /\.live-status__time, \.live-status__tokens \{ flex: none; font-variant-numeric: tabular-nums;/);
  assert.match(css, /\.live-status__verb \{ flex: 0 1 24ch; min-width: 0;/, 'the word is the one that gives way');
});

const f = (path, extra = {}) => ({ path, abs: `/w/${path}`, name: path.split('/').at(-1), kind: 'code', status: 'created', size: 10, lines: 3, added: 3, removed: 0, ...extra });

test('8: where a run delivered to, from its files', () => {
  assert.equal(dirOf('site/css/a.css'), 'site/css');
  assert.equal(dirOf('a.css'), '');
  assert.equal(dirOf('site\\css\\a.css'), 'site/css');
  assert.equal(rootOf({ abs: '/w/pagoda/site/a.css', path: 'site/a.css' }), '/w/pagoda');
  assert.equal(rootOf({ abs: 'C:\\w\\pagoda\\site\\a.css', path: 'site/a.css' }), 'C:/w/pagoda');
  assert.equal(rootOf(undefined), '');
  // the file the answer names decides
  const files = [f('site/index.html', { kind: 'html', order: 1 }), f('site/a.css', { order: 2 }), f('notes/readme.md', { kind: 'markdown', order: 3 })];
  assert.equal(deliveryFolder(files, 'Open site/index.html'), 'site');
  assert.equal(deliveryFolder(files, 'See notes/readme.md'), 'notes');
  // nothing previewable: the top-level folder with the most files
  const code = [f('src/a.ts', { order: 1 }), f('src/b.ts', { order: 2 }), f('src/c.ts', { order: 3 }), f('docs/x.ts', { order: 4 })];
  assert.equal(deliveryFolder(code, ''), 'src');
  assert.equal(deliveryFolder([f('x.ts', { order: 1 })], ''), '', 'a file in the chat folder itself: the chat folder');
  assert.equal(deliveryFolder([], ''), '');
  assert.equal(deliveryFolder(undefined, ''), '');
  // 379 files
  const many = [f('site/index.html', { kind: 'html', order: 379 }), ...Array.from({ length: 378 }, (_, i) => f(`site/assets/p${i}.js`, { order: i + 1 }))];
  assert.equal(deliveryFolder(many, 'I stopped here.'), 'site');
});

test('8: a few files keep their cards; many are one compact block with an accessible, keyboard-usable count', () => {
  assert.equal(COMPACT_FILES_FROM, 6);
  const chat = src('src/screens/Chat.tsx'), parts = src('src/components/Deliverables.tsx'), css = src('src/styles/liquid.css');
  assert.match(chat, /m\.deliverables\.length >= COMPACT_FILES_FROM \? <FilesSummary files=\{m\.deliverables\}/);
  assert.match(chat, /<FileCard key=\{f\.path\}/, 'a short list is unchanged');
  const block = parts.slice(parts.indexOf('export function FilesSummary'), parts.indexOf('The saved skills an answer used'));
  assert.match(block, /<button type="button" className="fsum__toggle" aria-expanded=\{open\} aria-controls=\{id\}/, 'a real button: Enter and Space work, the state is announced');
  assert.match(block, /className="fsum__open" aria-label=\{tr\("Open \{0\}", \[p\.name\]\)\}/, 'each folder button says which folder');
  assert.match(block, /\{open \? <ul className="fsum__list" id=\{id\}>/, 'the list is closed until asked');
  assert.match(block, /<button type="button" className="fsum__file"/, 'every row is a button');
  assert.match(css, /\.fsum__list \{[^}]*max-height: min\(320px, 50vh\);[^}]*overflow: auto;/, 'long lists scroll inside the block');
  assert.match(css, /\.fsum__toggle:focus-visible/);
  assert.match(css, /prefers-reduced-motion: reduce\) \{\s*\.fsum__chev/);
});

test('"Leave room for my work" everywhere the old name was, and the new words in nine languages', () => {
  for (const file of ['src/components/NativeModels.tsx', 'src/components/GuidedSetup.tsx', 'src/lib/plain-errors.ts']) assert.ok(!src(file).includes('Leave 20%'), file);
  assert.match(src('src/components/NativeModels.tsx'), /Leave room for my work/);
  assert.match(src('src/components/GuidedSetup.tsx'), /title: 'Leave room for my work', hint: 'Keeps 5-8 GB of memory free for other programs\.'/);
  assert.match(src('src/lib/plain-errors.ts'), /choose “Leave room for my work”/);
  const cat = Object.fromEntries(LANGS.map(l => [l, JSON.parse(src(`src/locales/gui-${l}.json`))]));
  const keys = Object.keys(cat.en);
  assert.equal(keys.length, 6);
  for (const l of LANGS) {
    assert.deepEqual(Object.keys(cat[l]), keys, l);
    for (const k of keys) {
      assert.ok(cat[l][k] && cat[l][k].trim(), `${l} ${k}`);
      assert.deepEqual((cat[l][k].match(/\{\d+\}/g) ?? []).sort(), (cat.en[k].match(/\{\d+\}/g) ?? []).sort(), `${l} ${k} keeps its placeholders`);
      if (l !== 'en') assert.notEqual(cat[l][k], cat.en[k], `${l} ${k} is translated`);
    }
  }
  assert.equal(cat.en.gf01, 'Leave room for my work');
  assert.equal(cat.en.gf02, 'Keeps 5-8 GB of memory free for other programs.');
});
