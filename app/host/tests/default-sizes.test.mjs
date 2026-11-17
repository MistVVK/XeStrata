import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync as read } from 'node:fs';
import { DEFAULT_CONTENT_FS, DEFAULT_UI_SCALE, SIZE_DEFAULTS_VERSION, savedSizes } from '../src/lib/default-sizes.ts';

const src = name => read(new URL(`../${name}`, import.meta.url), 'utf8');
const MIN = 10, MAX = 22;

test('the defaults are a 12 px reading size and a 90 percent interface', () => {
  assert.equal(DEFAULT_CONTENT_FS, 12);
  assert.equal(DEFAULT_UI_SCALE, 0.9);
  const store = src('src/lib/store.ts');
  assert.match(store, /contentFontSize: DEFAULT_CONTENT_FS,/);
  assert.match(store, /uiScale: DEFAULT_UI_SCALE,/);
  assert.match(store, /contentFontSize: PREFS_DEFAULTS\.contentFontSize,[\s\S]*uiScale: PREFS_DEFAULTS\.uiScale,/, 'Reset appearance returns to the defaults, whatever they are');
  assert.doesNotMatch(src('src/styles/app.css'), /--st-content-fs, 14px/, 'the stylesheet fallbacks follow the new default');
});

test('nothing saved gives the defaults; a size the person chose stays', () => {
  assert.deepEqual(savedSizes(null, MIN, MAX), { contentFontSize: 12, uiScale: 0.9 });
  assert.deepEqual(savedSizes({}, MIN, MAX), { contentFontSize: 12, uiScale: 0.9 });
  assert.deepEqual(savedSizes({ contentFontSize: 18, uiScale: 1.25 }, MIN, MAX), { contentFontSize: 18, uiScale: 1.25 }, 'an old save that chose a size keeps it');
  assert.deepEqual(savedSizes({ contentFontSize: 11, uiScale: 1.1 }, MIN, MAX), { contentFontSize: 11, uiScale: 1.1 });
});

test('an old save that holds the old defaults moves to the new ones once, then everything is kept as saved', () => {
  assert.deepEqual(savedSizes({ contentFontSize: 14, uiScale: 1 }, MIN, MAX), { contentFontSize: 12, uiScale: 0.9 });
  const marked = { sizeDefaults: SIZE_DEFAULTS_VERSION };
  assert.deepEqual(savedSizes({ ...marked, contentFontSize: 14, uiScale: 1 }, MIN, MAX), { contentFontSize: 14, uiScale: 1 }, 'picked after the change: 14 px and 100% are a choice');
  assert.deepEqual(savedSizes({ ...marked, contentFontSize: 12, uiScale: 0.9 }, MIN, MAX), { contentFontSize: 12, uiScale: 0.9 });
});

test('a damaged save falls back to the defaults and the reading size stays inside its range', () => {
  assert.deepEqual(savedSizes({ contentFontSize: 'big', uiScale: 7 }, MIN, MAX), { contentFontSize: 12, uiScale: 0.9 });
  assert.equal(savedSizes({ sizeDefaults: SIZE_DEFAULTS_VERSION, contentFontSize: 99, uiScale: 1 }, MIN, MAX).contentFontSize, MAX);
  assert.equal(savedSizes({ sizeDefaults: SIZE_DEFAULTS_VERSION, contentFontSize: 3, uiScale: 1 }, MIN, MAX).contentFontSize, MIN);
});

test('code keeps its 12 px floor, so at the default it is the size of the prose and stays apart by its font, bar and background', () => {
  const css = src('src/styles/app.css');
  assert.match(css, /--st-fs-read-sm: max\(12px, calc\(var\(--st-content-fs, 12px\) \* \.8125\)\);/);
  assert.match(css, /\.st-bubble \.st-code pre, \.st-bubble \.st-code pre code, \.st-bubble \.st-code \.syntax-text \{ font-size: var\(--st-fs-read-sm\); \}/);
});

test('the Settings controls show the stored values as they are, with no second default of their own', () => {
  const settings = src('src/screens/Settings.tsx');
  assert.match(settings, /value=\{prefs\.contentFontSize\}/);
  assert.match(settings, /value=\{String\(prefs\.uiScale\)\}/);
});
