/** The version list: for one base model, the recommendation and the downloaded versions on top, every other version once below, a fit word per row. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/native-fit.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { versionGroups, versionFit, recommendedVersions, PICK_NOTE } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const source = await readFile(new URL('../src/components/VersionPicker.tsx', import.meta.url), 'utf8');

const v = (sourceFamily, model, download_gb, extra = {}) => ({ id: sourceFamily === 'qwen' ? model.toLowerCase() : `${sourceFamily}-${model.toLowerCase()}`, sourceFamily, model, download_gb, memory_gb: download_gb * 0.62, ...extra });
const qwen38 = [v('qwen', 'Q2_0', 66.4), v('qwen', 'IQ2_XS', 68), v('qwen', 'IQ3_XXS', 75.8), v('qwen', 'IQ3_S', 83.6), v('swift', 'IQ2_XS', 68), v('swift', 'IQ3_S', 83.6), v('coder', 'IQ1_M', 58.4), v('unsloth', 'UD-Q4_K_XL', 111.3), v('gyro', 'Gyro-S', 58.5), v('gyro', 'Gyro-M', 92)];
const ids = list => list.map(x => x.id);

test('top: the recommendation and the downloaded versions; below: all the others; nothing doubled, nothing lost', () => {
  const have = new Set(['gyro-gyro-s', 'swift-iq2_xs']);
  const { top, more } = versionGroups(qwen38, new Set(), have, 'iq3_s');
  assert.deepEqual(ids(top), ['iq3_s', 'gyro-gyro-s', 'swift-iq2_xs'], 'the pick first, then the downloaded ones by size');
  assert.equal(top.length + more.length, qwen38.length);
  assert.deepEqual([...ids(top), ...ids(more)].sort(), ids(qwen38).sort());
  assert.ok(!more.some(x => top.includes(x)), 'a version at the top never appears again below');
  assert.deepEqual(more.map(x => x.download_gb), [...more.map(x => x.download_gb)].sort((a, b) => a - b), 'the others by size');
});

test('IQ3_S downloaded and recommended: it is at the top once, with the Downloaded tag only, and nowhere else', () => {
  const pick = { family: 'qwen', model: 'IQ3_S', why: '64 GB tier' };
  const recommended = recommendedVersions(qwen38, pick);
  assert.equal(recommended[0].v.id, 'iq3_s', 'the pick for this PC comes first');
  assert.equal(recommended[0].r.note, PICK_NOTE);
  const { top, more } = versionGroups(qwen38, new Set(), new Set(['iq3_s']), recommended.map(r => r.v.id));
  assert.equal(top[0].id, 'iq3_s');
  assert.equal([...top, ...more].filter(x => x.id === 'iq3_s').length, 1);
  // GUI-FB3 item 1: the dropdown carries no "Recommended" tag or badge (the owner, 11-02); the order and the "Downloaded" tag stay
  assert.doesNotMatch(source, /tr\("Recommended"\)/);
  assert.doesNotMatch(source, /version-tag--pick/);
  assert.match(source, /tr\(have \? <span className="version-row__tags"><span className="version-tag version-tag--have">\{tr\("Downloaded"\)\}<\/span><\/span> : null\)/);
});

test('the recommended versions from us come first, in order, then the downloaded ones, then the rest', () => {
  const pick = { family: 'qwen', model: 'IQ3_S', why: '64 GB tier' };
  const rec = recommendedVersions(qwen38, pick);
  assert.deepEqual(rec.map(r => r.v.id), ['iq3_s', 'gyro-gyro-s', 'swift-iq2_xs', 'unsloth-ud-q4_k_xl'], 'the pick, then the fixed picks');
  const have = new Set(['coder-iq1_m', 'iq3_s']);
  const { top, more } = versionGroups(qwen38, new Set(), have, rec.map(r => r.v.id));
  assert.deepEqual(ids(top), ['iq3_s', 'gyro-gyro-s', 'swift-iq2_xs', 'unsloth-ud-q4_k_xl', 'coder-iq1_m']);
  assert.equal(top.length + more.length, qwen38.length, 'none lost');
  assert.ok(!more.some(x => ids(top).includes(x.id)), 'none twice');
  // A base model whose versions are not in the fixed list still shows the engine's own pick.
  const other = [{ id: 'other-a', sourceFamily: 'other', model: 'A', download_gb: 50 }, { id: 'other-b', sourceFamily: 'other', model: 'B', download_gb: 60 }];
  assert.deepEqual(recommendedVersions(other, { family: 'other', model: 'B' }).map(r => r.v.id), ['other-b']);
  assert.deepEqual(recommendedVersions(other, null), [], 'no pick and no fixed picks: nothing is recommended');
});

test('no downloads and no pick: everything is in the one list below, and the first-run page leaves out what it shows elsewhere', () => {
  const { top, more } = versionGroups(qwen38, new Set(['iq3_s', 'gyro-gyro-s']), undefined, undefined);
  assert.equal(top.length, 0);
  assert.equal(more.length, qwen38.length - 2);
  assert.ok(!more.some(x => x.id === 'iq3_s' || x.id === 'gyro-gyro-s'));
});

test("the list writes Fits nowhere; only 'may run slower' is a word", () => {
  const big = { id: 'x', download_gb: 111, memory_gb: 78 }, small = { id: 'y', download_gb: 66, memory_gb: 35 };
  assert.equal(versionFit(small, { ram_gb: 128, vram_gb: 24 }), 'fits');
  assert.equal(versionFit(big, { ram_gb: 64, vram_gb: 12 }), 'ssd', 'part of the experts read from the SSD');
  assert.equal(versionFit(small, null), 'unknown', 'no hardware figures: no claim');
  assert.ok(!/'Fits'|"Fits"|Fits with SSD streaming/.test(source), 'no Fits word in the picker');
  assert.match(source, /state === 'ssd' \? <span className="version-row__fit"/, 'a version that fits has no fit line at all');
  assert.ok(!/Doesn't fit|Uses SSD/.test(source), "no Doesn't fit and no SSD figure any more");
  assert.match(source, /'may run slower'/);
  assert.ok(!/SSD streaming/.test(source), 'the old words are gone');
  assert.ok(!/nativeVersionLabel|plainLabel/.test(source), 'no Fastest, Compact, High quality words');
});

test("one rule (owner 2026-11-08): the weight files without the lookup table against RAM + VRAM, nothing taken off", () => {
  // NVFP4: 69 GB of weights. 63.1 GB of RAM and a 11.9 GB card hold 75 GB: it fits, whatever the engine's old RAM-only words said.
  const nvfp4 = { id: 'nvfp4-nvfp4', download_gb: 128, ram_needed_gb: 80, experts_gb: 68, memory_gb: 69, on_this_pc: 'does not fit' };
  assert.equal(versionFit(nvfp4, { ram_gb: 63.1, vram_gb: 11.9 }), 'fits');
  assert.equal(versionFit(nvfp4, { ram_gb: 31, vram_gb: 32 }), 'ssd', 'the V100 32 GB with 31 GB of RAM: 63 GB');
  assert.equal(versionFit({ id: 'u', download_gb: 84, memory_gb: 51.3 }, { ram_gb: 31, vram_gb: 32 }), 'fits', 'UD-IQ3_XXS / IQ3_S there: 51 GB');
  assert.equal(versionFit({ id: 'x', download_gb: 1, memory_gb: 63 }, { ram_gb: 31, vram_gb: 32 }), 'fits', 'exactly as much as there is');
  assert.equal(versionFit({ id: 'x', download_gb: 1, memory_gb: 63.1 }, { ram_gb: 31, vram_gb: 32 }), 'ssd');
  assert.equal(versionFit({ id: 'x', download_gb: 1, memory_gb: 40, fit_gb: 150 }, { ram_gb: 64, vram_gb: 24 }), 'ssd', "the catalog's own files size, not the CPU's share");
  assert.equal(versionFit({ id: 'x', download_gb: 1, memory_gb: 40, fit_gb: 60 }, { ram_gb: 64, vram_gb: 24 }), 'fits');
  assert.equal(versionFit({ ...nvfp4, on_this_pc: 'does not fit' }, { ram_gb: 32, vram_gb: 24 }), 'ssd', 'a smaller PC still runs it, slower');
  assert.equal(versionFit({ ...nvfp4, on_this_pc: 'tight' }, { ram_gb: 128, vram_gb: 32 }), 'fits');
});

test("no card, little memory, a Mac with too little: the same two words, never 'Doesn't fit' (owner 11-08)", () => {
  const v = { id: 'x', download_gb: 66, memory_gb: 35 };
  for (const words of ['does not fit: it needs one or more graphics cards', 'does not fit', 'does not fit: this Mac has too little memory'])
    assert.equal(versionFit({ ...v, on_this_pc: words }, { ram_gb: 8, vram_gb: 0 }), 'ssd', words);
  assert.equal(versionFit({ ...v, on_this_pc: 'does not fit' }, null), 'unknown', 'no hardware figures: no claim');
  assert.equal(versionFit({ ...v, on_this_pc: 'does not fit: about 120 GB of its experts would be read from the SSD' }, { ram_gb: 128, vram_gb: 24 }), 'fits', '35 GB fit 152 GB, whatever the old words said');
});

test('the size is the files without the lookup table (fit_gb): UD-IQ4_XS-v31 is 64.9 GB, 63 GB do not hold it', () => {
  const pc = { ram_gb: 31, vram_gb: 32 };
  assert.equal(versionFit({ id: 'ud-iq4_xs-v31', download_gb: 93.7, memory_gb: 61.4, fit_gb: 64.9 }, pc), 'ssd');
  assert.equal(versionFit({ id: 'ud-q3_k_xl', download_gb: 89.5, memory_gb: 57.2, fit_gb: 60.7 }, pc), 'fits');
  assert.equal(versionFit({ id: 'iq3_s', download_gb: 83.6, memory_gb: 51.3, fit_gb: 54.8 }, pc), 'fits');
});

test("the list is for one base model: the page passes only the chosen model's versions", async () => {
  const screen = await readFile(new URL('../src/components/NativeModels.tsx', import.meta.url), 'utf8');
  assert.match(screen, /<VersionPicker [^>]*versions=\{selectedFamily\?\.sizes \?\? \[\]\}/);
  assert.ok(!/models=\{families\}/.test(screen));
});

test('the picker has no maker groups, and the technical lines sit behind Show details', () => {
  assert.ok(!/VERSION_GROUPS|groupOpen|version-group/.test(source), 'no Swift / GSQ-RCO / Unsloth group headings');
  assert.ok(!/\(GSQ-RCO\)/.test(source), 'the maker is not part of the main name');
  assert.match(source, /details \? <span className="version-row__tech">/, 'maker, RAM + VRAM and the SSD figure are shown only with details on');
});
