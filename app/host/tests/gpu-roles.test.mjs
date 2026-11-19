/** GPU roles (Main / Helper / Off): src/lib/gpu-roles.ts and server/gpu-roles.mjs run the vectors strata_control/gpu_roles.py runs (tests/data/gpu_roles_vectors.json). */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/gpu-roles.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const R = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const S = await import('../server/gpu-roles.mjs');
const V = JSON.parse(await readFile(new URL('../../../terminal/tests/data/gpu_roles_vectors.json', import.meta.url), 'utf8'));
const roles = d => d === null || d === undefined ? d : Object.fromEntries(Object.entries(d).map(([k, v]) => [Number(k), v]));

test('recommend: the best card is Main, others Helper, 8 GB or less / built-in / unusable start Off', () => {
  for (const c of V.recommend) {
    const r = R.recommend(V.hw[c.hw]);
    assert.deepEqual(r.roles, roles(c.roles), c.hw);
    assert.equal(r.main, c.main, c.hw);
    assert.equal(r.why, c.why, c.hw);
  }
});

test('setRole: one click, exactly one Main among the cards that are on', () => {
  for (const c of V.set_role) {
    const out = R.setRole(roles(c.roles), V.hw[c.hw], c.card, c.role);
    assert.deepEqual(out.roles, roles(c.after), JSON.stringify(c));
    assert.equal(out.note, c.note, JSON.stringify(c));
  }
});

test('order: Helpers first, Main LAST, Off left out; one card is a number in the mirror', () => {
  for (const c of V.order) {
    assert.deepEqual(R.order(roles(c.roles), V.hw[c.hw]), c.gpu, JSON.stringify(c));
    assert.deepEqual(S.order(roles(c.roles), V.hw[c.hw]), c.gpu, 'server copy: ' + JSON.stringify(c));
  }
  const m = R.mirror({ 0: 'off', 1: 'main' }, V.hw.two);
  assert.equal(m.gpu, 1);
  assert.equal(m.gpu_order, 'as_given');
  assert.deepEqual(R.mirror({ 0: 'helper', 1: 'main' }, V.hw.two).gpu, [0, 1]);
  const none = R.mirror({ 0: 'off', 1: 'off' }, V.hw.two);
  assert.deepEqual([none.gpu, none.gpu_none], [[], true]);
  assert.deepEqual(S.mirror({ 0: 'helper', 1: 'main' }, V.hw.two), R.mirror({ 0: 'helper', 1: 'main' }, V.hw.two));
  assert.deepEqual(S.mirror({ 0: 'off', 1: 'off' }, V.hw.two), none);
});

test('warnings: only built-in graphics as Main, a card the engine cannot run, too little memory, CPU only', () => {
  for (const c of V.warnings) {
    const w = R.warnings(roles(c.roles), V.hw[c.hw], { spillGb: c.spill_gb ?? 0 });
    assert.deepEqual(w.map(x => x.code), c.codes, JSON.stringify(c));
    if (c.fix) assert.deepEqual(w[0].fix, c.fix);
    for (const x of w) assert.ok(x.text.length < 90, x.text);
  }
  // a slower card as Main is understood by people: no warning
  assert.deepEqual(R.warnings({ 0: 'main', 1: 'helper' }, V.hw.two), []);
  assert.equal(R.warnings({ 0: 'off', 1: 'off' }, V.hw.two, { engineRunsCpuOnly: true })[0].text, 'Performance is degraded on CPU only.');
  assert.equal(R.warnings({ 0: 'off', 1: 'off' }, V.hw.two)[0].text, R.NO_CARD_LINE);
});

test('legacy gpu settings show the roles the engine really uses', () => {
  for (const c of V.legacy) assert.deepEqual(R.fromLegacy(c.gpu, V.hw[c.hw], c.gpu_order), roles(c.roles), JSON.stringify(c));
});

test('saved roles: parse; the server copy agrees on which are usable', () => {
  assert.deepEqual(R.parseSaved({ 0: 'helper', 1: 'main' }), { 0: 'helper', 1: 'main' });
  for (const bad of [{ 0: 'boss' }, { 0: 'main', 1: 'main' }, 'x', {}, null]) { assert.equal(R.parseSaved(bad), null); assert.equal(S.parseSaved(bad), null); }
});

test('words (0217 item 12): alike cards are said once, with the Terminal\'s words; unlike cards keep their reason', () => {
  assert.equal(R.sentence(V.hw.equal), 'Both cards are RTX A4000, so they are alike: one is Main and the other helps.');
  assert.doesNotMatch(R.sentence(V.hw.equal), /the cards are alike/);
  const four = [0, 1, 2, 3].map(index => ({ index, name: 'NVIDIA RTX A4000', vram_gb: 16, speed: 50 }));
  assert.equal(R.sentence(four), 'All 4 cards are RTX A4000, so they are alike: one is Main and the others help.');
  assert.match(R.sentence(V.hw.two), /RTX 4070 Ti is Main: it is faster\. RTX 3060 helps\./);
});

test('the app page line (GpuRoles.tsx recommendationLine) and the Terminal use the same alike sentence', async () => {
  const tsx = await readFile(new URL('../src/components/GpuRoles.tsx', import.meta.url), 'utf8');
  const py = await readFile(new URL('../../../terminal/strata_control/gpu_roles.py', import.meta.url), 'utf8');
  const two = 'Both cards are {0}, so they are alike: one is Main and the other helps.';
  const many = 'All {0} cards are {1}, so they are alike: one is Main and the others help.';
  assert.ok(tsx.includes(`'${two}'`) && tsx.includes(`'${many}'`));
  assert.ok(py.includes('Both cards are {short_name(m)}, so they are alike: one is Main and the other helps.'));
  assert.ok(py.includes('All {n} cards are {short_name(m)}, so they are alike: one is Main and the others help.'));
  const en = JSON.parse(await readFile(new URL('../src/locales/en.json', import.meta.url), 'utf8'));
  assert.ok(Object.values(en).includes(two) && Object.values(en).includes(many), 'both templates are in the English catalog');
});

test('words: the sentence and the one-line summary', () => {
  assert.equal(R.sentence(V.hw.one), 'Strata will use your RTX 4070 Ti.');
  assert.match(R.sentence(V.hw.three), /RTX 4070 Ti is Main: it is faster\./);
  assert.match(R.sentence(V.hw.three), /RTX 2060 starts Off: 8 GB or less\./);
  assert.equal(R.describe({ 0: 'helper', 1: 'main' }, V.hw.two), 'Main: GPU 2 RTX 4070 Ti. Helper: GPU 1 RTX 3060.');
  assert.equal(R.noteText('old_main_helper:1', V.hw.two), 'RTX 4070 Ti is now a Helper.');
  assert.equal(R.noteText('only_main', V.hw.two), 'The only card that is on has to be Main.');
});
