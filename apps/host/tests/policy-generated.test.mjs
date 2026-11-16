/** The host's launch-policy tables come from strata_control (policy-generated.mjs, written by python -m strata_control.app_policy; tests/terminal/test_app_policy.py checks the file is current).
 *  Here: the host really uses them, and the screen's own copy of the family list (src/lib/drafting.ts) is the same list. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { SPEED_SWITCH_TABLE, DRAFTING_FAMILIES, DFLASH2_MODEL_PATTERN } from '../server/policy-generated.mjs';
import { transform } from 'esbuild';
import * as S from '../server/native-settings.mjs';

test('the speed switches of the settings are the generated table, with the Speed group added where the table names none', () => {
  assert.deepEqual(S.SPEED_SWITCHES.map(({ group, ...rest }) => rest), SPEED_SWITCH_TABLE.map(({ group, ...rest }) => rest));
  assert.ok(S.SPEED_SWITCHES.every(s => s.group === (SPEED_SWITCH_TABLE.find(t => t.id === s.id).group ?? 'Speed')));
  assert.deepEqual(SPEED_SWITCH_TABLE.map(s => s.id), ['faster_long_prompts', 'faster_12gb_nvidia', 'faster_radeon_9000']);
});

test('the drafting families and the model-name rule are the generated ones', () => {
  assert.deepEqual(S.DRAFTING_FAMILIES, DRAFTING_FAMILIES);
  const re = new RegExp(DFLASH2_MODEL_PATTERN, 'i');
  assert.ok(re.test('qwen3.8-flash-next-iq3_s') && re.test('qwen3.8-flash-next-unsloth-ud-q4_k_xl') && !re.test('swift-1.5'));
  assert.equal(S.resolveDrafter('auto', 'qwen3.8-flash-next-iq3_s', true, false), 'dflash2');
  assert.equal(S.resolveDrafter('auto', 'swift-1.5-iq3_s', true, false), 'mtp');
});

test('the screen copy of the drafting families (src/lib/drafting.ts) is the generated list', async () => {
  const ts = await readFile(new URL('../src/lib/drafting.ts', import.meta.url), 'utf8');
  const m = /export const DRAFTING_FAMILIES = (\[[^\]]*\]);/.exec(ts);
  assert.ok(m, 'DRAFTING_FAMILIES in src/lib/drafting.ts');
  assert.deepEqual(JSON.parse(m[1].replaceAll("'", '"')), DRAFTING_FAMILIES);
});

/* The rules (where a switch applies, which models have a Drafting choice) are code in the host and in the screen; strata_control answers the same cases (policy-vectors.json, written by python -m strata_control.app_policy). */
const V = JSON.parse(await readFile(new URL('./data/policy-vectors.json', import.meta.url), 'utf8'));
const { code } = await transform(await readFile(new URL('../src/lib/drafting.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const T = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('the host shows the speed switches strata_control shows, for every card, engine and model of the vectors', () => {
  assert.ok(V.switches.length > 100);
  for (const row of V.switches) {
    const model = V.models[row.model];
    const context = { backend: row.backend, gpus: [V.cards[row.card]], arch: model.arch ?? null, vision: !!model.vision };
    const shown = S.SPEED_SWITCHES.filter(s => S.speedSwitchApplies(s.id, context)).map(s => s.id).sort();
    assert.deepEqual(shown, row.shows, `${row.model} on ${row.backend} with ${row.card}`);
  }
});

test('the host and the screen agree with strata_control on which models and families have a Drafting choice', () => {
  for (const { name, supported } of V.names) assert.equal(S.resolveDrafter('dflash2', name, true, false) === 'dflash2', supported, name);
  for (const { family, supported } of V.families) {
    assert.equal(T.resolveDrafter('dflash2', family, true, false) === 'dflash2', supported, `screen: ${family}`);
    assert.equal(S.resolveDrafter('dflash2', family, true, false) === 'dflash2', supported, `host: ${family}`);
  }
});
