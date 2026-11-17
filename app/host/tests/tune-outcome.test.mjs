/** Items 11, 13 and 15: one result layout for every outcome, and the tuned values in Expert mode. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const src = f => fileURLToPath(new URL(`../src/${f}`, import.meta.url));
async function load(file) {
  const result = await build({ entryPoints: [src(`lib/${file}`)], bundle: true, format: 'esm', platform: 'node', write: false, alias: { react: fileURLToPath(new URL('./helpers/react-stub.mjs', import.meta.url)) }, logLevel: 'silent', loader: { '.json': 'json' } });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}
globalThis.window ??= globalThis; globalThis.localStorage ??= { getItem: () => null, setItem: () => {} };
globalThis.navigator ??= { language: 'en' };
const { readOutcome, tokS, failureDetails } = await load('tune-outcome.ts');
const { tunedFields, describeChange } = await load('tune-values.ts');

test('every outcome the host reports lands on one card with a tone, an icon and its reason', () => {
  const o = (result, state = 'done') => readOutcome(result, state);
  const faster = o({ code: 'faster', values: { gain_pct: 18.4, base_tok_s: 41.2, tuned_tok_s: 48.6, preset_name: 'Fast · 64K' } });
  assert.deepEqual([faster.code, faster.tone, faster.icon, faster.gainPct, faster.baseTokS, faster.tunedTokS, faster.presetName], ['faster', 'ok', 'check', 18.4, 41.2, 48.6, 'Fast · 64K']);
  assert.deepEqual([o({ code: 'already_fastest', values: { base_tok_s: 41.2, minutes: 12 } }).tone, o({ code: 'already_fastest' }).icon], ['ok', 'check']);
  const busy = o({ code: 'busy_pc', reason_code: 'noisy', values: { noise_pct: 14 } });
  assert.deepEqual([busy.code, busy.tone, busy.icon, busy.reason, busy.noisePct], ['busy_pc', 'warn', 'clock', 'noisy', 14]);
  assert.equal(o({ code: 'busy_pc', reason_code: 'not_tried' }).reason, 'not_tried');
  const failed = o({ code: 'failed', reason_code: 'no_memory' });
  assert.deepEqual([failed.code, failed.tone, failed.icon, failed.reason], ['failed', 'bad', 'x', 'no_memory']);
  assert.equal(o({ code: 'stopped' }).tone, 'neutral');
  assert.equal(o(null, 'cancelled').code, 'stopped');
  assert.equal(o(null, 'error').code, 'failed', 'a job that ended in an error is a failed tune');
});

test('answers from a check of a saved tuning and from an older Control still get a card', () => {
  const o = (r, s = 'done') => readOutcome(r, s);
  assert.equal(o({ verdict: 'keep' }).code, 'still_fast'); assert.equal(o({ verdict: 'retune' }).code, 'slower'); assert.equal(o({ verdict: 'inconclusive' }).code, 'unsure');
  assert.equal(o({ outcome: 'improved', gain_pct: 12, base: { decode_tok_s: 40 }, chosen: { decode_tok_s: 45 } }).tunedTokS, 45);
  assert.equal(o({ outcome: 'default_confirmed' }).code, 'already_fastest'); assert.equal(o({ outcome: 'inconclusive' }).code, 'busy_pc'); assert.equal(o({ outcome: 'base_failed' }).code, 'failed');
  assert.equal(tokS(41.234), '41.2'); assert.equal(tokS(null), '—');
});

test('the busy-PC card says it in plain words, in the same card as every other result, with Try again', async () => {
  const card = await readFile(src('components/TuneRunView.tsx'), 'utf8');
  assert.match(card, /Your PC was busy during the test, so we couldn't measure reliably\. Nothing was changed\. Try again when other programs are closed\./);
  const dialog = await readFile(src('components/NativeTune.tsx'), 'utf8');
  assert.match(dialog, /<TuneOutcomeCard outcome=\{outcome\}/); assert.match(dialog, /'Try again'/);
  assert.ok(!/tune-orbit/.test(dialog), 'no second, different-looking result screen');
});

test('item 14 (changed by GUI feedback 2, item 3): the running step is calm - no glow, no beep, no spinning ring - and reduced motion is honoured', async () => {
  const css = await readFile(src('styles/app.css'), 'utf8');
  assert.ok(!/tune-glow|tune-beep/.test(css), 'the glow and the beep animations are gone from the running step');
  assert.ok(!/\.tune-step\[data-s="run"\] \.tune-step__badge::before/.test(css), 'the spinning ring is gone');
  assert.match(css, /prefers-reduced-motion: reduce\) \{[\s\S]*tune-step__state::before/, 'reduced motion switches the pulse off too');
});

test('item 11: Expert mode shows the values the tuning set', () => {
  const f = tunedFields([{ flag: '--kv', was: 'int8', now: 'k8v4' }, { flag: '--resident-experts', was: null, now: '' }, { flag: '--vram-reserve-mib', was: '1536', now: '2048' }, { flag: '--spec', was: '4', now: '6' }]);
  assert.deepEqual(f, { kv: 'k8v4', lowRam: 'resident', vramReserveMib: 2048 });
  assert.deepEqual(tunedFields([{ flag: '--mmap-experts', was: '', now: null }]), { lowRam: 'auto' });
  assert.deepEqual(tunedFields(undefined), {});
  assert.deepEqual(describeChange({ flag: '--pool-workers', was: '6', now: '8' }), { name: 'pool workers', now: '8', was: '6' });
});

test('#4805: a tuning that did not finish carries the reason the tuner gave and what this PC has in its Details', () => {
  const hardware = { ram_gb: 32, gpus: [{ name: 'NVIDIA GeForce RTX 4050 Laptop GPU', vram_gb: 6 }] };
  const text = failureDetails({ message: 'Your current settings could not be measured: the engine stopped: no VRAM is left for the expert cache', notes: ['The model was unloaded for the test.'] }, hardware);
  assert.match(text, /^Your current settings could not be measured: the engine stopped: no VRAM is left for the expert cache/);
  assert.match(text, /The model was unloaded for the test\./);
  assert.match(text, /Memory: 32 GB in this PC/);
  assert.match(text, /Graphics card: NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB/);
  assert.equal(failureDetails({ message: 'Not tuned: not enough free memory to start it safely (3.1 GiB free, 9.0 GiB needed)' }, null), 'Not tuned: not enough free memory to start it safely (3.1 GiB free, 9.0 GiB needed)');
  assert.equal(failureDetails({ summary: 'x' }, hardware), null, 'nothing to say: no empty Details');
  assert.equal(failureDetails(null, hardware), null);
  assert.equal(failureDetails({ message: 'a', notes: ['a'] }, null), 'a', 'the same sentence is not repeated');
});
