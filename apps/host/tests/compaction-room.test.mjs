// #5091: the context popover says how much room is left before the assistant summarises older messages by itself, from the host's own figure
// (GET /api/agent/compaction, model.thresholdTokens), not from a percentage of the page's own.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const load = async name => {
  const { code } = await transform(src(name), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
};
const LANGS = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
const LABEL = 'Room left before older messages are summarized';

test('room left = the host\'s summarising point minus what the chat holds, only for the window the chat runs with', async () => {
  const { roomBeforeSummary } = await load('lib/compaction-room.ts');
  const model = { context: 32768, thresholdTokens: 20071 };
  assert.equal(roomBeforeSummary(5000, 32768, model), 15071);
  assert.equal(roomBeforeSummary(20071, 32768, model), 0);
  assert.equal(roomBeforeSummary(25000, 32768, model), 0, 'past the point: nothing left, never negative');
  assert.equal(roomBeforeSummary(5000, 65536, model), null, 'another window (another model): no figure rather than a wrong one');
  assert.equal(roomBeforeSummary(null, 32768, model), null, 'no reading yet');
  assert.equal(roomBeforeSummary(5000, null, model), null);
  assert.equal(roomBeforeSummary(5000, 32768, null), null, 'the host named no model');
  assert.equal(roomBeforeSummary(5000, 32768, { context: 32768 }), null);
  assert.equal(roomBeforeSummary(5000, 32768, { context: 32768, thresholdTokens: 0 }), null);
});

test('the popover asks the host once per window and shows the row beside the space remaining', () => {
  const p = src('components/TelemetryPopover.tsx');
  assert.match(p, /api\.get<\{ model\?: CompactionModel \| null \}>\('\/api\/agent\/compaction'\)/);
  assert.match(p, /\}, \[m\.max\]\);\n[\s\S]*const remoteChat = session\?\.messages\.filter\(x => x\.role === 'assistant'\)\.at\(-1\)\?\.remoteModel === true;\n\s+const room = roomBeforeSummary\(m\.used, m\.max, remoteChat \? null : compaction\);/);
  assert.ok(p.includes(`<Row label={tr("Space remaining")}`));
  assert.ok(p.includes(`{tr(room !== null ? <Row label={tr("${LABEL}")} value={<Num value={room} format={compactTokens}/>}/> : null)}`));
  const hooks = p.indexOf('const room = roomBeforeSummary'), early = p.indexOf('if (!m.hasReplies)');
  assert.ok(hooks > 0 && early > hooks, 'the hooks run before the early return');
});

test('the label is in the catalog in all ten languages', () => {
  const cat = Object.fromEntries(LANGS.map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/guic-${l}.json`, import.meta.url), 'utf8'))]));
  const key = Object.keys(cat.en).find(k => cat.en[k] === LABEL);
  assert.ok(key, `${LABEL} is in guic-en.json`);
  for (const l of LANGS) assert.ok(cat[l][key] && cat[l][key].trim(), `${l} has it`);
  assert.equal(/[^  ][;:?!]| [;:?!]/.test(cat.fr[key]), false);
});
