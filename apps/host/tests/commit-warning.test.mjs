// #1607: on Windows the engine warns (in its log) that the commit limit (memory plus page file) is close and it may stop without a message. serve/commit_limit.py puts the
// numbers in /metrics engine.commit_warning; the loaded model's card says it in plain words with one next step.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const { code } = await transform(src('lib/commit-warning.ts'), { loader: 'ts', format: 'esm' });
const { commitWarning } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const LANGS = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
const NOTICE = 'Windows is close to its memory limit (memory plus page file). Strata may stop without a message. Make the Windows page file larger, or choose a shorter context.';
const LEFT = '{0} GB of {1} GB are left.';

test('the card reads the engine\'s warning from /metrics engine.commit_warning, with one decimal; no warning, no notice', () => {
  assert.deepEqual(commitWarning({ commit_warning: { limit_gib: 92.8, left_gib: 0.8, planned_gib: 2, floor_gib: 4 } }), { left: '0.8', limit: '92.8' });
  assert.deepEqual(commitWarning({ commit_warning: { limit_gib: 31.94, left_gib: 1.46 } }), { left: '1.5', limit: '31.9' });
  for (const engine of [undefined, null, {}, { commit_warning: null }, { commit_warning: {} }, { commit_warning: { limit_gib: 0, left_gib: 0 } },
    { commit_warning: { limit_gib: 'x', left_gib: 1 } }, { commit_warning: { limit_gib: 10, left_gib: -1 } }, { commit_warning: 5 }, 'text', 7])
    assert.equal(commitWarning(engine), null, JSON.stringify(engine));
});

test('the model card shows the notice for the loaded model only, with the numbers line', () => {
  const card = src('components/NativeModels.tsx');
  assert.match(card, /import \{ commitWarning \} from '\.\.\/lib\/commit-warning';/);
  assert.ok(card.includes('const loadedCommit = commitWarning(loadedSelected?.metrics?.engine);'));
  assert.ok(card.includes(`{tr(loadedCommit ? <div className="native-commit-note" role="status"><p className="small"><Icon name="warning" size={16}/>{tr("${NOTICE}")}</p><p className="small muted">{tr("${LEFT}", [loadedCommit.left, loadedCommit.limit])}</p></div> : null)}`));
});

test('both texts are in the catalog in all ten languages with the language rules', () => {
  const cat = Object.fromEntries(LANGS.map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/guic-${l}.json`, import.meta.url), 'utf8'))]));
  for (const text of [NOTICE, LEFT]) {
    const key = Object.keys(cat.en).find(k => cat.en[k] === text);
    assert.ok(key, `${text} is in guic-en.json`);
    for (const l of LANGS) assert.ok(cat[l][key] && cat[l][key].trim(), `${l}: ${key}`);
    assert.equal(/[^  ][;:?!]| [;:?!]/.test(cat.fr[key]), false, 'French: ; : ? ! only after a no-break space');
  }
  const left = Object.keys(cat.en).find(k => cat.en[k] === LEFT);
  for (const l of LANGS) assert.ok(cat[l][left].includes('{0}') && cat[l][left].includes('{1}'), `${l} keeps both numbers`);
  const notice = Object.keys(cat.en).find(k => cat.en[k] === NOTICE);
  assert.match(cat.de[notice], /Vergrößern Sie/, 'German uses Sie');
});
