/** The line shown while a download server has not answered yet (lane NET, W14): in all nine languages, the host and the seconds filled in from
 *  setup's English label (strata_control/netpolicy.py WORDS["waiting"], handed over as the install step's label). */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
async function bundle(entry) {
  const result = await build({
    entryPoints: [here(entry)], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent',
    alias: { react: here('./helpers/react-stub.mjs') }, loader: { '.json': 'json' },
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

const LINE = 'Waiting for {0} to answer ({1} s so far)';

test('the waiting line exists in all nine languages with both placeholders', () => {
  for (const lang of ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr']) {
    const file = JSON.parse(readFileSync(here(`../src/locales/net-${lang}.json`), 'utf8'));
    assert.deepEqual(Object.keys(file), ['net01'], lang);
    assert.ok(/\{0\}/.test(file.net01) && /\{1\}/.test(file.net01), `${lang} keeps {0} and {1}`);
    if (lang === 'en') assert.equal(file.net01, LINE, 'the approved wording');
    else assert.notEqual(file.net01, LINE, `${lang} is translated`);
  }
});

test('the first-run screen takes the waiting line in place of the stage text, only while setup waits', async () => {
  const eta = await bundle('../src/lib/download-eta.ts');
  const i18n = await bundle('../src/lib/i18n.ts');
  assert.equal(eta.waitingText({ phase: 'waiting', label: 'Waiting for github.com to answer (5 s so far)' }), 'Waiting for github.com to answer (5 s so far)');
  assert.equal(eta.waitingText({ phase: 'download', label: 'Getting the model files' }), undefined);
  assert.equal(eta.waitingText(undefined), undefined);
  const src = readFileSync(here('../src/components/GuidedSetup.tsx'), 'utf8');
  assert.match(src, /<strong className="onboard-download-status">\{tr\(waitingText\(installJob\?\.result\?\.install\) \?\? \(progress \? stageText/,
    'the same element as the stage text: no new element, no new style');
  void i18n;
});

test('setup\'s English label is translated with the host and the seconds kept', async () => {
  const i18n = await bundle('../src/lib/i18n.ts');
  await i18n.setTranslationLanguage('de');
  assert.equal(i18n.tr('Waiting for huggingface.co to answer (15 s so far)'), 'Warten auf eine Antwort von huggingface.co (bisher 15 s)');
  await i18n.setTranslationLanguage('en');
  assert.equal(i18n.tr('Waiting for huggingface.co to answer (15 s so far)'), 'Waiting for huggingface.co to answer (15 s so far)');
});
