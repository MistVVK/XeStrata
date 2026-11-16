import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

const root = resolve(import.meta.dirname, '..');
const source = readFileSync(resolve(root, 'src/components/PrivacyNotes.tsx'), 'utf8');
const SENTENCES = ['Strata sends no usage telemetry.', "Conversations stay on this computer, in Strata's data folder.", 'When you choose an API model, that conversation and enabled skill instructions are sent to its provider.', 'The UI request log keeps client, timing and token metadata in memory.', 'Native API monitoring can retain request content in memory when enabled.', 'Connected tools access the resources you configure.'];

test('Privacy is one compact paragraph with all six sentences, in order', () => {
  const text = source.match(/const NOTES = "([^"]*)";/)?.[1];
  assert.equal(text, SENTENCES.join(' '));
  assert.equal((source.match(/<p>/g) ?? []).length, 1, 'one paragraph, not a line per sentence');
});

test('the paragraph is translated as a whole in all 9 languages', () => {
  const joined = SENTENCES.join(' ');
  const en = JSON.parse(readFileSync(resolve(root, 'src/locales/en.json'), 'utf8'));
  const id = Object.keys(en).find(k => en[k] === joined);
  assert.ok(id, 'the English catalog has the whole paragraph');
  for (const lang of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he']) {
    const cat = JSON.parse(readFileSync(resolve(root, `src/locales/${lang}.json`), 'utf8'));
    assert.ok(cat[id] && cat[id] !== joined && cat[id].length > 80, `${lang} has its own translation of the paragraph`);
  }
});
