// Visual walkthrough (GUI-VIS-1117, finding 6): with no model loaded Send was greyed and nothing said why or what to do. A plain line under the box now does.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const chat = readFileSync(new URL('../src/screens/Chat.tsx', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const HINT = 'No model is loaded. Load one in Models to send this message.';
const LANGS = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];

test('the hint shows only for typed text or a picture, no model, no running answer, and not for the command menu or /time-skip', () => {
  const line = chat.split('\n').find(l => l.includes(HINT)) ?? '';
  assert.ok(line.includes("showNoModelHint({ streaming, canChat, hasText: !!draft.trim(), hasPhotos: photos.length > 0, bareSlash: draft.trim() === '/', isTimeSkip: timeSkipPrompt(draft) !== null })"), 'the shared condition (tests/remote-model-chip.test.mjs runs it)');
  assert.ok(line.includes('role="status"'), 'announced politely');
});

test('the same condition greys Send (the hint explains exactly that state)', () => {
  assert.ok(chat.includes('(!canChat && timeSkipPrompt(draft) === null)'));
});

test('the hint exists in all ten languages with no raw slot', () => {
  const en = JSON.parse(readFileSync(new URL('../src/locales/guic-en.json', import.meta.url), 'utf8'));
  const key = Object.keys(en).find(k => en[k] === HINT);
  assert.ok(key, 'the English key');
  for (const l of LANGS) {
    const text = JSON.parse(readFileSync(new URL(`../src/locales/guic-${l}.json`, import.meta.url), 'utf8'))[key];
    assert.ok(text && text.length > 10, l);
    if (l !== 'en') assert.notEqual(text, HINT, l);
  }
});
