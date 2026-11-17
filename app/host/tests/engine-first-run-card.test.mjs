// Packaged-app walkthrough (GUI-VIS-1117 A10), the card: an unreadable download list has its own words and one next step (no word of updates or Settings), and the offline and
// error cards can go on without an engine. The offline person sees the card that already existed once Control's words are recognised (tests/engine-first-run-offline.test.mjs).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('the card: offline keeps its card, an unreadable list has its own words and one next step, both can go on without an engine', () => {
  const card = readFileSync(new URL('../src/components/EngineFirstRun.tsx', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  const offline = card.slice(card.indexOf("else if (info.status === 'offline')"), card.indexOf("else if (info.status === 'error')"));
  const error = card.slice(card.indexOf("else if (info.status === 'error')"), card.indexOf("else if (info.status === 'wrong-make')"));
  assert.ok(offline.includes('Strata could not reach github.com.'));
  assert.ok(offline.includes('{withoutEngine}'));
  assert.ok(error.includes('Strata couldn’t read its download list, so it can’t pick an engine yet. Press Try again, or choose another way below.'));
  assert.ok(!/Settings|update list/.test(error.replace(/\/\/.*\n/g, '')), 'no word of Settings or the update list');
  assert.ok(error.includes('{withoutEngine}') && error.includes('onClick={look}') && error.includes('{pick("Choose a downloaded file…")}'));
  assert.ok(card.includes('const withoutEngine = <button type="button" className="onboard-continue-anyway" onClick={onDone}>{tr("Continue without an engine")}</button>;'));
});

test('the new line exists in all ten languages', () => {
  const LINE = 'Strata couldn’t read its download list, so it can’t pick an engine yet. Press Try again, or choose another way below.';
  const en = JSON.parse(readFileSync(new URL('../src/locales/guic-en.json', import.meta.url), 'utf8'));
  const key = Object.keys(en).find(k => en[k] === LINE);
  assert.ok(key, 'the English key');
  for (const l of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he']) {
    const text = JSON.parse(readFileSync(new URL(`../src/locales/guic-${l}.json`, import.meta.url), 'utf8'))[key];
    assert.ok(text && text !== LINE, l);
  }
});
