// Every language the app offers is a language of the desktop shell too: the host's shell prefs (server/shell-prefs.mjs) and the Tauri
// shell (apps/desktop/src-tauri/src/shell.rs: LANGUAGES and the tray menu's own words). French was missing from both, so a person who
// chose Français kept the tray of the language before (or English), and the shell prefs refused the choice.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { LANGUAGES as SHELL } from '../server/shell-prefs.mjs';

const read = rel => readFileSync(new URL(rel, import.meta.url), 'utf8');
const app = [...read('../src/lib/i18n.ts').match(/export const LANGUAGES=\[(.*?)\] as const;/s)[1].matchAll(/value:'([a-z]{2})'/g)].map(m => m[1]);
const rs = read('../../desktop/src-tauri/src/shell.rs');
const rsLanguages = [...rs.match(/pub const LANGUAGES: \[&str; \d+\] = \[(.*?)\];/s)[1].matchAll(/"([a-z]{2})"/g)].map(m => m[1]);
const trayArms = [...rs.match(/pub fn tray_labels[\s\S]*?\n}\n/)[0].matchAll(/^\s*"([a-z]{2})" => \(/gm)].map(m => m[1]);

test('the app\'s languages, the host\'s shell prefs and the Tauri shell agree', () => {
  assert.ok(app.length >= 9 && app.includes('fr'), `the app's list (${app})`);
  assert.deepEqual([...SHELL].sort(), [...app].sort(), 'server/shell-prefs.mjs');
  assert.deepEqual([...rsLanguages].sort(), [...app].sort(), 'shell.rs LANGUAGES');
  assert.match(rs, new RegExp(`pub const LANGUAGES: \\[&str; ${rsLanguages.length}\\]`), 'the array length matches');
});

test('every language but English has its own tray words', () => {
  assert.deepEqual([...trayArms].sort(), app.filter(l => l !== 'en').sort());
});
