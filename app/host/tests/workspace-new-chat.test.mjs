// #4913: a new chat could only be started from the one "New chat" button, in whichever workspace happened to be active. Each workspace row in the
// sidebar now has its own New chat button: it makes that workspace the active one (newSession starts a chat in the active workspace) and starts the chat.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const LANGS = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
const TEXT = 'New chat in {0}';

test('each workspace row has a New chat button that starts a chat in that workspace', () => {
  const side = src('app/Sidebar.tsx');
  const button = /<IconBtn name="new-chat" title=\{tr\("New chat in \{0\}", \[w\.name\]\)\} onClick=\{\(\) => \{(.*?) \}\}\/>/.exec(side);
  assert.ok(button, 'the workspace row has the button');
  const order = ['setActiveWorkspace(w.id)', "runCommand('new')"].map(s => button[1].indexOf(s));
  assert.ok(order[0] >= 0 && order[1] > order[0], 'the workspace is made active before the chat is started');
  assert.match(button[1], /setFolders\(f => \(\{ \.\.\.f, \[w\.id\]: true \}\)\)/, 'the workspace opens so the new chat shows');
});

test('three actions on a workspace row sit side by side, not on top of each other', () => {
  const css = src('styles/liquid.css').replace(/\r\n/g, '\n');
  const right = n => Number(new RegExp(`\\.ws__row > \\.st-btn--icon:nth-last-child\\(${n}\\) \\{\\n  right: (\\d+)px;`).exec(css)?.[1]);
  const spots = [2, right(2), right(3)];            // Remove (last), Rename, New chat; a button is 28px wide
  assert.deepEqual(spots, [2, 30, 58]);
  assert.ok(spots[1] - spots[0] >= 28 && spots[2] - spots[1] >= 28, 'each button clears the next');
});

test('the chat is started in the active workspace (the rule the button relies on)', () => {
  const store = src('lib/store.ts');
  assert.match(store, /workspace: state\.workspaces\.find\(w => w\.id === state\.activeWorkspace\)\?\.path \?\? state\.workspacePath/);
  assert.match(store, /export function setActiveWorkspace\(id: string\) \{\n  set\(\{ activeWorkspace: id \}\);/);
  assert.match(src('lib/commands.ts'), /id: 'new', label: 'New chat'[^\n]*run: \(\) => \{ const s = newSession\(\); go\('chat', s\.id\); \}/);
});

test('the button text is in the catalog in all ten languages, with the workspace name in it, German with the usual form', () => {
  const cat = Object.fromEntries(LANGS.map(l => [l, JSON.parse(readFileSync(new URL(`../src/locales/guic-${l}.json`, import.meta.url), 'utf8'))]));
  const key = Object.keys(cat.en).find(k => cat.en[k] === TEXT);
  assert.ok(key, `${TEXT} is in guic-en.json`);
  for (const l of LANGS) assert.ok(cat[l][key].includes('{0}'), `${l} keeps {0}`);
  assert.match(cat.de[key], /^Neuer Chat in \{0\}$/);
});
