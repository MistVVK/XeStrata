// The sidebar preview kept a file's FIRST content after the file changed on disk (QAGUI, 4d482e627). Three places had to change together:
// openPreview dropped the freshly read file, the panel's loading effect had no content version, and the effect that re-adds a message's own
// artifacts put the stored first copy back over the newer one. A fix of the first two alone ran the renderer out of memory: a ping-pong
// between the message copy and the opened copy. These tests pin the rules (lib/preview-merge.ts) and that nothing can loop.
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, writeFileSync, mkdtempSync, utimesSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { build } from 'esbuild';
import { createRequire } from 'node:module';

const root = resolve(import.meta.dirname, '..');
const bundle = await build({ stdin: { contents: `export * from './src/lib/preview-merge.ts'; export { textRevision } from './src/lib/preview-render.ts';`, resolveDir: root, loader: 'ts' }, bundle: true, write: false, platform: 'node', format: 'cjs' });
const mod = { exports: {} };
new Function('require', 'module', 'exports', bundle.outputFiles[0].text)(createRequire(import.meta.url), mod, mod.exports);
const { previewVersion, openItem, syncArtifacts, messageArtifacts } = mod.exports;
const { previewFile } = await import('../server/chat-files.mjs');

const md = (content, at, extra = {}) => ({ id: 'file:/w/notes.md', kind: 'markdown', title: 'notes.md', content, at, ...extra });
const T = (n) => new Date(Date.UTC(2026, 10, 4, 10, 0, n)).toISOString();

test('the version follows the content: the text, else the host version, else the address', () => {
  assert.equal(previewVersion(md('# One')), previewVersion(md('# One')));
  assert.notEqual(previewVersion(md('# One')), previewVersion(md('# Two')));
  assert.notEqual(previewVersion({ url: '/a.png', version: '1:5' }), previewVersion({ url: '/a.png', version: '2:5' }));
  assert.equal(previewVersion({ url: '/a.png', version: '1:5' }), previewVersion({ url: '/a.png', version: '1:5' }));
  assert.notEqual(previewVersion({ url: '/a.png' }), previewVersion({ url: '/b.png' }));
  assert.notEqual(previewVersion({ content: '' }), previewVersion({ url: '' }));
});

test('opening a file that changed replaces its tab in place (it used to be dropped)', () => {
  const old = [md('# Version One', 1000), { id: 'other', kind: 'code', title: 'x.ts', content: 'x' }];
  const next = openItem(old, md('# Version Two', 2000));
  assert.equal(next.length, 2);
  assert.equal(next[0].content, '# Version Two');
  assert.equal(next[1].id, 'other', 'its place in the strip is kept');
  assert.equal(old[0].content, '# Version One', 'the old array is not mutated');
});

test('opening a file that did not change, or a new file, behaves as before; the same array comes back when nothing changed', () => {
  const old = [md('# Version One', 1000)];
  assert.equal(openItem(old, md('# Version One', 5000)), old);
  const added = openItem(old, { id: 'file:/w/b.md', kind: 'markdown', title: 'b.md', content: 'b' });
  assert.equal(added.length, 2);
  assert.equal(added[0], old[0]);
});

test('the QAGUI case: the message still carries Version One, the file was opened as Version Two, and Version Two stays', () => {
  const messages = [{ at: T(0), artifacts: [md('# Version One')] }];
  let tabs = syncArtifacts([], messageArtifacts(messages), []);          // the first message arrives: the tab shows Version One
  assert.equal(tabs[0].content, '# Version One');
  tabs = openItem(tabs, md('# Version Two', Date.parse(T(60))));         // "Open in sidebar" reads the file again
  assert.equal(tabs[0].content, '# Version Two');
  for (let i = 0; i < 5; i++) {                                           // a later message count change runs the sync again
    const again = syncArtifacts(tabs, messageArtifacts(messages), []);
    assert.equal(again, tabs, 'the same array: nothing to update');
    assert.equal(again[0].content, '# Version Two', 'the message copy did not come back');
  }
});

test('a message newer than the copy on show, with other content, does replace it; an older or equal one never does', () => {
  const shown = [md('# Version Two', Date.parse(T(60)))];
  const older = syncArtifacts(shown, messageArtifacts([{ at: T(30), artifacts: [md('# Version One')] }]), []);
  assert.equal(older, shown);
  const equal = syncArtifacts(shown, messageArtifacts([{ at: T(60), artifacts: [md('# Version Three')] }]), []);
  assert.equal(equal, shown);
  const newer = syncArtifacts(shown, messageArtifacts([{ at: T(90), artifacts: [md('# Version Three')] }]), []);
  assert.equal(newer[0].content, '# Version Three');
  assert.equal(shown[0].content, '# Version Two');
  // the same content in a newer message is not a change
  assert.equal(syncArtifacts(shown, messageArtifacts([{ at: T(90), artifacts: [md('# Version Two')] }]), []), shown);
});

test('a missing tab is added, a closed one is removed and not brought back', () => {
  const items = messageArtifacts([{ at: T(0), artifacts: [md('a'), { id: 'b', kind: 'code', title: 'b', content: 'b' }] }]);
  const added = syncArtifacts([], items, []);
  assert.deepEqual(added.map(a => a.id), ['file:/w/notes.md', 'b']);
  const closed = syncArtifacts(added, items, ['b']);
  assert.deepEqual(closed.map(a => a.id), ['file:/w/notes.md']);
  assert.deepEqual(syncArtifacts(closed, items, ['b']).map(a => a.id), ['file:/w/notes.md'], 'and it stays closed');
  assert.equal(syncArtifacts(closed, items, ['b']), closed);
  // a tab opened by hand that no message knows stays
  const mine = [{ id: 'mine', kind: 'code', title: 'm', content: 'm' }];
  assert.equal(syncArtifacts(mine, [], []), mine);
});

test('message artifacts: stamped with their message time, the last of an id wins, a bad time is left unstamped', () => {
  const list = messageArtifacts([{ at: T(0), artifacts: [md('one')] }, { at: T(30), artifacts: [md('two')] }, { at: 'not a time', artifacts: [{ id: 'z', kind: 'code', title: 'z', content: 'z' }] }, {}]);
  assert.equal(list.length, 2);
  assert.equal(list[0].content, 'two');
  assert.equal(list[0].at, Date.parse(T(30)));
  assert.equal(list[1].at, undefined);
  assert.deepEqual(messageArtifacts(undefined), []);
});

test('no loop: the open and the sync handed to each other 2000 times change the tabs only when the content changes', () => {
  const messages = [{ at: T(0), artifacts: [md('# Version One')] }];
  let tabs = [], changes = 0, loads = 0, lastKey = '';
  const set = next => { if (next !== tabs) changes++; tabs = next; };
  // what the panel does: reload its text only when (id, version) of the shown tab changes
  const panel = () => { const cur = tabs[0]; const key = cur ? cur.id + '|' + previewVersion(cur) : ''; if (key !== lastKey) { loads++; lastKey = key; } };
  set(syncArtifacts(tabs, messageArtifacts(messages), [])); panel();
  for (let i = 0; i < 2000; i++) {
    // the chat renders for every streamed token; the sync runs when the artifact count changes, the open when a run finishes
    set(openItem(tabs, md('# Version Two', Date.parse(T(60)))));
    set(syncArtifacts(tabs, messageArtifacts(messages), []));
    set(openItem(tabs, md('# Version Two', Date.parse(T(60 + i)))));
    panel();
  }
  assert.equal(changes, 2, 'the first tab, then Version Two once');
  assert.equal(loads, 2, 'the panel loaded its text twice: Version One, then Version Two');
});

test('no loop: a file that changes every second is one load per change, not one per render', () => {
  let tabs = [], loads = 0, lastKey = '';
  for (let s = 0; s < 300; s++) {
    for (let render = 0; render < 20; render++) {                           // 20 renders per second of streamed tokens
      if (render === 0) tabs = openItem(tabs, md('# v' + s, 1000 + s));
      const cur = tabs[0]; const key = cur.id + '|' + previewVersion(cur);
      if (key !== lastKey) { loads++; lastKey = key; }
    }
  }
  assert.equal(loads, 300);
});

test('the host sends the file version: it moves when the file is rewritten', () => {
  const dir = mkdtempSync(join(tmpdir(), 'strata-prev-'));
  writeFileSync(join(dir, 'n.md'), '# Version One\n');
  const a = previewFile(dir, 'n.md', 's');
  writeFileSync(join(dir, 'n.md'), '# Version Two\n');
  utimesSync(join(dir, 'n.md'), new Date(Date.now() + 5000), new Date(Date.now() + 5000));
  const b = previewFile(dir, 'n.md', 's');
  assert.equal(typeof a.mtime, 'number');
  assert.notEqual(a.mtime, b.mtime);
  assert.equal(b.text, '# Version Two\n');
});

test('the three places are changed together', () => {
  const chat = readFileSync(resolve(root, 'src/screens/Chat.tsx'), 'utf8');
  assert.match(chat, /setPreviews\(old => openItem\(old, item\)\)/);
  assert.match(chat, /setPreviews\(old => syncArtifacts\(old, items, session\?\.closedPreviews\)\)/);
  assert.doesNotMatch(chat, /old\.some\(a => a\.id === item\.id\) \? old/);
  assert.doesNotMatch(chat, /\[\.\.\.old\.filter\(a => !items\.some/);
  const panel = readFileSync(resolve(root, 'src/components/ArtifactPanel.tsx'), 'utf8');
  assert.match(panel, /\}, \[current\?\.id, version, revision\]\);/);
  assert.match(panel, /const version = useMemo\(\(\) => current \? previewVersion\(current\) : '', \[current\]\)/);
  // the reading view is reset by the file, not by its content
  assert.match(panel, /setMode\('preview'\);[\s\S]*?\}, \[current\?\.id\]\);/);
  const deliver = readFileSync(resolve(root, 'src/lib/deliverables.ts'), 'utf8');
  assert.match(deliver, /version = a\.mtime !== undefined/);
});
