// #5079: after the desktop shell loads the page again because its renderer died, the page starts with the sidebar previews closed, once.
// The shell side (which failures reload, the limit of two in two minutes, the address) is unit-tested in apps/desktop/src-tauri/src/recovery.rs.
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { transform } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const { code } = await transform(readFileSync(resolve(root, 'src/lib/recovery.ts'), 'utf8'), { loader: 'ts', format: 'esm' });
const lib = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

function fakeWindow(url) {
  const u = new URL(url);
  const calls = [];
  globalThis.window = {
    location: { pathname: u.pathname, search: u.search, hash: u.hash },
    history: { state: null, replaceState: (state, title, address) => { calls.push(address); } },
  };
  return calls;
}

test('the recovery number is read from the query, and only a single digit counts', () => {
  assert.equal(lib.parseRecovery('?recovery=1'), 1);
  assert.equal(lib.parseRecovery('?x=1&recovery=2'), 2);
  for (const bad of ['', '?', '?recovery=', '?recovery=0', '?recovery=10', '?recovery=x', '?recovery=-1', '?recovery=1.5', '?recoveryx=1'])
    assert.equal(lib.parseRecovery(bad), 0, bad);
});

test('the mark is taken out of the query and nothing else is', () => {
  assert.equal(lib.withoutRecovery('?recovery=1'), '');
  assert.equal(lib.withoutRecovery('?recovery=1&a=b'), '?a=b');
  assert.equal(lib.withoutRecovery('?a=b'), '?a=b');
  assert.equal(lib.withoutRecovery(''), '');
});

test('an ordinary start defers nothing and leaves the address alone', () => {
  const calls = fakeWindow('http://127.0.0.1:47831/#/chat/s_1');
  assert.equal(lib.startRecovery(), 0);
  assert.deepEqual(calls, []);
  assert.equal(lib.previewsDeferred('s_1'), false);
  assert.equal(lib.previewsDeferred(), false);
});

test('a recovery start defers the previews of the chat that was open, removes the mark and keeps the screen', () => {
  const calls = fakeWindow('http://127.0.0.1:47831/?recovery=1#/chat/s_1');
  assert.equal(lib.startRecovery(), 1);
  assert.deepEqual(calls, ['/#/chat/s_1'], 'the address loses the mark, keeps the hash');
  assert.equal(lib.previewsDeferred(), true);
  assert.equal(lib.previewsDeferred('s_1'), true);
  assert.equal(lib.previewsDeferred('s_1'), true, 'asked again for the same chat (the effects run more than once)');
  // another chat is an ordinary one, and the deferral is over for good
  assert.equal(lib.previewsDeferred('s_2'), false);
  assert.equal(lib.previewsDeferred('s_1'), false);
});

test('opening a preview ends the deferral; a second start in the same page is ordinary again', () => {
  fakeWindow('http://127.0.0.1:47831/?recovery=2');
  assert.equal(lib.startRecovery(), 2);
  assert.equal(lib.previewsDeferred('s_1'), true);
  lib.endPreviewDeferral();
  assert.equal(lib.previewsDeferred('s_1'), false);
  fakeWindow('http://127.0.0.1:47831/');
  assert.equal(lib.startRecovery(), 0);
  assert.equal(lib.previewsDeferred('s_1'), false);
});

test('no address to read is an ordinary start, not an error', () => {
  globalThis.window = undefined;
  assert.equal(lib.startRecovery(), 0);
});

test('the page reads the mark before the first render, and the chat honours it', () => {
  const main = readFileSync(resolve(root, 'src/main.tsx'), 'utf8');
  assert.ok(main.indexOf('startRecovery();') > 0 && main.indexOf('startRecovery();') < main.indexOf('createRoot(el).render'));
  const chat = readFileSync(resolve(root, 'src/screens/Chat.tsx'), 'utf8');
  assert.match(chat, /setSelectedPreview\(session && previewsDeferred\(session\.id\) \|\|/);
  assert.match(chat, /function openPreview\(item: PreviewArtifact\) \{\s*endPreviewDeferral\(\);/);
  assert.match(chat, /function selectPreview\(id: string\) \{\s*endPreviewDeferral\(\);/);
});

test('the shell hooks: Windows ProcessFailed, macOS web content process, the same recovery path, no new text', () => {
  const main = readFileSync(resolve(root, '../desktop/src-tauri/src/main.rs'), 'utf8');
  assert.match(main, /fn hook_renderer_failures\(window: &tauri::WebviewWindow, port: u16\)/);
  assert.match(main, /add_ProcessFailed\(&handler, &mut token\)/);
  assert.match(main, /on_web_content_process_terminate\(/);
  assert.match(main, /recovery::plan\(/);
  const cargo = readFileSync(resolve(root, '../desktop/src-tauri/Cargo.toml'), 'utf8');
  assert.match(cargo, /\[target\.'cfg\(windows\)'\.dependencies\][\s\S]*webview2-com = "0\.39"/);
});
