// Packaged-app walkthrough (GUI-VIS-1117 A15 / A16): after "Only connect to Strata on another computer" and a custom model API (Fetch, tick, Apply), the sidebar chip
// stayed "No model loaded / View model versions" although the model was picked in the header and chats worked. The chip, the placeholder and the "No model is loaded"
// line (b60358f27) must all count a picked API model as a model to chat with.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const { code } = await transform(src('lib/composer-model.ts'), { loader: 'ts', format: 'esm' });
const { chosenRemote, hasModelToChat, showNoModelHint } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

const providers = { catalog: [], connections: [{ id: 'c1', kind: 'custom', provider: null, name: 'My API', baseUrl: 'http://10.0.0.5:8080/v1', protocol: 'openai-chat', apiKeySet: true, models: [{ id: 'big', name: 'Big model' }, { id: 'small', name: 'Small model' }] }] };
const typed = { streaming: false, hasText: true, hasPhotos: false, bareSlash: false, isTimeSkip: false };

test('a picked API model is found with its connection; a missing pick is not', () => {
  const found = chosenRemote(providers, { connectionId: 'c1', modelId: 'big' });
  assert.equal(found.model.name, 'Big model');
  assert.equal(found.connection.name, 'My API');
  assert.equal(chosenRemote(providers, null), null);
  assert.equal(chosenRemote(null, { connectionId: 'c1', modelId: 'big' }), null);
  assert.equal(chosenRemote(providers, { connectionId: 'gone', modelId: 'big' }), null, 'a removed connection');
  assert.equal(chosenRemote(providers, { connectionId: 'c1', modelId: 'gone' }), null, 'a removed model');
});

test('with an API model picked and NO local model there is a model to chat with, and the "No model is loaded" line is not shown', () => {
  const remote = chosenRemote(providers, { connectionId: 'c1', modelId: 'big' });
  const canChat = hasModelToChat({ remote, localLoaded: null, nativeReady: false });
  assert.equal(canChat, true);
  assert.equal(showNoModelHint({ ...typed, canChat }), false);
  assert.equal(showNoModelHint({ ...typed, canChat, hasText: false, hasPhotos: true }), false, 'a picture, too');
});

test('with nothing picked and nothing loaded the line is shown for typed text or a picture, never for "/", /time-skip or a running answer', () => {
  const canChat = hasModelToChat({ remote: null, localLoaded: null, nativeReady: false });
  assert.equal(canChat, false);
  assert.equal(showNoModelHint({ ...typed, canChat }), true);
  assert.equal(showNoModelHint({ ...typed, canChat, hasText: false, hasPhotos: true }), true);
  assert.equal(showNoModelHint({ ...typed, canChat, hasText: false }), false, 'nothing typed');
  assert.equal(showNoModelHint({ ...typed, canChat, bareSlash: true }), false);
  assert.equal(showNoModelHint({ ...typed, canChat, isTimeSkip: true }), false);
  assert.equal(showNoModelHint({ ...typed, canChat, streaming: true }), false);
});

test('a local model or a ready native instance alone is also a model to chat with', () => {
  assert.equal(hasModelToChat({ remote: null, localLoaded: { id: 'x' }, nativeReady: false }), true);
  assert.equal(hasModelToChat({ remote: null, localLoaded: null, nativeReady: true }), true);
});

test('the composer and the sidebar chip use the shared helpers', () => {
  const chat = src('screens/Chat.tsx'), side = src('app/Sidebar.tsx');
  assert.ok(chat.includes('hasModelToChat({ remote: chosenRemote(remoteProviders, remotePick), localLoaded: loaded,'), 'the composer asks the picked API model itself');
  assert.ok(chat.includes('showNoModelHint({ streaming, canChat,'), 'the line uses the helper');
  assert.ok(side.includes('const remoteChosen = !current && !nativeCurrent ? chosenRemote(remoteProviders, remoteChoice) : null;'));
  assert.ok(side.includes('remoteChosen?.model.name ?? tr(\'No model loaded\')'), 'the chip names the picked model');
  assert.ok(side.includes('remoteChosen?.connection.name ?? tr(\'View model versions\')'), 'and the connection under it');
  assert.ok(side.includes("!current && !nativeCurrent && !remoteChosen ? 'No model loaded'"), 'the status is not "No model loaded" then');
});
