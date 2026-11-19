// Lane GUI-TUI (0.2.17) QA: a picture of an earlier message that the browser no longer has must not stop every later message of the chat.
// Seen: after a picture chat, the same host opened in another browser profile answered every new message with
// "red-square.png is no longer in browser storage. Attach it again. Try sending the message again." and nothing could fix it.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const { code } = await transform(src('lib/history-pictures.ts'), { loader: 'ts', format: 'esm' });
const { readHistoryPictures, lostPicturesNote } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('a picture the browser still has is read, one it lost is named, and the rest goes on', async () => {
  const have = { id: 'a', name: 'cat.png' }, gone = { id: 'b', name: 'red-square.png' }, also = { id: 'c', name: 'dog.jpg' };
  const read = async p => { if (p.id === 'b') throw new Error('red-square.png is no longer in browser storage. Attach it again.'); return { url: 'data:image/png;base64,' + p.id }; };
  const { found, lost } = await readHistoryPictures([have, gone, also], read);
  assert.deepEqual(found.map(f => f.photo.name), ['cat.png', 'dog.jpg']);
  assert.deepEqual(found.map(f => f.url), ['data:image/png;base64,a', 'data:image/png;base64,c']);
  assert.deepEqual(lost, ['red-square.png']);
});

test('nothing lost: no note; lost pictures are named for the model', () => {
  assert.equal(lostPicturesNote([]), '');
  assert.match(lostPicturesNote(['a.png', 'b.png']), /\[Earlier pictures: a\.png, b\.png\. They are no longer stored in this browser/);
});

test('the chat send reads earlier pictures through readHistoryPictures and no longer throws for a missing one', () => {
  const store = src('lib/store.ts');
  assert.match(store, /readHistoryPictures\(included,photoData\)/);
  assert.doesNotMatch(store, /image_url:\{url:\(await photoData\(photo\)\)\.url\}/, 'the history must not read a picture with a call that throws and ends the send');
  assert.match(store, /Some earlier pictures are missing/);
});
