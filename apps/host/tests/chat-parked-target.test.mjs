// #5806: a message waiting to be sent in one chat (Send after / Interrupt) went into ANOTHER chat when the person switched chats.
// A click in the side bar makes the other chat the store's active one (selectSession) a moment before the page shows it; in that moment
// the chat on screen was still the first one, its answer read as finished (the store's streaming is the active chat's), so it sent its
// parked message, and sendMessage sends to the active chat: the other one.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const load = async name => {
  const { code } = await transform(src(name), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
};

test('a parked message is sent only from the chat that is also the active one, and only when it is quiet', async () => {
  const { parkedReady } = await load('lib/while-busy.ts');
  assert.equal(parkedReady('chat-1', 'chat-1', false, 1), true, 'the chat on screen is the active one and its answer is done');
  assert.equal(parkedReady('chat-1', 'chat-2', false, 1), false, 'the side bar already made chat 2 active: chat 1 must not send now');
  assert.equal(parkedReady('chat-1', 'chat-1', true, 1), false, 'still answering');
  assert.equal(parkedReady('chat-1', 'chat-1', false, 0), false, 'nothing parked');
  assert.equal(parkedReady(undefined, 'chat-1', false, 1), false);
  assert.equal(parkedReady('chat-1', null, false, 1), false, 'no active chat yet: sendMessage would start a new one');
});

test('two chats: a message parked in chat 1 goes out once, from chat 1, however the person switches and comes back (#5806)', async () => {
  const { parkedReady } = await load('lib/while-busy.ts');
  // The page's rule (Chat.tsx effect): the chat on screen sends its first parked message when parkedReady says so; the store then holds one less.
  const store = { parked: { c1: ['later'], c2: [] }, active: 'c1', streaming: true, sent: [] };
  const tick = viewed => {
    if (!parkedReady(viewed, store.active, store.streaming, store.parked[viewed]?.length ?? 0)) return;
    store.sent.push({ to: store.active, text: store.parked[viewed].shift() });
  };
  tick('c1');                                   // chat 1 is still answering
  store.active = 'c2'; tick('c1');              // the side bar made chat 2 active, chat 1 is still on screen
  store.streaming = false; tick('c1');          // chat 2 is quiet; chat 1's old view must not send into it
  tick('c2');                                   // chat 2 on screen: nothing parked there
  assert.deepEqual(store.sent, [], 'nothing crossed to the other chat');
  store.active = 'c1'; tick('c1');              // the person comes back to chat 1
  tick('c1');
  assert.deepEqual(store.sent, [{ to: 'c1', text: 'later' }], 'sent once, to the chat that parked it');
  // A chat that was deleted while its message waited: no view, no active chat of that id, nothing is sent.
  store.parked.c3 = ['orphan']; store.active = 'c1'; tick('c3'); tick(undefined); tick(null);
  assert.equal(store.sent.length, 1);
});

test('the chat screen asks parkedReady with the active chat and runs again when the active chat changes', () => {
  const chat = src('screens/Chat.tsx');
  assert.match(chat, /import \{ busyPlan, parkedReady \} from '\.\.\/lib\/while-busy';/);
  assert.match(chat, /if \(!session \|\| !parkedReady\(session\.id, activeId, streaming, parked\?\.length \?\? 0\)\)\s*return;\s*const next = takeParked\(session\.id\);\s*if \(next !== undefined\)\s*void sendMessage\(next\);\s*\}, \[streaming, parked\?\.length, session\?\.id, activeId\]\);/);
});
