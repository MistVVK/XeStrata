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

test('the chat screen asks parkedReady with the active chat and runs again when the active chat changes', () => {
  const chat = src('screens/Chat.tsx');
  assert.match(chat, /import \{ busyPlan, parkedReady \} from '\.\.\/lib\/while-busy';/);
  assert.match(chat, /if \(!session \|\| !parkedReady\(session\.id, activeId, streaming, parked\?\.length \?\? 0\)\)\s*return;\s*const next = takeParked\(session\.id\);\s*if \(next !== undefined\)\s*void sendMessage\(next\);\s*\}, \[streaming, parked\?\.length, session\?\.id, activeId\]\);/);
});
