// SOAK-APP: a chat file saved without `updatedAt` blanked the whole chat page (TypeError on localeCompare in recent-chats.ts), and a missing title or message text threw on
// .trim(). A damaged or older chat file must never take the page down: the lists tolerate missing or odd fields and the chats are repaired where they are read.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { build, transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const load = async name => {
  const { code } = await transform(src(name), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
};
const { recentChats, recentChatDescription } = await load('lib/recent-chats.ts');
const { repairSession } = await load('lib/session-repair.ts');
// format.ts reads the language through ./i18n (the catalogs): bundled here with a one-line stand-in for it
const formatBundle = await build({ entryPoints: [new URL('../src/lib/format.ts', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')], bundle: true, write: false, format: 'esm', platform: 'node', plugins: [{ name: 'i18n-stub', setup(b) {
  b.onResolve({ filter: /^\.\/i18n$/ }, () => ({ path: 'i18n', namespace: 'stub' }));
  b.onLoad({ filter: /.*/, namespace: 'stub' }, () => ({ contents: 'export const tr = s => s; export const localeTag = () => "en"; export const currentLanguage = () => "en";', loader: 'js' }));
} }] });
const { relTimeShort } = await import(`data:text/javascript;base64,${Buffer.from(formatBundle.outputFiles[0].text).toString('base64')}`);

const good = (id, updatedAt) => ({ id, title: `Chat ${id}`, createdAt: updatedAt, updatedAt, workspace: 'C:\\Work', messages: [{ id: 'm1', role: 'user', content: `Hello from ${id}`, at: updatedAt }] });
const store = () => [
  good('a', '2026-10-10T10:00:00.000Z'),
  { id: 'no-date', title: 'No date', workspace: 'C:\\Work', messages: [{ id: 'm', role: 'user', content: 'x', at: '' }] },                  // no updatedAt, no createdAt
  { id: 'no-title', createdAt: '2026-10-09T10:00:00.000Z', updatedAt: '2026-10-09T10:00:00.000Z', messages: [] },                           // no title, no workspace
  { id: 'odd', title: 42, updatedAt: 1760000000000, workspace: null, messages: [null, 'junk', { role: 'user' }, { role: 'assistant', content: 7 }, { content: 'text only' }] },
  { id: 'no-messages', title: 'No messages', updatedAt: '2026-10-08T10:00:00.000Z' },
  good('b', '2026-10-07T10:00:00.000Z'),
];

test('the recent-chats list tolerates a chat without a date, title or message text, and lists every other chat', () => {
  const sessions = store();
  const before = structuredClone(sessions);
  assert.doesNotThrow(() => recentChats(sessions, 'a'));
  const ids = recentChats(sessions, 'a').map(s => s.id);
  assert.deepEqual(ids.slice(0, 2), ['a', 'no-title'], 'dated chats keep their activity order');
  assert.ok(ids.indexOf('b') > -1 && ids.indexOf('no-messages') > -1, 'the other chats are listed');
  const withMissing = recentChats([sessions[1], sessions[0], sessions[5]]);
  assert.deepEqual(withMissing.map(s => s.id), ['a', 'b', 'no-date'], 'a missing date sorts last');
  assert.deepEqual(sessions, before, 'the source list is left as it was');
  for (const s of sessions) assert.doesNotThrow(() => recentChatDescription(s), s.id);
  assert.equal(recentChatDescription(sessions[3]), 'No messages yet', 'odd messages are skipped; the last user message has no text');
  assert.equal(recentChatDescription({ messages: [null, { role: 'user', content: 5 }, { role: 'assistant', content: 'Welcome' }] }), 'No messages yet', 'a user message whose text is not a string reads as empty, never throws');
  assert.equal(recentChatDescription(sessions[4]), 'No messages yet');
  assert.equal(recentChatDescription({}), 'No messages yet');
});

test('a chat read from a saved file is repaired: every text field a list reads is a string, well-formed chats are untouched', () => {
  const [a, noDate, noTitle, odd, noMessages, b] = store().map(repairSession);
  assert.deepEqual(a, good('a', '2026-10-10T10:00:00.000Z'), 'a well-formed chat comes back as it was');
  assert.deepEqual(b, good('b', '2026-10-07T10:00:00.000Z'));
  for (const s of [a, noDate, noTitle, odd, noMessages, b]) {
    for (const field of ['id', 'title', 'createdAt', 'updatedAt', 'workspace']) assert.equal(typeof s[field], 'string', `${s.id}.${field}`);
    assert.ok(Array.isArray(s.messages));
    for (const m of s.messages) { assert.equal(typeof m.content, 'string'); assert.equal(typeof m.role, 'string'); assert.equal(typeof m.id, 'string'); }
  }
  assert.equal(noDate.updatedAt, '', 'no date stays empty (sorts last)');
  assert.equal(noTitle.title, 'New conversation'); assert.equal(noTitle.defaultTitle, true);
  assert.equal(noTitle.workspace, '');
  assert.equal(noTitle.updatedAt, '2026-10-09T10:00:00.000Z');
  assert.equal(odd.title, 'New conversation');
  assert.deepEqual(odd.messages.map(m => [m.role, m.content]), [['user', ''], ['assistant', ''], ['assistant', 'text only']], 'only objects are kept, their text made a string');
  assert.deepEqual(noMessages.messages, []);
  const order = recentChats([noDate, a, noTitle, odd, noMessages, b].map(s => s), 'a').map(s => s.id);
  assert.equal(order.length, 6, 'every chat is listed');
  assert.equal(repairSession(odd).updatedAt, odd.updatedAt, 'repairing twice changes nothing');
});

test('no usable date shows no time instead of "NaNm"', () => {
  assert.equal(relTimeShort(''), '');
  assert.equal(relTimeShort(undefined), '');
  assert.equal(relTimeShort('not a date'), '');
  assert.equal(relTimeShort(new Date(Date.now() - 5 * 60_000).toISOString()), '5m');
});

test('the store repairs every chat it reads (fixSession)', () => {
  const store = src('lib/store.ts');
  assert.match(store, /import \{ repairSession \} from '\.\/session-repair';/);
  assert.match(store, /function fixSession\(raw: Session\): Session \{\n  const session = repairSession\(raw\);/);
});
