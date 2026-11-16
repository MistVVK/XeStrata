/** Chats as files in the data folder: big chats, restart, a full disk, a refused write, a crash in the middle of a write, two windows. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, readdirSync, readFileSync, writeFileSync, statSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createChatStore, realIo, validChatId, storeFailure } from '../server/chat-store.mjs';

const scratch = () => mkdtempSync(join(tmpdir(), 'strata-chats-'));
const chat = (id, n, len, extra = {}) => ({
  id, title: `Chat ${id}`, createdAt: `2026-11-0${(id.charCodeAt(0) % 8) + 1}T10:00:00.000Z`, updatedAt: `2026-11-04T10:00:0${n % 10}.000Z`, workspace: 'C:/w', tokensIn: 0, tokensOut: 0,
  messages: Array.from({ length: n }, (_, i) => ({ id: `${id}-m${i}`, role: i % 2 ? 'assistant' : 'user', content: `${id}:${i}:` + String.fromCharCode(97 + (i % 26)).repeat(len) })),
  ...extra,
});
const text = (session) => JSON.stringify(session);
const readChat = async (store, id) => JSON.parse(await store.read(id));
const files = (dir) => readdirSync(join(dir, 'chats')).sort();

test('ids: only plain names, nothing that climbs out of the folder or names a device or a reserved file', () => {
  for (const ok of ['s_ab12cd34', 'chat-1', 'legacy_0', 'a.b']) assert.ok(validChatId(ok), ok);
  for (const bad of ['', '..', '../x', 'a/b', 'a\\b', '.hidden', 'con', 'NUL', 'index', 'import', 'conflicts', 'x'.repeat(200), null, 5]) assert.ok(!validChatId(bad), String(bad));
});

test('more than 10 MB of chats: every chat is saved whole, listed, and read back byte for byte after a restart (twice)', async () => {
  const dir = scratch();
  try {
    const chats = ['a', 'b', 'c', 'd'].map((id) => chat(id, 12, 140_000));   // 4 x 1.7 MB ... plus one very large one
    chats.push({ ...chat('e', 4, 10), messages: [{ id: 'e-0', role: 'assistant', content: 'q'.repeat(4_000_000), tools: [{ name: 'read', result: 'y'.repeat(3_000_000) }] }] });
    const total = chats.reduce((n, c) => n + text(c).length, 0);
    assert.ok(total > 10_000_000, `fixture is ${total} characters`);
    let store = createChatStore({ dir });
    for (const c of chats) await store.save(c.id, text(c));
    for (let restart = 1; restart <= 2; restart++) {
      store = createChatStore({ dir });   // a new host process
      assert.deepEqual(store.list().chats.map((c) => c.id).sort(), ['a', 'b', 'c', 'd', 'e']);
      for (const c of chats) {
        const back = await readChat(store, c.id);
        assert.equal(JSON.stringify(back.session), text(c), `chat ${c.id} after restart ${restart}`);
        await store.save(c.id, JSON.stringify(back.session));   // saved again unchanged
      }
    }
    assert.equal((await readChat(store, 'a')).rev, 3, 'rev counts the saves');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('the files: one per chat, a version field, an index, no stray temporary files', async () => {
  const dir = scratch();
  try {
    const store = createChatStore({ dir });
    await store.save('s_one', text(chat('s_one', 2, 10)));
    await store.save('s_two', text(chat('s_two', 2, 10)));
    assert.deepEqual(files(dir), ['index.json', 's_one.json', 's_two.json']);
    const file = JSON.parse(readFileSync(join(dir, 'chats', 's_one.json'), 'utf8'));
    assert.deepEqual([file.v, file.rev, file.session.id], [1, 1, 's_one']);
    const index = JSON.parse(readFileSync(join(dir, 'chats', 'index.json'), 'utf8'));
    assert.equal(index.v, 1); assert.deepEqual(index.chats.map((c) => c.id).sort(), ['s_one', 's_two']);
    if (process.platform !== 'win32') {
      assert.equal(statSync(join(dir, 'chats')).mode & 0o777, 0o700, 'the folder is owner-only');
      assert.equal(statSync(join(dir, 'chats', 's_one.json')).mode & 0o777, 0o600, 'the chat file is owner-only');
      assert.equal(statSync(join(dir, 'chats', 'index.json')).mode & 0o777, 0o600);
    }
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a crash in the middle of a write loses nothing older: the chat keeps its last whole version, the half file is removed at the next start', async () => {
  const dir = scratch();
  try {
    let store = createChatStore({ dir });
    const first = chat('s_x', 6, 5_000);
    await store.save('s_x', text(first));
    const bigger = chat('s_x', 12, 5_000);
    // The process dies after half of the new text reached the temporary file: the write throws and nothing after it (cleanup) runs.
    const crashing = createChatStore({ dir, io: { ...realIo,
      async writeFile(path, data) { await realIo.writeFile(path, data.slice(0, Math.floor(data.length / 2))); throw Object.assign(new Error('killed'), { code: 'EIO' }); },
      async unlink() { throw new Error('process is gone'); } } });
    await assert.rejects(crashing.save('s_x', text(bigger)));
    assert.ok(files(dir).some((n) => /\.tmp-/.test(n)), 'the crash left its temporary file');
    store = createChatStore({ dir });   // restart
    assert.ok(!files(dir).some((n) => /\.tmp-/.test(n)), 'the half-written file is removed at start');
    const back = await readChat(store, 's_x');
    assert.equal(back.rev, 1);
    assert.equal(JSON.stringify(back.session), text(first), 'the older version is whole');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a full disk: the save is refused with a disk-full error, the older version stays, and the next save works when there is room', async () => {
  const dir = scratch();
  try {
    let full = false;
    const store = createChatStore({ dir, io: { ...realIo, async writeFile(path, data) { if (full) throw Object.assign(new Error('no space left on device'), { code: 'ENOSPC' }); return realIo.writeFile(path, data); } } });
    const v1 = chat('s_d', 3, 100);
    await store.save('s_d', text(v1));
    full = true;
    await assert.rejects(store.save('s_d', text(chat('s_d', 9, 100))), (e) => e.code === 'disk-full' && e.status === 507);
    assert.equal(JSON.stringify((await readChat(store, 's_d')).session), text(v1));
    assert.ok(!files(dir).some((n) => /\.tmp-/.test(n)), 'no half file stays');
    full = false;
    const v3 = chat('s_d', 9, 100);
    assert.equal((await store.save('s_d', text(v3))).rev, 2, 'the failed save did not use a revision');
    assert.equal(JSON.stringify((await readChat(store, 's_d')).session), text(v3));
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a refused write (permission) is reported as such and changes nothing', async () => {
  const dir = scratch();
  try {
    const store = createChatStore({ dir, io: { ...realIo, async writeFile() { throw Object.assign(new Error('access denied'), { code: 'EACCES' }); } } });
    await assert.rejects(store.save('s_p', text(chat('s_p', 1, 1))), (e) => e.code === 'permission' && e.status === 403);
    assert.equal(store.list().chats.length, 0);
    assert.equal(storeFailure(Object.assign(new Error('x'), { code: 'EPERM' })).code, 'permission');
    assert.equal(storeFailure(Object.assign(new Error('x'), { code: 'ENOSPC' })).code, 'disk-full');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('two windows: each saves its own chats and neither removes the other\'s; the same chat saved from an older copy keeps the displaced version', async () => {
  const dir = scratch();
  try {
    const store = createChatStore({ dir });
    // window A and window B, same host
    const a1 = await store.save('s_a', text(chat('s_a', 2, 10)), { baseRev: null });
    const b1 = await store.save('s_b', text(chat('s_b', 2, 10)), { baseRev: null });
    assert.deepEqual(store.list().chats.map((c) => c.id).sort(), ['s_a', 's_b']);
    // both windows loaded s_a at rev 1; A saves a longer chat, then B saves from its older copy
    const fromA = chat('s_a', 6, 10), fromB = chat('s_a', 4, 10, { title: 'edited in B' });
    const savedA = await store.save('s_a', text(fromA), { baseRev: a1.rev });
    assert.equal(savedA.conflict, false);
    const savedB = await store.save('s_a', text(fromB), { baseRev: a1.rev });   // B still thinks rev 1
    assert.equal(savedB.conflict, true);
    assert.equal(JSON.stringify((await readChat(store, 's_a')).session), text(fromB), 'the later save is the chat');
    const kept = readdirSync(join(dir, 'chats', 'conflicts'));
    assert.equal(kept.length, 1);
    assert.equal(JSON.parse(readFileSync(join(dir, 'chats', 'conflicts', kept[0]), 'utf8')).session.messages.length, 6, 'window A\'s version is kept');
    assert.ok(b1.rev === 1 && (await readChat(store, 's_b')).rev === 1, 'the other chat is untouched');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('saves of one host never overlap, in any order they arrive', async () => {
  const dir = scratch();
  try {
    const store = createChatStore({ dir });
    const done = await Promise.all(Array.from({ length: 30 }, (_, i) => store.save(`s_${i % 3}`, text(chat(`s_${i % 3}`, 1 + (i % 7), 50)))));
    assert.deepEqual(done.map((d) => d.rev).sort((a, b) => a - b), Array.from({ length: 30 }, (_, i) => Math.floor(i / 3) + 1).sort((a, b) => a - b));
    assert.equal(store.list().chats.length, 3);
    for (const c of store.list().chats) JSON.parse(await store.read(c.id));   // every file is whole JSON
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('the index is rebuilt from the files when it is lost, damaged or behind; an unreadable chat file is reported and left alone', async () => {
  const dir = scratch();
  try {
    const store = createChatStore({ dir });
    await store.save('s_1', text(chat('s_1', 2, 10)));
    await store.save('s_2', text(chat('s_2', 2, 10)));
    writeFileSync(join(dir, 'chats', 'index.json'), '{"v":1,"chats":[');   // damaged
    writeFileSync(join(dir, 'chats', 's_bad.json'), '{"v":1,"rev":1,"session":{"id":"s_bad","mess');
    const again = createChatStore({ dir });
    assert.deepEqual(again.list().chats.map((c) => c.id).sort(), ['s_1', 's_2']);
    assert.deepEqual(again.list().damaged, ['s_bad']);
    assert.equal(readFileSync(join(dir, 'chats', 's_bad.json'), 'utf8').startsWith('{"v":1'), true, 'left as it is');
    rmSync(join(dir, 'chats', 'index.json'));   // lost
    assert.equal(createChatStore({ dir }).list().chats.length, 2);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a failed index write after a chat save cannot roll its revision back at restart or lose a stale window conflict', async () => {
  const dir = scratch();
  try {
    let failIndex = false, failures = 0;
    const store = createChatStore({ dir, io: { ...realIo, async writeFile(path, data) {
      if (failIndex && path.startsWith(join(dir, 'chats', 'index.json.tmp-'))) {
        failures++;
        throw Object.assign(new Error('no room for the index'), { code: 'ENOSPC' });
      }
      return realIo.writeFile(path, data);
    } } });
    const first = chat('s_index', 2, 10);
    await store.save(first.id, text(first));
    failIndex = true;
    const newer = chat(first.id, 6, 100, { title: 'Saved before the index failed' });
    assert.equal((await store.save(first.id, text(newer), { baseRev: 1 })).rev, 2);
    assert.equal(failures, 1, 'only the index write failed');
    assert.equal(JSON.parse(readFileSync(join(dir, 'chats', 'index.json'), 'utf8')).chats[0].rev, 1, 'the old index is still valid JSON');
    assert.equal((await readChat(store, first.id)).rev, 2, 'the chat was committed before the index failed');

    const restarted = createChatStore({ dir });
    const listed = restarted.list().chats[0];
    assert.equal(restarted.rev(first.id), 2, 'the chat envelope owns its revision');
    assert.equal(listed.title, newer.title);
    assert.equal(listed.updatedAt, newer.updatedAt);
    assert.equal(listed.bytes, statSync(join(dir, 'chats', `${first.id}.json`)).size);
    assert.deepEqual((await readChat(restarted, first.id)).session, newer);
    const saved = await restarted.save(first.id, text(first), { baseRev: 1 });
    assert.equal(saved.rev, 3, 'the revision is never reused after restart');
    assert.equal(saved.conflict, true, 'the older window still displaces a newer chat');
    const displaced = JSON.parse(readFileSync(join(dir, 'chats', 'conflicts', `${first.id}.rev2.json`), 'utf8'));
    assert.deepEqual(displaced.session, newer, 'the newer chat is kept before the older window saves');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('an indexed chat that becomes damaged is reported at restart without changing its file', async () => {
  const dir = scratch();
  try {
    const store = createChatStore({ dir });
    await store.save('s_broken', text(chat('s_broken', 2, 10)));
    const path = join(dir, 'chats', 's_broken.json'), broken = '{"v":1,"rev":2,"session":';
    writeFileSync(path, broken);
    const restarted = createChatStore({ dir });
    assert.deepEqual(restarted.list().chats, []);
    assert.deepEqual(restarted.list().damaged, ['s_broken']);
    assert.equal(readFileSync(path, 'utf8'), broken, 'recovery leaves the damaged chat untouched');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('a chat is refused when it does not match its id or is not JSON; delete removes the file and tolerates a chat that is gone', async () => {
  const dir = scratch();
  try {
    const store = createChatStore({ dir });
    await assert.rejects(store.save('s_a', text(chat('s_other', 1, 1))), (e) => e.status === 400);
    await assert.rejects(store.save('s_a', '{nope'), (e) => e.status === 400);
    await assert.rejects(store.save('../evil', text(chat('../evil', 1, 1))), (e) => e.status === 400);
    await store.save('s_a', text(chat('s_a', 1, 1)));
    assert.deepEqual(await store.remove('s_a'), { id: 's_a', removed: true });
    assert.deepEqual(await store.remove('s_a'), { id: 's_a', removed: false });
    assert.ok(!existsSync(join(dir, 'chats', 's_a.json')));
    assert.equal(createChatStore({ dir }).list().chats.length, 0);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('import: only chats the host does not have are taken, the ones it has are left alone, a bad one is reported', async () => {
  const dir = scratch();
  try {
    const store = createChatStore({ dir });
    await store.save('s_have', text(chat('s_have', 2, 10, { title: 'on the host' })));
    const result = await store.importMany([chat('s_have', 5, 10, { title: 'in the browser' }), chat('s_new', 3, 10), { id: '../x' }, { nope: true }]);
    assert.deepEqual(result.imported, ['s_new']);
    assert.deepEqual(result.skipped, ['s_have']);
    assert.equal(result.failed.length, 2);
    assert.equal((await readChat(store, 's_have')).session.title, 'on the host');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
