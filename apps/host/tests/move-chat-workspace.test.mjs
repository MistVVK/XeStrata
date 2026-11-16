// #6051: a chat can be moved to another workspace from its Actions menu; the sidebar then lists it there and its next turn works there.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/session-workspace.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { moveSessionToWorkspace, workspaceIdForSession } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

const workspaces = [{ id: 'root', name: 'Strata', path: 'C:\\Docs\\Strata\\Workspace', aliases: ['C:\\old\\Workspace'] }, { id: 'b', name: 'LocalLLM presentation', path: 'D:\\Projects\\LocalLLM presentation' }];
const sessions = [{ id: 'one', workspace: 'C:\\Docs\\Strata\\Workspace', title: 'A', messages: [{ id: 'm1' }] }, { id: 'two', workspace: 'C:\\old\\Workspace', title: 'B' }];
// the sidebar's own grouping rule (Sidebar.tsx): a chat is under a workspace when its saved path is the workspace's path or one of its aliases
const listedUnder = (list, w) => list.filter(s => s.workspace === w.path || !!w.aliases?.includes(s.workspace)).map(s => s.id);

test('moving a chat lists it under the destination only, and keeps everything else of the chat', () => {
  const { sessions: after, refused } = moveSessionToWorkspace(sessions, 'one', workspaces, 'b');
  assert.equal(refused, null);
  assert.deepEqual(listedUnder(after, workspaces[1]), ['one']);
  assert.deepEqual(listedUnder(after, workspaces[0]), ['two']);
  assert.equal(workspaceIdForSession(workspaces, after.find(s => s.id === 'one')), 'b');
  const moved = after.find(s => s.id === 'one');
  assert.deepEqual({ ...moved, workspace: sessions[0].workspace }, sessions[0]);       // title, messages and ids unchanged
  assert.equal(sessions[0].workspace, 'C:\\Docs\\Strata\\Workspace');                 // the input list is not changed in place
});

test('the move survives saving and reading the chat list back (restart)', () => {
  const { sessions: after } = moveSessionToWorkspace(sessions, 'two', workspaces, 'b');
  const reread = JSON.parse(JSON.stringify(after));
  assert.equal(workspaceIdForSession(workspaces, reread.find(s => s.id === 'two')), 'b');
  // and back again: the chat now sits on the workspace's own path, not on the old alias
  const back = moveSessionToWorkspace(reread, 'two', workspaces, 'root').sessions;
  assert.equal(back.find(s => s.id === 'two').workspace, 'C:\\Docs\\Strata\\Workspace');
});

test('a working chat, an unknown chat or destination, and the workspace it is already in change nothing', () => {
  for (const [args, why] of [
    [['one', workspaces, 'b', new Set(['one'])], 'working'],
    [['gone', workspaces, 'b'], 'no-chat'],
    [['one', workspaces, 'deleted'], 'no-workspace'],
    [['two', workspaces, 'root'], 'same'],          // 'two' is under root through the alias
  ]) {
    const result = moveSessionToWorkspace(sessions, ...args);
    assert.equal(result.refused, why);
    assert.deepEqual(result.sessions, sessions);
  }
});

test('the Actions menu offers the move with the approved words, in all nine languages', async () => {
  const sidebar = await readFile(new URL('../src/app/Sidebar.tsx', import.meta.url), 'utf8');
  const en = JSON.parse(await readFile(new URL('../src/locales/guic-en.json', import.meta.url), 'utf8'));
  for (const text of ['Move to workspace\u2026', 'Move chat to another workspace', 'Workspace', 'Move chat',
    'The chat is listed under this workspace. Its next answer works in that workspace\u2019s folder, unless you chose a folder for this chat.',
    'This chat is still working. Stop it or wait for it to finish, then move it.']) {
    assert.ok(Object.values(en).includes(text), text);
    assert.ok(sidebar.includes(`"${text}"`) || sidebar.includes(JSON.stringify(text).slice(1, -1)) || sidebar.includes(text.replace('\u2026', '\\u2026').replace('\u2019', '\\u2019')), `Sidebar.tsx uses: ${text}`);
  }
  assert.match(sidebar, /moveSession\(moveId, moveTo\)/);
  for (const lang of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr']) {
    const cat = JSON.parse(await readFile(new URL(`../src/locales/guic-${lang}.json`, import.meta.url), 'utf8'));
    for (const key of Object.keys(en)) assert.ok(cat[key] && cat[key] !== en[key] || key === 'guic03' && cat[key], `${lang} ${key}`);
  }
});
