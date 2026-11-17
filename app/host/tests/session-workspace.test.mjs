import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/session-workspace.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { workspaceIdForSession } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const workspaces = [{ id: 'a', name: 'A', path: 'C:\A', aliases: ['C:\old'] }, { id: 'b', name: 'B', path: 'D:\Project' }];

test('a chat finds the workspace it was made in, by path or by an earlier folder (#3754)', () => {
  assert.equal(workspaceIdForSession(workspaces, { workspace: 'D:\Project' }), 'b');
  assert.equal(workspaceIdForSession(workspaces, { workspace: 'C:\old' }), 'a');
});
test('a chat with an unknown or no workspace changes nothing', () => {
  assert.equal(workspaceIdForSession(workspaces, { workspace: 'E:\gone' }), null);
  assert.equal(workspaceIdForSession(workspaces, undefined), null);
});
