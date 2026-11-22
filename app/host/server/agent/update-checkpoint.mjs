import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { open, rename, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';

const run = promisify(execFile);
async function git(cwd, ...args) {
  const { stdout } = await run('git', ['--no-optional-locks', '-C', cwd, ...args],
    { windowsHide: true, timeout: 15000, maxBuffer: 64 * 1024 * 1024, env: { ...process.env, LC_ALL: 'C' } });
  return stdout;
}

export async function workspaceState(cwd) {
  try { await git(cwd, 'rev-parse', '--show-toplevel'); }
  catch (error) {
    if (error.stderr?.includes('not a git repository')) return { cwd, repository: false };
    throw error;
  }
  const flags = ['--no-ext-diff', '--no-textconv', '--binary'];
  return { cwd, status: await git(cwd, 'status', '--porcelain=v1', '--branch', '--untracked-files=all'),
    staged: await git(cwd, 'diff', '--cached', ...flags), unstaged: await git(cwd, 'diff', ...flags) };
}

export async function saveCheckpoint(dir, name, value) {
  await mkdir(dir, { recursive: true, mode: 0o700 });
  const file = join(dir, name + '.json');
  const tmp = file + '.' + randomUUID() + '.tmp';
  const handle = await open(tmp, 'wx', 0o600);
  try { await handle.writeFile(JSON.stringify(value, null, 2)); await handle.sync(); }
  finally { await handle.close(); }
  await rename(tmp, file);
  return file;
}
