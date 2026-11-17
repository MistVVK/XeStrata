/**
 * Blocked folders (#5737, Settings -> Tools): the folders the assistant may not touch, in any mode, until the person removes them. The
 * approval gate (plugins/strata-gate.mjs blockedFolderIn) refuses every tool call whose arguments name a path inside one. This module
 * checks what the person enters and works out, once per folder, the spellings that reach the same place, so `D:\Private`,
 * `d:/private/`, `D:\PRIVAT~1`, `%USERPROFILE%\Secret`, `~/Secret` and `/mnt/d/private` are all caught.
 */
import { spawnSync } from 'node:child_process';
import { realpathSync } from 'node:fs';
import { homedir } from 'node:os';
import { pathForm } from './plugins/strata-gate.mjs';

export const MAX_BLOCKED = 50;

/** A folder or drive as the person typed it, cleaned, or an error in plain words. Only full paths: a drive (D:), a folder on one, a
 * network share (\\server\share) or, off Windows, a path from /. */
export function checkFolder(raw, platform = process.platform) {
  let s = String(raw ?? '').trim().replace(/^"(.*)"$/, '$1').trim();
  if (!s || s.length > 1000 || /[\0\r\n]/.test(s)) return { error: 'Enter the full path of a folder or a drive, for example D:\\Private or D:' };
  if (platform === 'win32') {
    if (/^[A-Za-z]:$/.test(s)) s += '\\';
    if (!/^[A-Za-z]:[\\/]/.test(s) && !/^\\\\[^\\]+\\[^\\]+/.test(s)) return { error: 'Enter the full path of a folder or a drive, for example D:\\Private or D:' };
    s = s.replace(/\//g, '\\');
    if (s.length > 3) s = s.replace(/\\+$/, '');
    return { folder: s[0].toUpperCase() + s.slice(1) };
  }
  if (!s.startsWith('/')) return { error: 'Enter the full path of a folder, for example /home/you/private' };
  return { folder: s.length > 1 ? s.replace(/\/+$/, '') : s };
}

/** The Windows short (8.3) name of a folder that exists, or null. */
function shortName(folder) {
  try {
    const r = spawnSync('cmd.exe', ['/d', '/c', `for %I in ("${folder}") do @echo %~sI`], { encoding: 'utf8', windowsHide: true, timeout: 5000 });
    const out = (r.stdout ?? '').trim().split(/\r?\n/).pop();
    return out && /^[A-Za-z]:\\/.test(out) ? out : null;
  } catch { return null; }
}

/** Every spelling the gate compares (pathForm: lower case, forward slashes, no trailing slash except a drive's). */
export function blockedForms(folders, { platform = process.platform, home = homedir(), realpath = p => realpathSync.native(p), short = shortName } = {}) {
  return (folders ?? []).map(folder => {
    const names = new Set([folder]);
    try { names.add(realpath(folder)); } catch { /* not there (yet): its own name still counts */ }
    if (platform === 'win32') { const s = short(folder); if (s) names.add(s); }
    const forms = new Set();
    for (const name of names) {
      const f = pathForm(name).replace(/(.)\/+$/, '$1');
      const drive = /^([a-z]):(\/.*)?$/.exec(f);
      if (drive && !drive[2]) { forms.add(`${drive[1]}:`); forms.add(`/mnt/${drive[1]}`); continue; }   // a whole drive
      forms.add(f);
      if (drive) { forms.add(`/mnt/${drive[1]}${drive[2]}`); forms.add(`/${drive[1]}${drive[2]}`); }
      const h = pathForm(home).replace(/\/+$/, '');
      if (h && (f === h || f.startsWith(`${h}/`))) {
        const rest = f.slice(h.length);
        for (const prefix of ['~', '%userprofile%', '$env:userprofile', '$home', '${env:userprofile}']) forms.add(prefix + rest);
      }
    }
    return { shown: folder, forms: [...forms].filter(Boolean) };
  });
}
