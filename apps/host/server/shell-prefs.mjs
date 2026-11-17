/**
 * The desktop shell's own choices, kept in <Strata data folder>/shell-prefs.json (#3402, #3477): the shell reads the file when it starts
 * (the port it asks for, whether closing the window keeps Strata in the tray, the language of the tray menu), before any page exists.
 * The page changes them through the host (POST /api/shell/prefs); a change of the port or the tray applies the next time Strata starts.
 *
 *   { "port": 0 | 1024..65535, "tray": false, "language": "en" }
 *
 * Port 0 is the default: the shell asks for 47831 and takes a free port when that one is busy.
 */
import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { componentsDirectory } from './control-runtime.mjs';

export const DEFAULT_PORT = 47831;
export const LANGUAGES = ['en', 'zh', 'ja', 'de', 'fr', 'es', 'ru', 'pt', 'ko'];
export const DEFAULTS = Object.freeze({ port: 0, tray: false, language: 'en' });

export const validPort = value => Number.isInteger(value) && value >= 1024 && value <= 65535;

export function shellPrefsFile({ env = process.env } = {}) {
  return join(dirname(componentsDirectory({ env })), 'shell-prefs.json');
}

/** The file's content with every field checked; anything missing or wrong is the default. */
export function cleanPrefs(raw) {
  const value = raw && typeof raw === 'object' ? raw : {};
  return {
    port: validPort(value.port) ? value.port : 0,
    tray: value.tray === true,
    language: LANGUAGES.includes(value.language) ? value.language : 'en',
  };
}

export function readShellPrefs(file = shellPrefsFile()) {
  try { return cleanPrefs(JSON.parse(readFileSync(file, 'utf8').replace(/^﻿/, ''))); } catch { return { ...DEFAULTS }; }
}

/** Merge a change into the file. Throws a plain sentence for a port that cannot be used. */
export function writeShellPrefs(patch, file = shellPrefsFile()) {
  const next = readShellPrefs(file);
  if (patch && 'port' in patch) {
    const wanted = patch.port === null || patch.port === '' || patch.port === 0 ? 0 : Number(patch.port);
    if (wanted !== 0 && !validPort(wanted)) throw new Error('Choose a port between 1024 and 65535, or leave it empty to let Strata choose.');
    next.port = wanted;
  }
  if (patch && 'tray' in patch) next.tray = patch.tray === true;
  if (patch && 'language' in patch && LANGUAGES.includes(patch.language)) next.language = patch.language;
  mkdirSync(dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  writeFileSync(temporary, JSON.stringify(next, null, 2) + '\n');
  renameSync(temporary, file);
  return next;
}

/**
 * What the shell tells the host about this start: STRATA_PORT_NOTICE is "busy:<wanted>:<used>" when the port it wanted was taken
 * (set by the shell, never by the person). The page turns it into a sentence in the person's language.
 */
export function parsePortNotice(value) {
  const m = /^busy:(\d{1,5}):(\d{1,5})$/.exec(String(value ?? ''));
  return m ? { wanted: Number(m[1]), used: Number(m[2]) } : null;
}

/** The notification route of the shell's bridge (loopback, token). Null outside the desktop app. */
export function createShellNotifier({ env = process.env, fetcher = fetch } = {}) {
  const base = env.STRATA_SHELL_BRIDGE, token = env.STRATA_SHELL_TOKEN;
  if (!base || !token || !/^http:\/\/127\.0\.0\.1:\d+$/.test(base)) return null;
  return {
    async notify({ title, body }) {
      const query = `title=${encodeURIComponent(String(title ?? '').slice(0, 120))}&body=${encodeURIComponent(String(body ?? '').slice(0, 300))}`;
      const response = await fetcher(`${base}/shell/notify?${query}`, { method: 'POST', headers: { authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(5000) });
      let reply = null;
      try { reply = await response.json(); } catch { /* no body */ }
      return { shown: response.ok && reply?.shown === true };
    },
  };
}
