/**
 * Where the Chromium binary the probes drive actually lives.
 *
 * The probes launched a fixed macOS cache path, so on Windows and Linux they
 * printed "No cached Chromium found" and every runtime gate (contrast, measure,
 * luminance, motion) was skipped rather than run. Resolution is now a single
 * place with the same order everywhere:
 *
 *   1. STRATA_CHROME — an explicit override, for a CI image or a second Chrome
 *   2. the cached Playwright build (macOS, the original platforms)
 *   3. an installed browser: Chrome, Edge, Chromium
 *
 * `requireChromium` exits with the harness message the scripts already used, so
 * a machine with no browser at all still fails loudly instead of silently
 * passing a gate.
 */
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const PLAYWRIGHT_MAC = [
  join(homedir(), 'Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'),
  join(homedir(), 'Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell'),
];

const INSTALLED = [
  // Windows
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  // macOS
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  // Linux
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
];

export function findChromium() {
  const candidates = [process.env.STRATA_CHROME, ...PLAYWRIGHT_MAC, ...INSTALLED].filter(Boolean);
  return candidates.find((path) => existsSync(path)) ?? null;
}

export function requireChromium() {
  const path = findChromium();
  if (!path) {
    console.error('No Chromium found. Set STRATA_CHROME, or install Chrome, Edge or Chromium.');
    process.exit(2);
  }
  return path;
}
