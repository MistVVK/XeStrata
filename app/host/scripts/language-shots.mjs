#!/usr/bin/env node
/**
 * Screenshots of the app screens in one interface language, from a hermetic preview host (simulated hardware and models,
 * scratch profile, no engine, no weights). The browser is a throw-away headless Chrome profile.
 *
 *   node scripts/language-shots.mjs <lang> <outDir> [route ...]      (needs `npm run build` first)
 *   example: node scripts/language-shots.mjs fr C:/Users/me/Desktop/fr-screens models settings setup
 *
 * Env: CHROME (path to Chrome/Edge), WIDTH (default 1280), HEIGHT (default 800).
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';
import { existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const [, , language = 'fr', outDir = 'language-shots', ...routes] = process.argv;
const screens = routes.length ? routes : ['models', 'settings', 'setup'];
if (!existsSync(join(root, 'dist/index.html'))) { console.error('Run npm run build first.'); process.exit(1); }
const chrome = process.env.CHROME ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', '/usr/bin/google-chrome', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(existsSync);
if (!chrome) { console.error('No Chrome or Edge found; set CHROME.'); process.exit(1); }

const home = mkdtempSync(join(tmpdir(), 'strata-language-shots-'));
const hostPort = 8830 + Math.floor(Math.random() * 40), webPort = hostPort + 100;
const scratch = { STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_WORKSPACE: home, STRATA_REAL_ENGINE: '0', STRATA_NATIVE_REPO: join(home, 'no-native-installation'),
  STRATA_SETTINGS_FILE: join(home, 'app-settings.json'), STRATA_COMPONENTS_DIR: join(home, 'components'), LOCALAPPDATA: join(home, 'Local'), APPDATA: join(home, 'Roaming'),
  XDG_DATA_HOME: join(home, 'xdg-data'), XDG_STATE_HOME: join(home, 'xdg-state'), XDG_CONFIG_HOME: join(home, 'xdg-config'), STRATA_PORT: String(hostPort) };
const children = [];
const run = (args, extra = {}) => { const child = spawn(process.execPath, args, { cwd: root, env: { ...process.env, ...scratch, ...extra }, stdio: 'ignore' }); children.push(child); return child; };
const stop = () => { for (const child of children) { try { child.kill(); } catch {} } try { rmSync(home, { recursive: true, force: true }); } catch {} };
process.on('exit', stop);

run(['server/index.mjs']);
for (let attempt = 0; ; attempt++) {
  try { if ((await fetch(`http://127.0.0.1:${hostPort}/api/health`)).ok) break; } catch {}
  if (attempt > 200) { console.error('The preview host did not start.'); process.exit(1); }
  await new Promise(done => setTimeout(done, 150));
}
run(['tests/fixtures/onboarding-ui-host.mjs'], { STRATA_UI_PREVIEW: '1', STRATA_ONBOARDING_UPSTREAM: `http://127.0.0.1:${hostPort}`, STRATA_ONBOARDING_PORT: String(webPort), STRATA_ONBOARDING_START: 'ready', STRATA_ONBOARDING_SCENARIO: 'telemetry', STRATA_ONBOARDING_PREVIEW: '1' });
for (let attempt = 0; ; attempt++) {
  try { if ((await fetch(`http://127.0.0.1:${webPort}/`)).ok) break; } catch {}
  if (attempt > 200) { console.error('The preview page did not start.'); process.exit(1); }
  await new Promise(done => setTimeout(done, 150));
}

mkdirSync(resolve(outDir), { recursive: true });
const browser = await chromium.launch({ executablePath: chrome, headless: true, args: ['--no-sandbox', '--disable-gpu', '--force-device-scale-factor=1'] });
const page = await browser.newPage({ viewport: { width: Number(process.env.WIDTH ?? 1280), height: Number(process.env.HEIGHT ?? 800) }, deviceScaleFactor: 1 });
await page.addInitScript(code => { try { localStorage.setItem('strata.prefs.v2', JSON.stringify({ language: code, onboardingSeen: true })); } catch {} }, language);
const problems = [];
page.on('pageerror', error => problems.push(`pageerror: ${error.message}`));
for (const screen of screens) {
  await page.goto(`http://127.0.0.1:${webPort}/#/${screen}`, { waitUntil: 'load' });
  await page.reload({ waitUntil: 'load' });
  await page.waitForTimeout(2500);
  const file = join(resolve(outDir), `${language}-${screen}.png`);
  await page.screenshot({ path: file });
  console.log(`${screen}: ${file} (lang=${await page.evaluate(() => document.documentElement.lang)})`);
}
await browser.close();
if (problems.length) console.log(problems.join('\n'));
stop();
process.exit(0);
