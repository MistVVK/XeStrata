#!/usr/bin/env node
/**
 * LOAD-PICS end to end: the Models page of a running host, driven in a real Chrome (DevTools protocol, no packages), pressing "Load model" on a model that is
 * downloaded, with Read pictures on and the picture files not on the computer.
 *
 *   HOST_URL=http://127.0.0.1:18700 CHROME=/usr/bin/google-chrome OUT=/tmp/lp-out node scripts/load-pictures-e2e.mjs
 *
 * It looks at the page, presses the button, and writes down once per half second what the page shows (the progress card's words and bar, any dialog, the job's kind
 * and state), until the job ends or LIMIT_S passes. It writes <OUT>/timeline.json and screenshots; the files on disk are checked by the shell script around it.
 *
 * Environment: HOST_URL (required), CHROME (default /usr/bin/google-chrome), OUT (default ./lp-out), LIMIT_S (default 900), CDP_PORT (default 9333),
 * MODEL (the version's name on the page, default "High quality"), BUTTON (default "Load model").
 */
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const HOST = process.env.HOST_URL;
if (!HOST) { console.error('HOST_URL is required'); process.exit(2); }
const CHROME = process.env.CHROME ?? '/usr/bin/google-chrome';
const OUT = process.env.OUT ?? './lp-out';
const LIMIT = Number(process.env.LIMIT_S ?? 900) * 1000;
const PORT = Number(process.env.CDP_PORT ?? 9333);
const BUTTON = process.env.BUTTON ?? 'Load model';
mkdirSync(OUT, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const profile = mkdtempSync(join(tmpdir(), 'lp-chrome-'));
const chrome = spawn(CHROME, ['--headless=new', '--no-sandbox', '--disable-gpu', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`, '--window-size=1400,1100', 'about:blank'], { stdio: 'ignore' });
const finish = code => { try { chrome.kill('SIGTERM'); } catch { /* gone */ } setTimeout(() => { try { rmSync(profile, { recursive: true, force: true }); } catch { /* best effort */ } process.exit(code); }, 800); };
process.on('SIGINT', () => finish(130)); process.on('SIGTERM', () => finish(143));

async function target() {
  for (let i = 0; i < 80; i++) {
    try { const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json(); const page = list.find(t => t.type === 'page'); if (page) return page.webSocketDebuggerUrl; } catch { /* not up yet */ }
    await sleep(250);
  }
  throw new Error('Chrome did not start');
}
const ws = new WebSocket(await target());
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = () => reject(new Error('DevTools socket')); });
let seq = 0; const waiting = new Map();
ws.onmessage = event => { const message = JSON.parse(event.data); const w = waiting.get(message.id); if (w) { waiting.delete(message.id); message.error ? w.reject(new Error(message.error.message)) : w.resolve(message.result); } };
const send = (method, params = {}) => new Promise((resolve, reject) => { const id = ++seq; waiting.set(id, { resolve, reject }); ws.send(JSON.stringify({ id, method, params })); });
const run = async expression => { const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(r.exceptionDetails.text + ' ' + (r.exceptionDetails.exception?.description ?? '')); return r.result.value; };
const shot = async name => { const r = await send('Page.captureScreenshot', { format: 'png' }); writeFileSync(join(OUT, `${name}.png`), Buffer.from(r.data, 'base64')); };

try {
  await send('Page.enable');
  await send('Page.navigate', { url: `${HOST}/#/models` });
  // the page is up when the Load button is on it and enabled
  const ready = `(() => { const b = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === ${JSON.stringify(BUTTON)}); return !!b && !b.disabled; })()`;
  let up = false;
  for (let i = 0; i < 240 && !up; i++) { up = await run(ready).catch(() => false); if (!up) await sleep(500); }
  if (!up) { await shot('never-ready'); throw new Error(`The "${BUTTON}" button never came up`); }
  const before = await run(`(() => { const t = document.body.innerText; const box = document.querySelector('.native-image-reading input[type=checkbox]'); return { readPicturesOn: !!box && box.checked, offersFiles: t.includes('Picture reading needs a few more files'), downloadButton: [...document.querySelectorAll('button')].some(b => b.textContent.trim() === 'Download picture-reading files'), modelFiles: document.querySelector('.native-path')?.textContent ?? null, shownModel: document.querySelector('.native-version')?.innerText?.slice(0, 200) ?? null }; })()`);
  await shot('1-before');
  const baseDialogs = await run("document.querySelectorAll('[role=dialog], dialog[open]').length");   // dialogs the page always has (closed ones the app keeps mounted)
  const clicked = await run(`(() => { const b = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === ${JSON.stringify(BUTTON)}); b.click(); return true; })()`);
  const started = Date.now();
  const timeline = []; let last = '', shots = 0, ended = null;
  while (Date.now() - started < LIMIT) {
    const seen = await run(`(async () => {
      const card = document.querySelector('.native-progress');
      const state = await (await fetch('/api/native')).json();
      const job = [...(state.jobs ?? [])].reverse()[0] ?? null;
      return { card: card ? card.innerText.replace(/\\s+/g, ' ').trim() : null, dialogs: document.querySelectorAll('[role=dialog], dialog[open]').length - ${baseDialogs},
        job: job && { startedAt: job.startedAt, kind: job.kind, state: job.state, summary: job.summary, error: job.error, details: job.details ?? null, bar: job.result?.install?.download ?? null },
        instance: (state.instances ?? []).map(i => ({ status: i.status, vision: i.vision, note: i.picturesNote ?? null, gpus: i.gpus })),
        note: [...document.querySelectorAll('.native-spill-note')].map(n => n.innerText.trim()), error: [...document.querySelectorAll('.inline-error, [role=alert]')].map(n => n.innerText.trim()).filter(Boolean) };
    })()`);
    const key = JSON.stringify([seen.card, seen.dialogs, seen.job?.kind, seen.job?.state, seen.job?.summary, seen.instance, seen.note, seen.error, Math.floor((seen.job?.bar?.percent ?? 0) / 25)]);
    if (key !== last) { last = key; timeline.push({ t: +((Date.now() - started) / 1000).toFixed(1), ...seen }); if (seen.card && shots < 2 && seen.job?.state === 'running') await shot(`2-running-${++shots}`); }
    if (seen.job && Date.parse(seen.job.startedAt) >= started - 2000 && seen.job.state !== 'running') { ended = seen.job.state; break; }
    await sleep(500);
  }
  await sleep(1500);
  await shot('3-after');
  const after = await run(`(() => ({ text: [...document.querySelectorAll('.native-spill-note, .native-ready, .toast, [role=status]')].map(n => n.innerText.trim()).filter(Boolean), buttons: [...document.querySelectorAll('.native-model-actions button')].map(b => b.textContent.trim()), pictureOffer: document.body.innerText.includes('Picture reading needs a few more files') }))()`);
  const posted = await run("fetch('/__fixture/posted').then(r => r.ok ? r.json() : null).catch(() => null)");   // only the UI-preview fixture keeps what the page posted
  writeFileSync(join(OUT, 'timeline.json'), JSON.stringify({ clicked, posted, before, ended, seconds: +((Date.now() - started) / 1000).toFixed(1), timeline, after }, null, 1));
  console.log(JSON.stringify({ ended, seconds: +((Date.now() - started) / 1000).toFixed(1), steps: timeline.length, before, after }));
  finish(ended === 'done' ? 0 : 1);
} catch (error) {
  console.error(error.message);
  writeFileSync(join(OUT, 'error.txt'), String(error.stack ?? error));
  finish(2);
}
