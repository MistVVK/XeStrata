// Measures the chat page's heap, DOM size and main-thread time per streamed reply. Needs a throwaway host (STRATA_REAL_ENGINE=0, port 18791), scripts/chat-growth-proxy.mjs (port 18790) and Chrome.
import { chromium } from 'playwright-core';
import { deflateSync } from 'node:zlib';
import { randomBytes } from 'node:crypto';
import { execSync } from 'node:child_process';
const BASE = process.env.BASE ?? 'http://127.0.0.1:18790';
const TURNS = Number(process.env.TURNS ?? 6), PICS = Number(process.env.PICS ?? 3), MON = process.env.MON !== '0';
const crc = (() => { const t = new Int32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; }); return b => { let c = -1; for (const x of b) c = t[(c ^ x) & 255] ^ (c >>> 8); return (c ^ -1) >>> 0; }; })();
function png(w, h) {
  const chunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const body = Buffer.concat([Buffer.from(type), data]); const c = Buffer.alloc(4); c.writeUInt32BE(crc(body)); return Buffer.concat([len, body, c]); };
  const raw = Buffer.alloc((w * 3 + 1) * h); for (let y = 0; y < h; y++) { raw[y * (w * 3 + 1)] = 0; randomBytes(w * 3).copy(raw, y * (w * 3 + 1) + 1); }
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 1 })), chunk('IEND', Buffer.alloc(0))]);
}
const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: ['--enable-precise-memory-info', '--js-flags=--expose-gc'] });
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();
const cdp = await ctx.newCDPSession(page);
await cdp.send('Performance.enable');
await fetch(BASE + '/api/model-providers', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ kind: 'custom', provider: null, name: 'Mock model', baseUrl: 'http://127.0.0.1:18898/v1', protocol: 'openai-chat', apiKey: '', models: [{ id: 'mock-model', name: 'Mock model' }] }) });
await page.goto(BASE + '/'); await page.waitForTimeout(1500);
const skip = page.getByText('Go straight to the app'); await skip.waitFor({ timeout: 10000 }).catch(() => {}); if (await skip.count()) { await skip.click(); await page.waitForTimeout(800); }
async function metrics(tag) {
  await cdp.send('HeapProfiler.collectGarbage').catch(() => {});
  const m = Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(x => [x.name, x.value]));
  const dom = await cdp.send('Memory.getDOMCounters').catch(() => ({}));
  const sess = await page.evaluate(async () => { try { const h = { 'x-strata-client': 'ui' }; const l = await (await fetch('/api/chats', { headers: h })).json(); return l.chats.reduce((n, c) => n + (c.bytes ?? 0), 0); } catch { return -1; } });
  console.log(`${tag.padEnd(22)} heap=${(m.JSHeapUsedSize / 1048576).toFixed(0)}MB total=${(m.JSHeapTotalSize / 1048576).toFixed(0)}MB nodes=${dom.nodes ?? m.Nodes} listeners=${dom.jsEventListeners ?? '-'} docs=${m.Documents} ls=${(sess / 1e6).toFixed(2)}M`);
  return m.JSHeapUsedSize;
}
await page.locator('.chat-model-picker').first().click(); await page.locator('.model-picker-api').first().click(); await page.waitForTimeout(500);
await metrics('start');
const ta = page.locator('.composer textarea');
for (let t = 1; t <= TURNS; t++) {
  if (t <= PICS) {
    const b64 = png(900, 900).toString('base64');
    await page.evaluate(b64 => { const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0)); const d = new DataTransfer(); d.items.add(new File([bytes], 'image.png', { type: 'image/png' })); document.querySelector('.composer textarea').dispatchEvent(new ClipboardEvent('paste', { clipboardData: d, bubbles: true, cancelable: true })); }, b64);
    await page.waitForTimeout(600);
  }
  await ta.fill('turn ' + t + ' write a long answer'); 
  await page.getByRole('button', { name: 'Send' }).click();
  const t0 = Date.now();
  let peak = 0; const cpu0 = (await cdp.send('Performance.getMetrics')).metrics.find(x => x.name === 'TaskDuration').value;
  for (;;) {
    await page.waitForTimeout(2000);
    const m = Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(x => [x.name, x.value]));
    peak = Math.max(peak, m.JSHeapUsedSize);
    const busy = await page.getByRole('button', { name: /Stop/ }).count();
    if (!busy || Date.now() - t0 > 240000) break;
  }
  const cpu1 = (await cdp.send('Performance.getMetrics')).metrics.find(x => x.name === 'TaskDuration').value;
  console.log(`  turn ${t}: ${(Date.now() - t0) / 1000}s peak-heap=${(peak / 1048576).toFixed(0)}MB main-thread-busy=${(cpu1 - cpu0).toFixed(1)}s`);
  await metrics('after turn ' + t);
}
if (MON) { await page.getByText('Monitor', { exact: true }).first().click().catch(() => console.log('no monitor link')); await page.waitForTimeout(25000); await metrics('monitor 25s'); }
try { console.log(execSync('powershell -NoProfile -c "(Get-Process chrome | Measure-Object WorkingSet64 -Sum).Sum/1MB"').toString().trim(), 'MB chrome working set total'); } catch {}
await browser.close();
