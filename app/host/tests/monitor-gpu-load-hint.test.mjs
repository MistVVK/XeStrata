/** #5484: the GPU load card says what it measures (the share of time the GPU had work, not compute in use), as the label's
 *  tooltip in the app's Monitor (all nine languages) and on the web page's Monitor (English-only view). */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const TEXT = 'Share of the time the GPU had work to do. It does not say how much of its compute is in use.';
const read = (p) => readFile(new URL(p, import.meta.url), 'utf8');

test('the app Monitor gives the GPU load label its tooltip, translated', async () => {
  const src = await read('../src/screens/Monitor.tsx');
  assert.match(src, /<div className="st-metric__label" title=\{hint\}>/);
  assert.ok(src.includes(`<MonitorMetric label={tr("GPU load")} hint={tr("${TEXT}")}`));
});

test('the sentence is in all nine catalogs, translated in eight', async () => {
  const langs = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he'];
  const ids = new Set();
  for (const lang of langs) {
    const cat = JSON.parse(await read(`../src/locales/${lang}.json`));
    const id = Object.keys(cat).find((k) => (lang === 'en' ? cat[k] === TEXT : false));
    if (lang === 'en') { assert.ok(id, 'English entry'); ids.add(id); continue; }
    const [en] = ids;
    assert.ok(cat[en] && cat[en] !== TEXT, `${lang}: ${cat[en]}`);
  }
});

test('the web Monitor shows the same sentence on its GPU load label', async () => {
  const app = await read('../../../terminal/serve/web/app.js');
  assert.ok(app.includes(`{key: "gpu", label: "GPU load", icon: "gpu", unit: "%", series: "gpu_util", max: 100,\n   hint: "${TEXT}"}`));
  // #4968 gave each card's label an id (ml-<key>) for the Model card's own title
  assert.ok(app.includes('<span class="st-metric__label" id="ml-${m.key}"${m.hint ? ` title="${esc(m.hint)}"` : ""}>'));
});
