// Packaged-app walkthrough (GUI-VIS-1117 A63): with the host process gone, Retry only asks the same address again and does nothing visible. After two failed Retries the banner
// says the background part has stopped and what helps (the Windows and macOS app has no host watcher yet: a 0.2.17 item).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const { code } = await transform(src('lib/connection-retry.ts'), { loader: 'ts', format: 'esm' });
const { hostSeemsStopped, failedRetriesAfter, STOPPED_AFTER_FAILED_RETRIES } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const LINE = 'Strata’s background part has stopped. Close Strata and open it again.';

test('the line appears after two failed Retries, not after one, and a live connection clears the count', () => {
  assert.equal(STOPPED_AFTER_FAILED_RETRIES, 2);
  let n = 0;
  assert.equal(hostSeemsStopped(n), false);
  n = failedRetriesAfter(n, 'offline'); assert.equal(hostSeemsStopped(n), false, 'one failed Retry');
  n = failedRetriesAfter(n, 'offline'); assert.equal(hostSeemsStopped(n), true, 'two');
  n = failedRetriesAfter(n, 'offline'); assert.equal(hostSeemsStopped(n), true, 'and it stays');
  assert.equal(failedRetriesAfter(n, 'live'), 0, 'a Retry that connects clears it');
  assert.equal(hostSeemsStopped(failedRetriesAfter(1, 'live')), false);
});

test('the banner counts each finished Retry against the connection it ends with, shows the line only when offline, and clears the count when live', () => {
  const app = src('app/App.tsx');
  assert.ok(app.includes("void bootstrap().finally(() => { setRetrying(false); setFailedRetries(n => failedRetriesAfter(n, getState().connection)); });"));
  assert.ok(app.includes("useEffect(() => { if (connection === 'live') setFailedRetries(0); }, [connection]);"));
  assert.ok(app.includes('offline && hostSeemsStopped(failedRetries) ? <p className="connbar__stopped" role="alert">'));
});

test('the line exists in all ten languages', () => {
  const en = JSON.parse(readFileSync(new URL('../src/locales/guic-en.json', import.meta.url), 'utf8'));
  const key = Object.keys(en).find(k => en[k] === LINE);
  assert.ok(key, 'the English key');
  for (const l of ['zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he']) {
    const text = JSON.parse(readFileSync(new URL(`../src/locales/guic-${l}.json`, import.meta.url), 'utf8'))[key];
    assert.ok(text && text !== LINE, l);
  }
});
