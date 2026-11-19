// 0217 lane FIX-APP, #6153: the App Host died with a V8 heap out-of-memory after about 9 h. The host writes a monitor frame about once a second
// to every open /api/events stream. A window that stops reading (a suspended or hidden WebView, a frozen tab) leaves its socket open; Node keeps
// every frame it cannot send in the socket's own queue, so the queue grows by the size of a frame each second without bound.
// The tests connect a client that never reads and send frames of about 100 KB, the size of a monitor frame with its history.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import net from 'node:net';
import { EventHub, EVENT_BACKLOG_MAX } from '../server/util.mjs';

const FRAME = { rows: 'x'.repeat(100 * 1024) };
const pause = ms => new Promise(r => setTimeout(r, ms));

/** A host with a hub, and a raw client that sends the request and then never reads a byte. Both are closed when the test ends, passed or not. */
async function stalled(t) {
  const hub = new EventHub();
  const held = [];
  const server = http.createServer((req, res) => { held.push(res); hub.add(res); });
  const port = await new Promise(r => server.listen(0, '127.0.0.1', () => r(server.address().port)));
  const client = net.connect(port, '127.0.0.1');
  client.on('error', () => {});
  client.pause();
  t.after(() => { client.destroy(); server.closeAllConnections(); server.close(); });
  await new Promise(r => client.on('connect', r));
  client.write('GET /api/events HTTP/1.1\r\nHost: x\r\n\r\n');
  while (!held.length) await pause(20);
  return { hub, res: held[0] };
}

test('an event stream client that does not read cannot grow the host without bound', { timeout: 60000 }, async t => {
  const { hub, res } = await stalled(t);
  let peak = 0;
  for (let i = 0; i < 400; i += 1) {                      // 400 frames of ~100 KB: 40 MB offered to a client that reads nothing
    hub.send('native', FRAME);                            // an event that is not a replaceable snapshot
    peak = Math.max(peak, res.writableLength);
  }
  assert.ok(peak <= EVENT_BACKLOG_MAX + 200 * 1024, `the unsent queue of one stalled client reached ${(peak / 1048576).toFixed(1)} MB`);
  assert.equal(hub.size, 0, 'a client that has fallen this far behind is let go (the page reconnects by itself)');
});

test('while a client is behind, a monitor snapshot is skipped (the next one replaces it) and the queue stays short', { timeout: 30000 }, async t => {
  const { hub, res } = await stalled(t);
  for (let i = 0; i < 50; i += 1) hub.send('monitor', FRAME);
  const queued = res.writableLength;
  for (let i = 0; i < 20; i += 1) hub.send('monitor', FRAME);
  assert.equal(res.writableLength, queued, 'no monitor frame is added to a queue that has not drained');
  assert.ok(queued < 1024 * 1024, `${(queued / 1048576).toFixed(1)} MB queued`);
  assert.equal(hub.size, 1, 'a client that only misses snapshots stays connected');
});

test('a client that reads is not touched: every frame arrives, in order', { timeout: 30000 }, async t => {
  const hub = new EventHub();
  const server = http.createServer((req, res) => hub.add(res));
  const port = await new Promise(r => server.listen(0, '127.0.0.1', () => r(server.address().port)));
  t.after(() => { server.closeAllConnections(); server.close(); });
  const got = [];
  let request;
  const done = new Promise((resolve, reject) => {
    request = http.get({ host: '127.0.0.1', port, path: '/api/events' }, res => {
      let text = '';
      res.setEncoding('utf8');
      res.on('data', chunk => {
        text += chunk;
        got.length = 0;
        for (const m of text.matchAll(/event: native\ndata: (\{"n":\d+\})\n\n/g)) got.push(m[1]);
        if (got.length === 200) resolve();
      });
      res.on('error', reject);
    });
    request.on('error', reject);
  });
  t.after(() => request.destroy());
  while (hub.size === 0) await pause(20);
  for (let i = 0; i < 200; i += 1) hub.send('native', { n: i });
  await done;
  assert.deepEqual(got, Array.from({ length: 200 }, (_, i) => `{"n":${i}}`));
});
