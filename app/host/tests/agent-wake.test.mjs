// A turn the Harness starts by itself after ours ended (a background command finished) is noticed, once, and only then.
import test from 'node:test';
import assert from 'node:assert/strict';
import { watchForWake, WAKE_EVENTS } from '../server/agent/wake-watch.mjs';

function fakeGateway() {
  const g = { opened: [], closed: 0, handlers: null };
  g.openStream = (method, params, handlers) => { g.opened.push({ method, params }); g.handlers = handlers; return { close: () => { g.closed++; } }; };
  g.snapshot = cursor => g.handlers.onItem({ type: 'snapshot', cursor, records: [] });
  g.event = (seq, type, data = {}) => g.handlers.onItem({ type: 'event', event: { seq, type, data } });
  return g;
}

test('a job completion notice after the turn ended wakes the chat once, with its sequence number', () => {
  const gw = fakeGateway(); const woke = [];
  watchForWake({ gw, sessionId: 's1', onWake: seq => woke.push(seq) });
  assert.deepEqual(gw.opened[0].params.request.address, { kind: 'session', sessionId: 's1' });
  gw.snapshot(40);
  gw.event(41, 'turn/start', { source: { kind: 'tool-jobs', form: 'notice' } });
  gw.event(42, 'user/message');
  assert.deepEqual(woke, [41]);
  assert.equal(gw.closed, 1, 'the idle watch ends when the real turn takes over');
});

test('what happened before the watch opened, and events that open no turn, never wake it', () => {
  const gw = fakeGateway(); const woke = [];
  watchForWake({ gw, sessionId: 's1', onWake: seq => woke.push(seq) });
  gw.event(5, 'turn/start');
  gw.snapshot(10);
  gw.event(10, 'turn/start');
  gw.event(11, 'usage');
  gw.event(12, 'agent/inbox/spliced');
  assert.deepEqual(woke, []);
  gw.event(13, 'step/start');
  assert.deepEqual(woke, [13]);
});

test('a lost stream or the time limit closes the watch without waking anything', async () => {
  const lost = fakeGateway(); const woke = [];
  watchForWake({ gw: lost, sessionId: 's', onWake: s => woke.push(s) });
  lost.snapshot(1);
  lost.handlers.onError(new Error('gone'));
  lost.event(2, 'turn/start');
  assert.deepEqual(woke, []);
  assert.equal(lost.closed, 1);

  const slow = fakeGateway();
  watchForWake({ gw: slow, sessionId: 's', onWake: s => woke.push(s), maxMs: 20 });
  slow.snapshot(1);
  await new Promise(r => setTimeout(r, 60));
  assert.equal(slow.closed, 1);
  slow.event(2, 'turn/start');
  assert.deepEqual(woke, []);
});

test('a turn that began while the socket was away is found in the snapshot of the re-opened stream', () => {
  const gw = fakeGateway(); const woke = [];
  watchForWake({ gw, sessionId: 's', onWake: s => woke.push(s) });
  gw.snapshot(10);
  // The gateway client re-opens the follow after a reconnect: a fresh snapshot, with what happened meanwhile in its records.
  gw.handlers.onItem({ type: 'snapshot', cursor: 14, records: [
    { type: 'event', event: { seq: 9, type: 'turn/start', data: {} } },
    { type: 'event', event: { seq: 11, type: 'usage', data: {} } },
    { type: 'event', event: { seq: 12, type: 'turn/start', data: {} } },
    { type: 'event', event: { seq: 13, type: 'user/message', data: {} } },
  ] });
  assert.deepEqual(woke, [12], 'the first new turn, once; nothing from before the watch opened');
  assert.equal(gw.closed, 1);
});

test('closing it by hand is final', () => {
  const gw = fakeGateway(); const woke = [];
  const w = watchForWake({ gw, sessionId: 's', onWake: s => woke.push(s) });
  gw.snapshot(1);
  w.close(); w.close();
  gw.event(2, 'turn/start');
  assert.deepEqual(woke, []);
  assert.equal(gw.closed, 1);
  assert.ok(WAKE_EVENTS.has('turn/start'));
});
