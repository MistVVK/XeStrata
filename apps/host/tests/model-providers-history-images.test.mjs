import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createModelProviders } from '../server/model-providers.mjs';
const PNG = 'data:image/png;base64,iVBORw0KGgo=';
const WEBP = 'data:image/webp;base64,UklGRg==';
const text = value => ({ type: 'text', text: value });
const image = url => ({ type: 'image_url', image_url: { url } });
function setup(t, { protocol = 'openai-chat', vision = true } = {}) {
  const directory = mkdtempSync(join(tmpdir(), 'strata-provider-history-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const requests = [];
  const fetchImpl = async (url, options) => {
    requests.push({ url, body: JSON.parse(options.body) });
    const events = protocol === 'anthropic'
      ? [{ type: 'content_block_delta', delta: { text: 'ok' } }, { type: 'message_stop' }]
      : protocol === 'openai-responses'
        ? [{ type: 'response.output_text.delta', delta: 'ok' }, { type: 'response.completed', response: {} }]
        : [{ choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: null }] }, { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }];
    return new Response(events.map(e => `data: ${JSON.stringify(e)}\n\n`).join(''), { headers: { 'content-type': 'text/event-stream' } });
  };
  const service = createModelProviders({ directory, fetchImpl });
  const connection = service.save({ kind: 'custom', name: 'Local API', baseUrl: 'http://127.0.0.1:9/v1', protocol,
    models: ['model-a', 'model-b'].map(id => ({ id, vision })) }).connections[0];
  const send = (options = {}) => service.stream({ selection: { connectionId: connection.id, modelId: 'model-a' }, message: 'What color was it?', onFrame: () => {}, ...options });
  return { service, connection, send, requests };
}

test('UI-format image history survives model changes without a current attachment', async t => {
  const { service, connection, requests } = setup(t);
  const history = [{ role: 'user', content: [text('What is in this picture?'), image(PNG)] }, { role: 'assistant', content: 'A red shape.' }];
  const original = structuredClone(history);
  for (const modelId of ['model-a', 'model-b', 'model-a']) {
    const frames = [];
    await service.stream({ selection: { connectionId: connection.id, modelId }, message: 'Which shape was red?', history, images: [], onFrame: frame => frames.push(frame) });
    const body = requests.at(-1).body;
    assert.equal(body.model, modelId);
    assert.deepEqual(body.messages, [...history, { role: 'user', content: 'Which shape was red?' }]);
    assert.equal(frames.at(-1).type, 'done');
  }
  assert.equal(requests.length, 3);
  assert.deepEqual(history, original, 'normalization must not mutate stored UI messages');
});

test('history and current pictures retain order and do not duplicate pixels', async t => {
  const { send, requests } = setup(t);
  await send({ history: [{ role: 'user', content: [text('Earlier question'), image(PNG), text('Earlier detail')] }], images: [{ url: WEBP }] });
  const messages = requests[0].body.messages;
  assert.deepEqual(messages[0].content, [text('Earlier question'), image(PNG), text('Earlier detail')]);
  assert.deepEqual(messages[1].content, [text('What color was it?'), image(WEBP)]);
});

test('only supported blocks and user image roles are forwarded; the question remains', async t => {
  const { send, requests } = setup(t);
  await send({ history: [null, { role: 'system', content: 'Not a user-supplied system instruction' }, { role: 'tool', content: 'ignored' },
    { role: 'user', content: [null, 3, [], { type: 'text', text: 7 }, { type: 'input_text', text: 'unsupported' },
      { type: 'text', text: 'Keep this question', unexpected: 'discard' }, { type: 'image_url', image_url: { url: PNG, unexpected: 'discard' }, unexpected: true }] },
    { role: 'assistant', content: [text('Keep this answer'), image('file:///not-forwarded')] }, { role: 'user', content: [{ type: 'audio', data: 'discard' }] }] });
  assert.deepEqual(requests[0].body.messages.slice(0, -1), [
    { role: 'user', content: [text('Keep this question'), image(PNG)] }, { role: 'assistant', content: 'Keep this answer' }]);
});

test('explicit text-only models keep historical questions without sending pictures', async t => {
  const { send, requests } = setup(t, { vision: false });
  await send({ history: [{ role: 'user', content: [text('Question'), image(PNG), text('Extra detail')] }] });
  assert.deepEqual(requests[0].body.messages[0], { role: 'user', content: 'Question\nExtra detail' });
});

test('unknown vision retains the existing pictures-on default', async t => {
  const { send, requests } = setup(t, { vision: null });
  await send({ history: [{ role: 'user', content: [text('Question'), image(PNG)] }] });
  assert.deepEqual(requests[0].body.messages[0].content, [text('Question'), image(PNG)]);
});

for (const protocol of ['openai-responses', 'anthropic']) test(`${protocol} keeps array-history text without image blocks`, async t => {
  const { send, requests } = setup(t, { protocol });
  await send({ history: [{ role: 'user', content: [text('Question'), image(PNG), text('Extra detail')] }] });
  assert.deepEqual((requests[0].body.messages ?? requests[0].body.input)[0], { role: 'user', content: 'Question\nExtra detail' });
});

for (const protocol of ['openai-chat', 'openai-responses', 'anthropic']) test(`${protocol} string history keeps exact content, roles and the last 40 entries`, async t => {
  const { send, requests } = setup(t, { protocol });
  const history = Array.from({ length: 45 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: i === 44 ? '' : `  unchanged ${i}\n` }));
  history.splice(8, 0, { role: 'system', content: 'ignored' }, { role: 'user', content: 12 });
  await send({ history });
  const expected = history.filter(m => ['user', 'assistant'].includes(m.role) && typeof m.content === 'string').slice(-40).map(m => ({ role: m.role, content: m.content }));
  assert.deepEqual((requests[0].body.messages ?? requests[0].body.input).slice(0, -1), expected);
});

test('malformed, remote and unsupported historical image URLs fail before a request', async t => {
  const { send, requests } = setup(t);
  for (const block of [{ type: 'image_url' }, { type: 'image_url', image_url: null }, image('file:///secret'), image('https://example.com/image.png'), image('data:image/gif;base64,R0lG'), image('data:image/png;base64,%%%')]) {
    await assert.rejects(send({ history: [{ role: 'user', content: [text('Keep the question'), block] }] }), /Pictures must be attached PNG, JPEG or WebP files/);
  }
  assert.equal(requests.length, 0);
});

test('historical image size and combined current/history count remain bounded', async t => {
  const { send, requests } = setup(t);
  const tooLarge = 'data:image/png;base64,' + Buffer.alloc(6 * 1024 * 1024 + 1).toString('base64');
  await assert.rejects(send({ history: [{ role: 'user', content: [text('Question'), image(tooLarge)] }] }), /smaller than 6 MB/);
  await assert.rejects(send({ history: [{ role: 'user', content: [text('Question'), image(PNG), image(PNG)] }], images: [{ url: PNG }, { url: PNG }, { url: PNG }] }), /up to four pictures/);
  assert.equal(requests.length, 0);
});
