// #5574: with nothing on Strata's 8095 and Ollama on 11434, the host adopted Ollama (its /v1/models answered with 12 ids) and then asked it
// for the placeholder model "local": every chat failed with 404 "model 'local' not found". The chat now names a model the server listed
// (the first chat model: not an embedding, reranker or speech model), and a server that lists none is not adopted.
import test from 'node:test';
import assert from 'node:assert/strict';

process.env.STRATA_REAL_ENGINE = '1';
process.env.STRATA_SERVER_URL = 'http://127.0.0.1:8095/v1';
const { chatModelId, detectBackend, streamChat } = await import('../server/inference.mjs');

const OLLAMA = ['qwen3-embedding:8b', 'nimble-v3:latest', 'dengcao/Qwen3-Reranker-8B:Q8_0', 'bge-m3:latest', 'gemma4:31b'];

function fakeFetch(models) {
  const sent = [];
  const fetch = async (url, init = {}) => {
    const u = String(url);
    if (u.startsWith('http://127.0.0.1:11434/v1/models')) return new Response(JSON.stringify({ object: 'list', data: models.map(id => ({ id, object: 'model' })) }), { status: 200 });
    if (u.startsWith('http://127.0.0.1:11434/v1/chat/completions')) {
      sent.push(JSON.parse(init.body));
      const body = 'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n';
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } });
    }
    throw new TypeError('fetch failed');   // nothing else listens
  };
  return { fetch, sent };
}

test('the chat model is one the server listed: the first that is not an embedding, reranker or speech model', () => {
  assert.equal(chatModelId(OLLAMA), 'nimble-v3:latest');
  assert.equal(chatModelId(['qwen3-embedding:latest', 'bge-m3:latest']), 'qwen3-embedding:latest', 'only non-chat models: the first, as listed');
  assert.equal(chatModelId([]), null);
  assert.equal(chatModelId(undefined), null);
  assert.equal(chatModelId(['whisper-large-v3', 'llama3.2:3b']), 'llama3.2:3b');
});

test('an adopted Ollama is asked for a model it listed, never "local"', async t => {
  const real = globalThis.fetch;
  const f = fakeFetch(OLLAMA);
  globalThis.fetch = f.fetch;
  t.after(() => { globalThis.fetch = real; });
  const be = await detectBackend({ force: true });
  assert.equal(be.kind, 'openai-compatible');
  assert.equal(be.name, 'Ollama');
  await streamChat({ message: 'Say hi in one word.', onFrame() {} });
  assert.equal(f.sent.length, 1);
  assert.equal(f.sent[0].model, 'nimble-v3:latest');
});

test('a server that lists no model is not adopted', async t => {
  const real = globalThis.fetch;
  globalThis.fetch = fakeFetch([]).fetch;
  t.after(() => { globalThis.fetch = real; });
  const be = await detectBackend({ force: true });
  assert.notEqual(be.kind, 'openai-compatible');
});
