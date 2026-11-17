import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { transform } from 'esbuild';

const { code } = await transform(await readFile(new URL('../src/lib/chat-attach-file.ts', import.meta.url), 'utf8'), { loader: 'ts', format: 'esm' });
const { uploadChatFile, FILE_ACCEPT } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('the file dialog lists PDFs and text on Linux: the types sit next to the extensions (#3838)', () => {
  for (const t of ['image/*', 'application/pdf', 'text/*', '.pdf', '.md']) assert.ok(FILE_ACCEPT.split(',').includes(t), t);
});
test('a PDF is sent to the chat folder through the upload route and named back', async () => {
  let seen;
  const fake = async (url, init) => { seen = { url, init }; return { ok: true, json: async () => ({ files: [{ name: 'report-2.pdf', attached: true }] }) }; };
  const out = await uploadChatFile(new File(['%PDF-1.4'], 'report.pdf', { type: 'application/pdf' }), 's 1', fake);
  assert.equal(out.name, 'report-2.pdf');
  assert.match(seen.url, /\/api\/chat\/upload\?sessionId=s%201&name=report\.pdf$/);
  assert.equal(seen.init.method, 'POST'); assert.equal(seen.init.headers['content-type'], 'application/pdf');
});
test('a refused upload says why', async () => {
  await assert.rejects(uploadChatFile(new File(['x'], 'a.pdf'), 's', async () => ({ ok: false, json: async () => ({ error: '"a.pdf" is bigger than 12 MB.' }) })), /bigger than 12 MB/);
});
