// #5856: "Run 10 steps on your computer?" showed only the start of what would run: Details held the card's preview, cut at 600 characters.
// Details now shows the call's own full command, every byte, in a box that scrolls and takes the keyboard focus.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { presentTool } from '../server/agent/events.mjs';

const built = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/approvals.ts', import.meta.url))], bundle: true, write: false, platform: 'node', format: 'esm' });
const { approvalDetail } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);

test('ten long steps, quotes and newlines: Details shows every byte the agent asks to run', () => {
  const steps = Array.from({ length: 10 }, (_, i) => `Copy-Item -Path "C:\\Projects\\very long folder name ${i}\\with 'quotes' and spaces\\file${i}.txt" -Destination "D:\\Backup\\${i}"`);
  const command = steps.join(' ; \n') + '\n';
  const card = presentTool({ callId: 'c1', name: 'pwsh', args: { command, description: 'copy files' } });
  assert.equal(card.scope, 'shell');
  assert.ok(card.preview.length < command.length, 'the card\'s own summary stays short');
  assert.equal(approvalDetail(card), command, 'Details: the exact command');
  for (const s of steps) assert.ok(approvalDetail(card).includes(s));
});

test('other actions keep their preview', () => {
  assert.equal(approvalDetail({ scope: 'write', args: { file_path: 'a.txt', content: 'x'.repeat(2000) }, preview: 'a.txt' }), 'a.txt');
  assert.equal(approvalDetail({ scope: 'shell', args: {}, preview: 'p' }), 'p', 'no command: the preview');
  assert.equal(approvalDetail({ scope: 'connected', args: { command: 'not a shell' }, preview: 'tool' }), 'tool');
});

test('the card shows it in a box that scrolls and takes the keyboard focus', () => {
  const card = readFileSync(fileURLToPath(new URL('../src/components/ApprovalCard.tsx', import.meta.url)), 'utf8');
  assert.match(card, /<summary><Icon name="chevron" size=\{11\}\/>\{tr\("Details"\)\}<\/summary><pre tabIndex=\{0\}>\{approvalDetail\(t\)\}<\/pre>/);
  const css = readFileSync(fileURLToPath(new URL('../src/styles/app.css', import.meta.url)), 'utf8');
  assert.match(css, /\.askcard__details pre \{[^}]*white-space: pre-wrap;[^}]*max-height: 40vh; overflow: auto; \}/);
});
