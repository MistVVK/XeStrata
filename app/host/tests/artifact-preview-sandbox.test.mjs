/** The side panel's page preview: a page that captures the mouse (a first-person game) may lock the pointer; the frame stays isolated. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const panel = readFileSync(new URL('../src/components/ArtifactPanel.tsx', import.meta.url), 'utf8');

test('the page preview frame allows scripts and pointer lock, nothing more', () => {
  const sandbox = /<iframe key=\{`\$\{current\.id\}-\$\{revision\}[^`]*`\}[^>]*\ssandbox="([^"]*)"/.exec(panel);
  assert.ok(sandbox, 'the page preview iframe has a sandbox attribute');
  assert.deepEqual(sandbox[1].split(/\s+/).sort(), ['allow-pointer-lock', 'allow-scripts']);
});
