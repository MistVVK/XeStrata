// #4926: "Use a local GGUF" asked setup for `--vision no` unless the chosen folder held the pinned picture file itself, so a person whose picture
// reader sits in Strata's models folder lost picture reading on every switch to own files.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { transform } from 'esbuild';

const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const load = async name => {
  const { code } = await transform(src(name), { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
};

test('picture reading stays on for own files when a reader is already on this computer', async () => {
  const { ownFilesVision } = await load('lib/own-files-vision.ts');
  const none = [];
  // the reporter's case: the folder has no mmproj, the shared reader is in the models folder (an installed model has its picture files)
  assert.equal(ownFilesVision(none, true, [{ visionPrepared: true }]), 'yes');
  assert.equal(ownFilesVision(undefined, true, [{ visionPrepared: false }, { visionPrepared: true }]), 'yes');
  // the folder holds the pinned reader, or one of the right kind with another name (the checker says Strata downloads the official one)
  assert.equal(ownFilesVision([{ status: 'ok', pinned: true }], true, []), 'yes');
  assert.equal(ownFilesVision([{ status: 'other_name', pinned: false }], true, []), 'yes');
  // nothing to read pictures with: off, as before (no surprise download of a reader for a person who has none)
  assert.equal(ownFilesVision(none, true, []), 'no');
  assert.equal(ownFilesVision(none, true, undefined), 'no');
  assert.equal(ownFilesVision([{ status: 'unsupported' }, { status: 'not_a_projector' }, { status: 'unreadable' }], true, [{ visionPrepared: false }]), 'no');
  // an engine without a picture reader: off whatever the files say
  assert.equal(ownFilesVision([{ status: 'ok', pinned: true }], false, [{ visionPrepared: true }]), 'no');
  assert.equal(ownFilesVision([{ status: 'ok', pinned: true }], undefined, [{ visionPrepared: true }]), 'no');
});

test('the own-files page asks for its vision value from that rule, not from the folder alone', () => {
  const page = src('components/NativeLocalModel.tsx');
  assert.match(page, /import \{ ownFilesVision \} from '\.\.\/lib\/own-files-vision';/);
  assert.match(page, /vision: ownFilesVision\(checked\.preview\.projectors, native\?\.visionReader, native\?\.installed\?\.models\)/);
  assert.doesNotMatch(page, /p\.status === 'ok' && p\.pinned\) && native\?\.visionReader \? 'yes' : 'no'/);
});
