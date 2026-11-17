import test from 'node:test';
import assert from 'node:assert/strict';
import { plainUpdateError, plainError } from '../server/update-messages.mjs';

// #4853: an engine that was downloaded and checked, then could not be put in place, used to read "The update didn't finish and nothing was
// changed" with no reason. The reason is now said in plain words and the technical text stays in details.

const moved = new Error('the new engine could not be moved into place (Strata could not use C:\\Users\\x\\AppData\\Local\\Strata\\components\\.s\\ab12\\engine (PermissionError)); copying it did not work either (OSError)');
Object.assign(moved, { exitCode: 6 });

test('an engine that could not be put in place says so and does not blame the download', () => {
  const plain = plainUpdateError(moved);
  assert.match(plain.message, /downloaded and checked, but Windows or a security program would not let Strata put it in place/);
  assert.match(plain.message, /Wait a minute, then press Update engine in Settings/);
  assert.doesNotMatch(plain.message, /PermissionError|C:\\|exit|\.s\\/);
  assert.equal(plain.details, moved.message);
});

test('the unpack and the switch failures are not the generic sentence either', () => {
  for (const text of ['the package could not be unpacked: Strata could not use X (PermissionError)', 'the new engine was not accepted after the switch (a start would pick other); the previous engine is unchanged']) {
    const plain = plainUpdateError(new Error(text));
    assert.doesNotMatch(plain.message, /didn't finish and nothing was changed/);
    assert.match(plain.message, /put it in place/);
  }
});

test('a disk that is full while unpacking is a disk-space message, not a security-program one', () => {
  const plain = plainUpdateError(new Error('the package could not be unpacked: Strata could not use X: No space left on device (OSError)'));
  assert.match(plain.message, /not enough free disk space/);
});

test('a path that is too long says that', () => {
  const plain = plainUpdateError(new Error('the package could not be unpacked: a file path is longer than Windows allows (300 characters, the limit is 260): X'));
  assert.match(plain.message, /too deep for Windows/);
  assert.match(plain.next, /Details/);
});

test('the first-run card has no Settings page: the next step names its own Continue button', () => {
  const plain = plainUpdateError(moved, 'engine', 'Continue');
  assert.match(plain.next, /^Wait a minute, then press Continue\./);
  assert.doesNotMatch(plain.message, /Settings/);
  const generic = plainUpdateError(new Error('something unexpected'), 'engine', 'Continue');
  assert.equal(generic.next, 'Press Continue to try again.');
  assert.equal(plainError(new Error('app-1 is serving.'), 'engine', 'Continue').message, 'A model is running, so nothing was changed. Unload the model in Models, then press Continue.');
  assert.match(plainUpdateError(new Error('something unexpected')).next, /Update engine in Settings/);   // elsewhere unchanged
});
