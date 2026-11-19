// Lane GUI-TUI (0.2.17): #6278, the desktop app next to a model started from the Terminal.
import test from 'node:test';
import assert from 'node:assert/strict';
import { parsePortNotice } from '../server/shell-prefs.mjs';

test('the port notice says when the holder of the wanted port is a model server started from the Terminal (#6278)', () => {
  assert.deepEqual(parsePortNotice('busy:8080:51234:terminal'), { wanted: 8080, used: 51234, terminal: true });
  // the old form is unchanged
  assert.deepEqual(parsePortNotice('busy:47831:51234'), { wanted: 47831, used: 51234 });
  assert.equal(parsePortNotice('busy:8080:51234:other'), null);
  assert.equal(parsePortNotice('busy:8080:51234:terminal:x'), null);
});
