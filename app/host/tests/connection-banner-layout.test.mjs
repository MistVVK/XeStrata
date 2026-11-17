// Packaged-app walkthrough (GUI-VIS-1117 A62-A65): when the host process died the pink banner "Strata isn't connected to its engine ..." was squeezed into a ~130 px column (one word
// per line, Retry drawn over the text). The "Connection details" disclosure has flex-basis 100% to take its own row, but the wrap rule was written for `.connectionbar`,
// an old class name; the banner is `.connbar`, so it never wrapped and the 100% item pushed the text and the button into the rest.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../src/styles/app.css', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const app = readFileSync(new URL('../src/app/App.tsx', import.meta.url), 'utf8').replace(/\r\n/g, '\n');

test('the banner is the class the wrap rule names, and the disclosure takes its own row', () => {
  assert.ok(app.includes('<div className="connbar" data-state={connection} role="status">'), 'the banner class');
  assert.ok(/\.connectionbar,\n\.connbar \{\n  flex-wrap: wrap;\n\}/.test(css), 'the wrap rule covers .connbar');
  assert.ok(/\.connection-details \{\n  flex-basis: 100%;/.test(css), 'the disclosure takes a row of its own');
});

test('the banner text has a width of its own to keep (it wraps the button below instead of being squeezed)', () => {
  assert.ok(/\.connbar__text \{\n  flex: 1 1 220px;\n  min-width: 0;\n\}/.test(css));
});
