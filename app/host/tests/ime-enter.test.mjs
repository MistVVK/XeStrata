// #5071: the Enter that commits a Japanese IME conversion must not send the message; the next Enter sends exactly once.
// The engines deliver the commit in different orders. Each order is replayed here as the event list it produces, through the same guard
// the composer uses. VERIFIED BY REPLAY ONLY: the sequences are the documented behaviour of each engine, not captured from a real
// macOS WKWebView or WebKitGTK in this run (the report says so).
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { transform } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const { code } = await transform(readFileSync(resolve(root, 'src/lib/ime-enter.ts'), 'utf8'), { loader: 'ts', format: 'esm' });
const { createImeGuard, IME_COMMIT_WINDOW_MS } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

// event constructors: t is the event's timeStamp in ms
const down = (t, key, o = {}) => ({ type: 'keydown', t, key, keyCode: o.keyCode ?? (key === 'Enter' ? 13 : key.toUpperCase().charCodeAt(0)), isComposing: o.isComposing ?? false, shift: !!o.shift, ctrl: !!o.ctrl, meta: !!o.meta, alt: !!o.alt });
const ime = (t, key, isComposing = true) => down(t, key, { keyCode: 229, isComposing });
const start = t => ({ type: 'compositionstart', t });
const end = t => ({ type: 'compositionend', t });

/** The composer's keydown path, reduced to what matters: the guard first, then the send binding (plain Enter by default), Shift+Enter is a newline. */
function replay(events) {
  const guard = createImeGuard();
  const out = { sends: 0, newlines: 0, imeKeys: 0, shortcuts: [] };
  for (const e of events) {
    if (e.type === 'compositionstart') guard.compositionStart();
    else if (e.type === 'compositionend') guard.compositionEnd(e.t);
    else if (e.type === 'blur') guard.reset();
    else if (e.type === 'keydown') {
      if (guard.keyDown({ key: e.key, isComposing: e.isComposing, keyCode: e.keyCode }, e.t)) { out.imeKeys++; continue; }
      const plain = !e.shift && !e.ctrl && !e.meta && !e.alt;
      if (e.key === 'Enter' && plain) out.sends++;
      else if (e.key === 'Enter' && e.shift) out.newlines++;
      else out.shortcuts.push(e.key);
    }
  }
  return out;
}

test('Chrome / Edge WebView2: keydown(229, composing) comes before compositionend; the next Enter sends once', () => {
  const out = replay([
    start(0), ime(10, 'k'), ime(60, 'a'), ime(120, ' '), ime(400, 'Enter'),   // Enter commits the conversion
    end(402),
    down(900, 'Enter'),                                                         // the second Enter
  ]);
  assert.equal(out.sends, 1);
  assert.equal(out.imeKeys, 4);
});

test('Safari / WKWebView: compositionend comes BEFORE the committing keydown, which has isComposing false and keyCode 229', () => {
  const out = replay([
    start(0), ime(10, 'k'), ime(60, 'a'), ime(120, ' '),
    end(400),
    ime(401, 'Enter', false),                                                   // the WebKit quirk: 229, but isComposing already false
    down(900, 'Enter'),
  ]);
  assert.equal(out.sends, 1, 'exactly one send, from the second Enter');
  assert.equal(out.imeKeys, 4);
});

test('Safari without the 229 (older WebKit): compositionend then a plain Enter in the same tick is still the commit', () => {
  const out = replay([start(0), ime(10, 'k'), end(400), down(401, 'Enter'), down(900, 'Enter')]);
  assert.equal(out.sends, 1);
});

test('WebKitGTK (IBus): compositionend then a plain Enter keydown with keyCode 13 right after is the commit', () => {
  const out = replay([start(0), ime(10, 'k'), ime(60, 'a'), end(300), down(303, 'Enter'), down(800, 'Enter')]);
  assert.equal(out.sends, 1);
  assert.equal(out.imeKeys, 3);
});

test('the window is a few milliseconds: an Enter after it is the person sending', () => {
  const edge = IME_COMMIT_WINDOW_MS;
  assert.equal(replay([end(1000), down(1000 + edge, 'Enter')]).sends, 0);
  assert.equal(replay([end(1000), down(1000 + edge + 1, 'Enter')]).sends, 1);
  assert.equal(replay([end(1000), down(1000 + 300, 'Enter')]).sends, 1);
});

test('the flag swallows exactly one Enter: the commit, then the next sends', () => {
  const out = replay([end(1000), down(1002, 'Enter'), down(1010, 'Enter')]);
  assert.equal(out.sends, 1, 'the second Enter, even 8 ms after, is a real one: the flag was used up by the first');
});

test('any other key uses the flag up, so it can never swallow a later Enter', () => {
  const out = replay([end(1000), down(1001, 'a'), down(1003, 'Enter')]);
  assert.equal(out.sends, 1);
  assert.deepEqual(out.shortcuts, ['a']);
});

test('Latin input: Enter sends once, Shift+Enter is a newline, nothing is swallowed', () => {
  const out = replay([down(0, 'h'), down(50, 'i'), down(100, 'Enter', { shift: true }), down(150, 'o'), down(200, 'Enter')]);
  assert.equal(out.sends, 1);
  assert.equal(out.newlines, 1);
  assert.equal(out.imeKeys, 0);
});

test('a conversion cancelled with Escape sends nothing, and the Enter after it sends once (both orders)', () => {
  const chrome = replay([start(0), ime(10, 'k'), ime(60, 'a'), ime(100, 'Escape'), end(102), down(700, 'Enter')]);
  assert.equal(chrome.sends, 1);
  const safari = replay([start(0), ime(10, 'k'), ime(60, 'a'), end(100), ime(101, 'Escape', false), down(700, 'Enter')]);
  assert.equal(safari.sends, 1);
  assert.equal(safari.imeKeys, 3);
});

test('a candidate chosen with the arrow keys and a space belongs to the IME in both orders', () => {
  const out = replay([start(0), ime(10, 'ArrowDown'), ime(20, 'ArrowDown'), ime(30, ' '), ime(40, 'Enter'), end(41), down(600, 'Enter')]);
  assert.equal(out.sends, 1);
  assert.deepEqual(out.shortcuts, []);
});

test('keyboard shortcuts are not touched, with or without a conversion just ended', () => {
  const out = replay([down(0, 'k', { ctrl: true }), down(10, '3', { alt: true }), down(20, 'Enter', { meta: true, keyCode: 13 }), down(30, 'ArrowUp')]);
  assert.equal(out.imeKeys, 0);
  assert.deepEqual(out.shortcuts, ['k', '3', 'Enter', 'ArrowUp']);
  // right after a conversion, Ctrl+K is still the shortcut (only an Enter can be the commit)
  const after = replay([end(100), down(102, 'k', { ctrl: true })]);
  assert.equal(after.imeKeys, 0);
  assert.deepEqual(after.shortcuts, ['k']);
});

test('a conversion that never ends (focus left) cannot leave the composer unable to send', () => {
  const out = replay([start(0), ime(10, 'k'), { type: 'blur', t: 20 }, down(900, 'Enter')]);
  assert.equal(out.sends, 1);
  // and with no blur at all: a real Enter (keyCode 13, not composing) is never taken for the IME, whatever started before
  assert.equal(replay([start(0), down(900, 'Enter')]).sends, 1);
});

test('the composer is wired to the guard, before the command menu and the send binding', () => {
  const chat = readFileSync(resolve(root, 'src/screens/Chat.tsx'), 'utf8');
  assert.match(chat, /const ime = useRef\(createImeGuard\(\)\)/);
  assert.match(chat, /onCompositionStart=\{\(\) => ime\.current\.compositionStart\(\)\}/);
  assert.match(chat, /onCompositionEnd=\{\(e\) => ime\.current\.compositionEnd\(e\.timeStamp\)\}/);
  assert.match(chat, /onBlur=\{\(\) => ime\.current\.reset\(\)\}/);
  const at = chat.indexOf('ime.current.keyDown(');
  assert.ok(at > 0);
  assert.ok(at < chat.indexOf("// The command menu owns the keyboard while it is open."), 'the guard runs before the command menu');
  assert.ok(at < chat.indexOf('composerSends(e)'));
  assert.doesNotMatch(chat, /!e\.nativeEvent\.isComposing && composerSends/, 'the old single check is gone');
});
