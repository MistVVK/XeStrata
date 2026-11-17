/** A streamed token changes one message. The others must not render again, or a long chat does work (and makes garbage) for every
 *  token in proportion to its whole history: measured 0.7 s of main-thread time for a 10-turn reply on a new chat and 1.9 s after ten
 *  replies, peak heap 95 MB instead of 50 MB (scripts/chat-growth.mjs). Source guards: the DOM is not available in this suite. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const chat = readFileSync(resolve(import.meta.dirname, '../src/screens/Chat.tsx'), 'utf8');

test('the message row is a memoised component', () => {
  assert.match(chat, /import \{[^}]*\bmemo\b[^}]*\} from 'react'/);
  assert.match(chat, /const Message = memo\(MessageView\)/);
});

test('the list passes handlers whose identity never changes', () => {
  assert.match(chat, /const messageHandlers = useMemo\(\(\) => \(\{[\s\S]*?\}\), \[\]\)/);
  for (const name of ['onEditPrompt', 'onPreview', 'onOpenTimeSkip', 'onOpenLink'])
    assert.match(chat, new RegExp(`${name}=\{messageHandlers\.${name}\}`), name);
  assert.doesNotMatch(chat, /onEditPrompt=\{editPrompt\}/);
});

const store = readFileSync(resolve(import.meta.dirname, '../src/lib/store.ts'), 'utf8');

test('streamed text and reasoning are applied in groups, and everything else applies what is waiting first', () => {
  assert.match(store, /const queueText = \(kind: 'reasoning' \| 'delta', text: string\)/);
  assert.match(store, /waitingTimer \?\?= setTimeout\(flushWaiting, 90\)/);
  // the order of frames the person sees is the order written: a frame of any other kind flushes first
  assert.match(store, /if \(type === 'delta' \|\| type === 'reasoning'\) \{ queueText\(type, String\(frame\.text \?\? ''\)\); return; \}\s*flushWaiting\(\);/);
  // a failure or the end of the turn never leaves text behind
  assert.match(store, /\} catch \(err\) \{\s*flushWaiting\(\);/);
  assert.match(store, /\} finally \{\s*flushWaiting\(\);/);
});
