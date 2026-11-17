// #5617: a question from the agent (ask_user_question) waited for the person with no notification, although an approval, which waits the same
// way, notifies when "Notify me when Strata needs an answer" is on. The question now gets that notification: the approval's own title, no body.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { build } from 'esbuild';

const compiled = await build({ stdin: { contents: "export * from './src/lib/attention'; export * from './src/lib/i18n';", resolveDir: process.cwd() }, bundle: true, write: false, format: 'esm', platform: 'node', loader: { '.json': 'json' } });
const A = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
const src = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8').replace(/\r\n/g, '\n');

test('the question notice is the approval notice\'s title with no new sentence, under the same opt-in', async () => {
  assert.deepEqual(A.questionNotice(), { title: 'Strata needs your answer', body: '' });
  assert.equal(A.questionNotice().title, A.approvalNotice('x').title);
  assert.equal(await A.questionAsked(false, false), false, 'off unless the person turned it on');
  assert.equal(await A.questionAsked(true, true), false, 'the person is looking at Strata');
});

test('a new question notifies once; the same question again (a reload, a repeat frame) does not', () => {
  const store = src('lib/store.ts');
  assert.match(store, /import \{ approvalNeeded, questionAsked, workFinished \} from '\.\/attention';/);
  const at = store.indexOf('export function addAgentAsk(');
  const body = store.slice(at, store.indexOf('\n}\n', at));
  assert.match(body, /if \(ask && loopScheduler\.isRunning\(sessionId\)\) \{[^\n]*return; \}/, 'a /loop question is skipped before anything else');
  assert.match(body, /const known = \(state\.agentAsks\[sessionId\] \?\? \[\]\)\.some\(a => a\.callId === ask\.callId\);/);
  assert.match(body, /if \(!known\) void questionAsked\(state\.approvalNotify\);/);
});
