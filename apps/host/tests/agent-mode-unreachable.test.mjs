// #5070: a turn whose first call to the agent failed with "fetch failed" (the agent was not answering) was shown as "Strata could not set
// this chat to Bypass mode", and nothing about it was in the host log. A call that never got an answer is now told apart from an answer that
// refused the mode: the chat says the assistant could not start (an existing sentence), the cause stays under Details and goes to the log.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import net from 'node:net';
import { agentUnreachable } from '../server/agent/agent-bridge.mjs';
import { GatewayError } from '../server/agent/gateway-client.mjs';

test('a call that got no answer is unreachable; an answer that refused is not', async () => {
  const s = net.createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const port = s.address().port; await new Promise(r => s.close(r));
  const refused = await fetch(`http://127.0.0.1:${port}/api/session/projections`, { method: 'POST' }).catch(e => e);
  assert.equal(agentUnreachable(refused), true, `a closed port: ${refused?.message} / ${refused?.cause?.code}`);
  assert.equal(agentUnreachable(Object.assign(new Error('timed out'), { name: 'TimeoutError' })), true);
  assert.equal(agentUnreachable(new GatewayError('not connected', { code: 'gateway/not-connected' })), true);
  assert.equal(agentUnreachable(new GatewayError('commands/execute: unknown command', { code: 'gateway/error' })), false);
  assert.equal(agentUnreachable(new Error('the session reports "plan" instead of "bypass"')), false);
});

test('applyMode words the two cases apart and logs the unreachable one', () => {
  const src = readFileSync(new URL('../server/agent/agent-bridge.mjs', import.meta.url), 'utf8');
  const apply = src.slice(src.indexOf('async function applyMode('), src.indexOf("if (mode === 'manual' && !runtime.gateChecked)"));
  assert.match(apply, /if \(agentUnreachable\(e\)\) \{[\s\S]*log\(`agent · session \$\{sessionId\}: the agent did not answer while the chat's mode was set[\s\S]*throw new AgentUnavailable\('The assistant could not start\. Try again in a moment\.', 'agent-unreachable', why\);/);
  assert.match(apply, /Strata could not set this chat to \$\{MODE_PRESETS\[mode\]\.name\} mode, so nothing ran\. Try again\./, 'a refused mode keeps its own sentence');
});
