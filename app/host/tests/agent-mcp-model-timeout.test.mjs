// #5788: the MCP servers set up for a model (its settings, MCP Tools tab) are the ones the desktop agent connects to, but the model's
// "Tool timeout (seconds)" (mcp_timeout_s) reached only the engine's own MCP client: the Harness's client cut every call at its 60 s
// default. The model's servers now take that setting unless a server names its own time.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildProfile } from '../server/agent/profile.mjs';

const rowsOf = profile => {
  const out = [];
  const walk = v => { if (Array.isArray(v)) v.forEach(walk); else if (v && typeof v === 'object') { if (v.name === '@deepseek-ai/dsh-mcp-client' && v.config) out.push(v); else Object.values(v).forEach(walk); } };
  walk(JSON.parse(profile.patch.split('\n').slice(1).join('\n')));   // the patch is a comment line, then JSON (as entriesOf in agent-tools-settings)
  return Object.fromEntries(out.map(r => [r.config.serverName, r.config]));
};
const local = extra => [{ id: 'a', modelId: 'm', baseUrl: 'http://x', context: 32768, mcpServers: { ue5: { url: 'http://127.0.0.1:8000/mcp' }, own: { command: 'node', args: ['s.js'], timeout_s: 30 } }, ...extra }];

test('the model\'s Tool timeout reaches its MCP servers in the agent; a server\'s own time wins', () => {
  const rows = rowsOf(buildProfile({ local: local({ mcpTimeoutS: 900 }), connections: [] }));
  assert.equal(rows.ue5.toolCallTimeoutMs, 900000);
  assert.equal(rows.own.toolCallTimeoutMs, 30000);
});

test('without the setting the client keeps its own default', () => {
  const rows = rowsOf(buildProfile({ local: local({}), connections: [] }));
  assert.equal(rows.ue5.toolCallTimeoutMs, undefined);
  assert.equal(rows.own.toolCallTimeoutMs, 30000);
});

test('the host hands the model\'s setting to the agent', () => {
  // a checkout with CRLF line ends (Git for Windows' default, GitHub's Windows runner) must read the same as one with LF
  const lf = rel => readFileSync(new URL(rel, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
  const index = lf('../server/index.mjs');
  assert.match(index, /mcpServers: i\.mcpServers,\n[^\n]*\n\s*mcpTimeoutS: i\.mcpTimeoutS \?\? null,/);
  const control = lf('../server/native-control.mjs');
  assert.match(control, /mcpServers:cfg\.mcp_servers\?\?cfg\.mcpServers\?\?\{\},mcpTimeoutS:cfg\.mcp_timeout_s\?\?null,/);
});
