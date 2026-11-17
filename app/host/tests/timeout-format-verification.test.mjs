import test from 'node:test';
import assert from 'node:assert/strict';
import { mcpRows } from '../server/agent/profile.mjs';

test('Format 1: server with timeoutSec (e.g. from Strata custom tools settings)', () => {
  const env = {};
  const servers = {
    'my-mcp': {
      command: 'node',
      args: ['server.js'],
      timeoutSec: 300,
    },
  };
  const rows = mcpRows(servers, env);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].name, '@deepseek-ai/dsh-mcp-client');
  assert.equal(rows[0].config.serverName, 'my-mcp');
  assert.equal(rows[0].config.toolCallTimeoutMs, 300000);
});

test('Format 2: server with timeout_s (e.g. from strata-<model>.json or mcp_servers.json)', () => {
  const env = {};
  const servers = {
    'relay-mcp': {
      command: 'node',
      args: ['relay.js'],
      timeout_s: 180,
    },
  };
  const rows = mcpRows(servers, env);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].config.serverName, 'relay-mcp');
  assert.equal(rows[0].config.toolCallTimeoutMs, 180000);
});

test('Format 3: server with timeout (e.g. Claude Desktop style)', () => {
  const env = {};
  const servers = {
    'claude-style-mcp': {
      command: 'node',
      args: ['claude.js'],
      timeout: 240,
    },
  };
  const rows = mcpRows(servers, env);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].config.serverName, 'claude-style-mcp');
  assert.equal(rows[0].config.toolCallTimeoutMs, 240000);
});

test('Format 4: server with direct toolCallTimeoutMs in milliseconds', () => {
  const env = {};
  const servers = {
    'direct-ms-mcp': {
      command: 'node',
      args: ['direct.js'],
      toolCallTimeoutMs: 450000,
    },
  };
  const rows = mcpRows(servers, env);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].config.serverName, 'direct-ms-mcp');
  assert.equal(rows[0].config.toolCallTimeoutMs, 450000);
});

test('Format 5: Streamable HTTP server with timeoutSec', () => {
  const env = {};
  const servers = {
    'remote-mcp': {
      url: 'https://mcp.example.com/sse',
      timeoutSec: 600,
    },
  };
  const rows = mcpRows(servers, env);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].config.serverName, 'remote-mcp');
  assert.equal(rows[0].config.transport, 'streamable-http');
  assert.equal(rows[0].config.toolCallTimeoutMs, 600000);
});

test('Format 6: environment variable STRATA_MCP_TIMEOUT_MS fallback', () => {
  const originalEnv = process.env.STRATA_MCP_TIMEOUT_MS;
  try {
    process.env.STRATA_MCP_TIMEOUT_MS = '500000';
    const env = {};
    const servers = {
      'fallback-mcp': {
        command: 'node',
        args: ['server.js'],
      },
    };
    const rows = mcpRows(servers, env);
    assert.equal(rows.length, 1);
    assert.equal(rows[0].config.toolCallTimeoutMs, 500000);
  } finally {
    if (originalEnv === undefined) delete process.env.STRATA_MCP_TIMEOUT_MS;
    else process.env.STRATA_MCP_TIMEOUT_MS = originalEnv;
  }
});

test('Format 7: explicit server timeout overrides environment variable', () => {
  const originalEnv = process.env.STRATA_MCP_TIMEOUT_MS;
  try {
    process.env.STRATA_MCP_TIMEOUT_MS = '500000';
    const env = {};
    const servers = {
      'override-mcp': {
        command: 'node',
        args: ['server.js'],
        timeoutSec: 120, // 120s should win over 500s env var
      },
    };
    const rows = mcpRows(servers, env);
    assert.equal(rows.length, 1);
    assert.equal(rows[0].config.toolCallTimeoutMs, 120000);
  } finally {
    if (originalEnv === undefined) delete process.env.STRATA_MCP_TIMEOUT_MS;
    else process.env.STRATA_MCP_TIMEOUT_MS = originalEnv;
  }
});

test('Format 8: string numeric values are safely parsed', () => {
  const env = {};
  const servers = {
    'string-timeout': {
      command: 'node',
      args: ['server.js'],
      timeoutSec: '150',
    },
  };
  const rows = mcpRows(servers, env);
  assert.equal(rows[0].config.toolCallTimeoutMs, 150000);
});

test('Format 9: omitted timeout preserves undefined (client schema default applies)', () => {
  const originalEnv = process.env.STRATA_MCP_TIMEOUT_MS;
  delete process.env.STRATA_MCP_TIMEOUT_MS;
  try {
    const env = {};
    const servers = {
      'default-mcp': {
        command: 'node',
        args: ['server.js'],
      },
    };
    const rows = mcpRows(servers, env);
    assert.equal(rows.length, 1);
    assert.equal(rows[0].config.toolCallTimeoutMs, undefined);
  } finally {
    if (originalEnv !== undefined) process.env.STRATA_MCP_TIMEOUT_MS = originalEnv;
  }
});
