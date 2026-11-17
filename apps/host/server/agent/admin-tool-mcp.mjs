#!/usr/bin/env node
/**
 * "Run as administrator" (#5968, Windows only, off unless the person turns it on in Settings -> Tools): the app's own tool server with
 * one tool, `run_as_administrator`. Bypass mode skips Strata's own approval card; it never gave Windows administrator rights, and the
 * agent had no way to ask for them. This tool asks Windows: every call shows the Windows consent prompt (UAC) for that one command,
 * nothing keeps the rights afterwards, and choosing No there means nothing ran. In Manual mode Strata's Approve/Deny card comes first
 * (plugins/strata-gate.mjs); Plan mode refuses the tool.
 *
 *   node admin-tool-mcp.mjs      MCP over stdio (newline-delimited JSON-RPC), like custom-tools-mcp.mjs
 *
 * The command never travels on a command line: it is written into a PowerShell script in a new folder under the temp folder, which
 * the elevated PowerShell runs. Its output and whether it reported errors (PowerShell's non-terminating errors included, so a refused
 * write is not reported as done) come back through two files in that folder. 2 minutes per call; 20,000 characters of output.
 */
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';

export const TOOL = 'run_as_administrator';
export const TIMEOUT_MS = 120_000;
export const MAX_OUTPUT = 20_000;
/** The Windows error for "the person chose No" at the consent prompt (ERROR_CANCELLED). */
export const CANCELLED = 1223;
/** The launcher's own code: the elevated PowerShell could not be started for another reason. */
export const NOT_STARTED = 1224;
/** The result text when the person chose No. events.mjs adds the person's line to the reply when a result starts with it. */
export const REFUSED = 'Refused: the person chose No at the Windows administrator prompt, so nothing ran.';

export const DESCRIPTION = 'Run one PowerShell command with Windows administrator rights, for a change that needs them (a machine-wide ' +
  'registry value, a service, a protected folder). Windows asks the person each time; if they choose No, nothing runs. Use it only when ' +
  'a normal command was refused for lack of rights, and say in `reason` why the command needs them.';
export const SCHEMA = { type: 'object', properties: {
  command: { type: 'string', description: 'The PowerShell command to run as administrator.' },
  reason: { type: 'string', description: 'One sentence: why this command needs administrator rights.' },
}, required: ['command', 'reason'], additionalProperties: false };

const psQuote = s => `'${String(s).replace(/'/g, "''")}'`;

/** The script the elevated PowerShell runs: the command, its output to `outFile`, and whether it reported errors to `statusFile`. */
export function innerScript(command, outFile, statusFile) {
  return [
    '$ErrorActionPreference = \'Continue\'',
    '$Error.Clear()',
    '$global:LASTEXITCODE = $null',
    `$text = & { ${command}\n} *>&1 | Out-String -Width 300`,
    '$exit = if ($global:LASTEXITCODE -ne $null) { [int]$global:LASTEXITCODE } else { 0 }',
    '$failed = ($Error.Count -gt 0) -or ($exit -ne 0)',
    `Set-Content -LiteralPath ${psQuote(outFile)} -Value $text -Encoding UTF8`,
    `Set-Content -LiteralPath ${psQuote(statusFile)} -Value (@{ failed = $failed; exit = $exit; errors = $Error.Count } | ConvertTo-Json -Compress) -Encoding UTF8`,
    'if ($failed) { exit 1 } else { exit 0 }',
    '',
  ].join('\r\n');
}

/** The script that asks Windows for the rights (Start-Process -Verb RunAs) and waits; exit CANCELLED when the person chose No. */
export function launcherScript(innerFile) {
  const ps = '$env:SystemRoot\\System32\\WindowsPowerShell\\v1.0\\powershell.exe';
  return [
    '$ErrorActionPreference = \'Stop\'',
    'try {',
    `  $p = Start-Process -FilePath "${ps}" -ArgumentList @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', ${psQuote(`"${innerFile}"`)}) -Verb RunAs -WindowStyle Hidden -Wait -PassThru`,
    '  exit $p.ExitCode',
    '} catch {',
    '  $e = $_.Exception',
    '  while ($e -and -not ($e -is [System.ComponentModel.Win32Exception])) { $e = $e.InnerException }',
    `  if ($e -and $e.NativeErrorCode -eq ${CANCELLED}) { exit ${CANCELLED} }`,
    '  [Console]::Error.WriteLine($_.Exception.Message)',
    `  exit ${NOT_STARTED}`,
    '}',
    '',
  ].join('\r\n');
}

const clip = text => text.length > MAX_OUTPUT ? `${text.slice(0, MAX_OUTPUT)}\n[cut at ${MAX_OUTPUT} characters]` : text;

/** What the model reads, from how the launcher ended and what the elevated command wrote. */
export function resultText({ code, timedOut = false, stderr = '', output = '', status = null }) {
  if (timedOut) return { ok: false, text: 'Still running after 2 minutes, so Strata stopped waiting. The command may still finish on its own.' };
  if (code === CANCELLED) return { ok: false, text: `${REFUSED} Do not try again unless the person asks for it.` };
  if (code === NOT_STARTED || status === null) return { ok: false, text: `Windows did not start the command as administrator. ${String(stderr).trim()}`.trim() };
  const body = clip(String(output).replace(/\s+$/, ''));
  if (status.failed) return { ok: false, text: `Ran as administrator, but the command reported errors (exit code ${status.exit}). Check the output before saying it worked.\n${body}` };
  return { ok: true, text: `Ran as administrator (exit code 0).${body ? `\n${body}` : ''}` };
}

/** One call: write the two scripts, ask Windows, wait at most TIMEOUT_MS, read the result. `run` is replaceable in tests. */
export async function runAsAdministrator(args, { platform = process.platform, run = runLauncher, timeoutMs = TIMEOUT_MS } = {}) {
  const command = String(args?.command ?? '').trim();
  if (platform !== 'win32') return { ok: false, text: 'Run as administrator works on Windows only.' };
  if (!command) return { ok: false, text: 'Give the PowerShell command to run.' };
  const dir = mkdtempSync(join(tmpdir(), 'strata-admin-'));
  const inner = join(dir, 'command.ps1'), launcher = join(dir, 'ask.ps1'), out = join(dir, 'output.txt'), statusFile = join(dir, 'status.json');
  // a byte-order mark, so Windows PowerShell 5.1 reads the command's own non-ASCII text as UTF-8
  writeFileSync(inner, '﻿' + innerScript(command, out, statusFile), 'utf8');
  writeFileSync(launcher, '﻿' + launcherScript(inner), 'utf8');
  const ended = await run(launcher, timeoutMs);
  const read = f => { try { return readFileSync(f, 'utf8').replace(/^﻿/, ''); } catch { return null; } };
  let status = null;
  try { status = JSON.parse(read(statusFile) ?? 'null'); } catch { status = null; }
  const result = resultText({ ...ended, output: read(out) ?? '', status });
  if (!ended.timedOut) { try { rmSync(dir, { recursive: true, force: true }); } catch { /* the elevated side may still hold a file */ } }
  return result;
}

/** Start the launcher (not elevated itself) and wait for it, at most `timeoutMs`. */
export function runLauncher(launcher, timeoutMs) {
  return new Promise(resolve => {
    const child = spawn('powershell.exe', ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', launcher], { windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '', done = false;
    child.stderr.on('data', d => { stderr += d; });
    const timer = setTimeout(() => { if (done) return; done = true; child.kill(); resolve({ code: null, timedOut: true, stderr }); }, timeoutMs);
    child.on('error', e => { if (done) return; done = true; clearTimeout(timer); resolve({ code: NOT_STARTED, stderr: String(e.message) }); });
    child.on('close', code => { if (done) return; done = true; clearTimeout(timer); resolve({ code, stderr }); });
  });
}

const send = message => process.stdout.write(JSON.stringify({ jsonrpc: '2.0', ...message }) + '\n');

async function handle(line) {
  let msg;
  try { msg = JSON.parse(line); } catch { return; }
  const { id, method, params } = msg;
  if (id === undefined) return;
  try {
    if (method === 'initialize') {
      return send({ id, result: { protocolVersion: params?.protocolVersion ?? '2024-11-05', capabilities: { tools: { listChanged: false } }, serverInfo: { name: 'strata-admin', version: '1.0.0' } } });
    }
    if (method === 'ping') return send({ id, result: {} });
    if (method === 'tools/list') return send({ id, result: { tools: [{ name: TOOL, description: DESCRIPTION, inputSchema: SCHEMA, annotations: { destructiveHint: true } }] } });
    if (method === 'tools/call') {
      if (params?.name !== TOOL) return send({ id, result: { isError: true, content: [{ type: 'text', text: 'That tool is not known.' }] } });
      const r = await runAsAdministrator(params?.arguments ?? {});
      return send({ id, result: { isError: !r.ok, content: [{ type: 'text', text: r.text }] } });
    }
    return send({ id, error: { code: -32601, message: `Method not found: ${method}` } });
  } catch (e) {
    return send({ id, error: { code: -32603, message: String(e?.message ?? e) } });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  createInterface({ input: process.stdin }).on('line', line => { void handle(line); }).on('close', () => process.exit(0));
}
