/**
 * Strata's correction for the staged Harness MCP client (#6014): an MCP server at an address that was restarted (the UE5 editor) no
 * longer knows the client's session, and every later tool call failed ("-32600 Unknown session id ...; client should reinitialize")
 * until the model was unloaded and loaded again.
 *
 * @deepseek-ai/dsh-mcp-client 0.2.1-alpha.3 reconnects (a new transport, initialize, a fresh tool list) only when the transport closes. A
 * Streamable HTTP server that forgot the session answers the request instead, so nothing closed. After this patch a tool call that fails
 * with an expired-session answer (HTTP 404, or an error naming an unknown, invalid, expired or missing session) closes that connection,
 * and the client's own reconnect starts a new session (its usual policy: first try after 500 ms). The failed call is not sent again:
 * it fails with the server's answer, and the next call runs on the new session.
 */
import { createHash } from 'node:crypto';
import { lstatSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

export const HARNESS_MCP_SESSION_PATCH = Object.freeze({
  id: 'strata-mcp-session-reconnect-v1',
  package: '@deepseek-ai/dsh-mcp-client',
  version: '0.2.1-alpha.3',
  file: 'node_modules/@deepseek-ai/dsh-mcp-client/lib/index.js',
  // the npm tarball's lib/index.js and the staged copy in the 0.2.15 app are byte-identical (checked 11-16)
  beforeSha256: '758d8fc473b700b5d351158d06ce9f9e92c16e8b25497b08bb7bd2122a87f5dd',
});

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

const HELPERS = `/**
* Strata patch ${HARNESS_MCP_SESSION_PATCH.id} (#6014): a server that was restarted no longer knows the session and answers every call
* with an error instead of closing the transport. Such a call closes its connection, so the reconnect below starts a new session.
*/
const strataSessionGone = new WeakMap();
function strataIsSessionGone(error) {
	if (error?.code === 404) return true;
	const text = String(error?.message ?? error ?? "");
	return /\\bsession\\b/i.test(text) && /unknown|invalid|expired|not found|no valid|missing|terminated|reinitiali[sz]e/i.test(text);
}
function strataCallTool(client, request, options) {
	return client.callTool(request, options).catch((error) => {
		if (strataIsSessionGone(error)) strataSessionGone.get(client)?.();
		throw error;
	});
}
`;

const EDITS = [
  ['async function syncTools(client, ctx, opts, previous) {\n', `${HELPERS}async function syncTools(client, ctx, opts, previous) {\n`],
  ['\t\t\tcall: (args, execution) => client.callTool({\n', '\t\t\tcall: (args, execution) => strataCallTool(client, {\n'],
  ['\t\tclient = generation;\n\t\tcloseClient = closeGeneration;\n',
   '\t\tclient = generation;\n\t\tcloseClient = closeGeneration;\n\t\tstrataSessionGone.set(generation, () => {\n\t\t\tif (!isCurrent(generation) || connectedAt === void 0) return;\n\t\t\tctx.logger.warn(`${label}: the server no longer knows this session (was it restarted?); reconnecting`);\n\t\t\tvoid closeGeneration();\n\t\t});\n'],
];

/** The patched text of the pinned source (exported for the tests). Throws when an edit does not find its text exactly once. */
export function patchedMcpSource(source) {
  let out = source;
  for (const [before, after] of EDITS) {
    if (out.split(before).length !== 2) throw new Error(`Harness MCP session patch: expected exactly one ${JSON.stringify(before.trim().slice(0, 60))}`);
    out = out.replace(before, () => after);
  }
  return out;
}

/** Patch only a flattened, disposable staged Harness; returns the BUNDLE.json provenance entry. Refuses before writing on any mismatch. */
export function patchStagedHarnessMcpSession(stagedRoot) {
  const fail = message => { throw new Error(`Harness MCP session patch: ${message}`); };
  const root = realpathSync(resolve(stagedRoot));
  const packageDir = join(root, 'node_modules', '@deepseek-ai', 'dsh-mcp-client');
  const target = join(root, HARNESS_MCP_SESSION_PATCH.file);
  let manifest, before;
  try {
    manifest = JSON.parse(readFileSync(join(packageDir, 'package.json'), 'utf8'));
    if (realpathSync(target) !== target || !lstatSync(target).isFile() || lstatSync(target).nlink !== 1) fail('expected an unlinked staged module');
    before = readFileSync(target);
  } catch (error) {
    if (String(error?.message).startsWith('Harness MCP session patch:')) throw error;
    fail(`cannot read a standalone staged dsh-mcp-client (${error.message})`);
  }
  if (manifest?.name !== HARNESS_MCP_SESSION_PATCH.package || manifest?.version !== HARNESS_MCP_SESSION_PATCH.version)
    fail(`expected ${HARNESS_MCP_SESSION_PATCH.package} ${HARNESS_MCP_SESSION_PATCH.version}; review the patch before staging a different version`);
  if (sha256(before) !== HARNESS_MCP_SESSION_PATCH.beforeSha256) fail('unexpected lib/index.js SHA-256; review the dependency source before staging');
  const after = Buffer.from(patchedMcpSource(before.toString('utf8')));
  writeFileSync(target, after);
  return { ...HARNESS_MCP_SESSION_PATCH, afterSha256: sha256(after) };
}
