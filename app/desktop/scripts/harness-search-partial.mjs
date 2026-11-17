/**
 * Strata's correction for the staged Harness search tools (#5964): one unreadable folder no longer blanks a whole glob / grep.
 *
 * @deepseek-ai/dsh-tool-fs-search 0.2.1-alpha.3 treats every ripgrep exit code other than 0 and 1 as a failure. ripgrep exits 2 when it
 * walked past a folder it may not read (on Windows an ordinary case: "rg: C:\...\locked: IO error for operation on ...: Access is denied.
 * (os error 5)") and still prints everything it found, so the tool threw the found paths and matches away. After this patch an exit 2
 * whose stderr lines are ALL such access errors returns what was found, and the result says that it is partial and which folders were
 * skipped. A pattern ripgrep rejects, any other stderr line, a stderr tail that was cut short, a signal and an abort fail as before.
 *
 * The note travels from the tool body to its result text by the call's argument object: the Harness runs execute(exec.arguments, exec)
 * and later render(exec.arguments, value) with that same object (dsh-tools createSuccessResult), while the value itself is snapshotted.
 * The output schema is unchanged.
 */
import { createHash } from 'node:crypto';
import { lstatSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

export const HARNESS_SEARCH_PATCH = Object.freeze({
  id: 'strata-fs-search-partial-v1',
  package: '@deepseek-ai/dsh-tool-fs-search',
  version: '0.2.1-alpha.3',
  file: 'node_modules/@deepseek-ai/dsh-tool-fs-search/lib/index.js',
  // the npm tarball's lib/index.js and the staged copy in the 0.2.15 app are byte-identical (checked 11-16)
  beforeSha256: '8d64d2e8a5310b442ae66bc0daf7d6957ec095dbe43cabd9eb208f7370e381ed',
});

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

const HELPERS = `/**
* Strata patch ${HARNESS_SEARCH_PATCH.id} (#5964): ripgrep exits 2 when it walked past a folder it may not read and still prints what it
* found. When every stderr line is such an access error, the found results are returned and marked partial; anything else fails as before.
*/
const strataPartialSearches = new WeakMap();
const STRATA_UNREADABLE_LINE = /^rg: (.+?): (?:IO error for operation on .+|.*\\(os error (?:5|13)\\)\\s*|.*(?:Permission denied|Access is denied).*)$/;
function strataUnreadablePaths(text, lossy) {
	if (lossy || /regex parse error|error parsing glob/i.test(text)) return null;
	const lines = String(text).split(/\\r?\\n/).filter((line) => line.trim().length > 0);
	if (lines.length === 0) return null;
	const paths = [];
	for (const line of lines) {
		const match = STRATA_UNREADABLE_LINE.exec(line.trimEnd());
		if (match === null) return null;
		if (!paths.includes(match[1])) paths.push(match[1]);
	}
	return paths;
}
function strataPartialText(args) {
	const paths = args !== null && typeof args === "object" ? strataPartialSearches.get(args) : void 0;
	if (paths === void 0 || paths.length === 0) return "";
	const one = paths.length === 1, shown = paths.slice(0, 5).join(", ") + (paths.length > 5 ? ", ..." : "");
	return one ? \`\\n\\n(Partial result: 1 folder could not be read and was skipped: \${shown}. Matches inside it are not listed.)\`
		: \`\\n\\n(Partial result: \${paths.length} folders could not be read and were skipped: \${shown}. Matches inside them are not listed.)\`;
}
`;

/** [old, new] pairs, each `old` present exactly once in the pinned source. */
const EDITS = [
  ['let rgPathPromise;\n', `${HELPERS}let rgPathPromise;\n`],
  ['\tif (outcome.exitCode !== 0 && outcome.exitCode !== 1) throw classifyRunFailure(toolName, outcome.exitCode, stderr.text, stderr.lossy);\n\treturn {\n\t\tstdout: completeStdout(toolName, stdout, rawOutputMaxBytes),\n\t\tnoMatches: outcome.exitCode === 1,\n\t\tworkdir\n\t};',
   '\tconst unreadable = outcome.exitCode === 2 ? strataUnreadablePaths(stderr.text, stderr.lossy) : null;\n\tif (outcome.exitCode !== 0 && outcome.exitCode !== 1 && unreadable === null) throw classifyRunFailure(toolName, outcome.exitCode, stderr.text, stderr.lossy);\n\tconst complete = completeStdout(toolName, stdout, rawOutputMaxBytes);\n\treturn {\n\t\tstdout: complete,\n\t\tnoMatches: outcome.exitCode === 1 || unreadable !== null && complete.trim().length === 0,\n\t\tworkdir,\n\t\t...unreadable !== null ? { unreadable } : {}\n\t};'],
  ['\t\t\tconst run = await runRipgrep(ctx, exec, "glob", buildGlobCommand(input), caps.rawOutputMaxBytes, caps.graceMs, caps.stderrMaxBytes);\n',
   '\t\t\tconst run = await runRipgrep(ctx, exec, "glob", buildGlobCommand(input), caps.rawOutputMaxBytes, caps.graceMs, caps.stderrMaxBytes);\n\t\t\tif (run.unreadable !== void 0 && args !== null && typeof args === "object") strataPartialSearches.set(args, run.unreadable);\n'],
  ['\t\t\tconst run = await runRipgrep(ctx, exec, "grep", buildGrepCommand(parseGrepArgs(args)), caps.rawOutputMaxBytes, caps.graceMs, caps.stderrMaxBytes);\n',
   '\t\t\tconst run = await runRipgrep(ctx, exec, "grep", buildGrepCommand(parseGrepArgs(args)), caps.rawOutputMaxBytes, caps.graceMs, caps.stderrMaxBytes);\n\t\t\tif (run.unreadable !== void 0 && args !== null && typeof args === "object") strataPartialSearches.set(args, run.unreadable);\n'],
  ['\t\t\t\ttext: renderGlobPaths(value.paths, caps, value.root)\n', '\t\t\t\ttext: renderGlobPaths(value.paths, caps, value.root) + strataPartialText(_args)\n'],
  ['\t\t\t\ttext: renderGlobPaths(paths, caps, value.root, spillRef)\n', '\t\t\t\ttext: renderGlobPaths(paths, caps, value.root, spillRef) + strataPartialText(exec.arguments)\n'],
  ['\t\t\t\ttext: formatRetainedGrep(retainGrepMatches(value.matches, caps.maxMatches, caps.maxLineBytes))\n', '\t\t\t\ttext: formatRetainedGrep(retainGrepMatches(value.matches, caps.maxMatches, caps.maxLineBytes)) + strataPartialText(_args)\n'],
  ['\t\t\t\ttext: formatRetainedGrep(retainGrepMatches(matches, caps.maxMatches, caps.maxLineBytes), spillRef)\n', '\t\t\t\ttext: formatRetainedGrep(retainGrepMatches(matches, caps.maxMatches, caps.maxLineBytes), spillRef) + strataPartialText(exec.arguments)\n'],
];

/** The patched text of the pinned source (exported for the tests). Throws when an edit does not find its text exactly once. */
export function patchedSearchSource(source) {
  let out = source;
  for (const [before, after] of EDITS) {
    if (out.split(before).length !== 2) throw new Error(`Harness search patch: expected exactly one ${JSON.stringify(before.trim().slice(0, 60))}`);
    out = out.replace(before, () => after);
  }
  return out;
}

/**
 * Patch only a flattened, disposable staged Harness. Returns the BUNDLE.json provenance entry, as harness-budget.mjs does.
 * An unknown package, version or source, a linked file or a repeat application fails before anything is written.
 */
export function patchStagedHarnessSearch(stagedRoot) {
  const fail = message => { throw new Error(`Harness search patch: ${message}`); };
  const root = realpathSync(resolve(stagedRoot));
  const packageDir = join(root, 'node_modules', '@deepseek-ai', 'dsh-tool-fs-search');
  const target = join(root, HARNESS_SEARCH_PATCH.file);
  let manifest, before;
  try {
    manifest = JSON.parse(readFileSync(join(packageDir, 'package.json'), 'utf8'));
    if (realpathSync(target) !== target || !lstatSync(target).isFile() || lstatSync(target).nlink !== 1) fail('expected an unlinked staged module');
    before = readFileSync(target);
  } catch (error) {
    if (String(error?.message).startsWith('Harness search patch:')) throw error;
    fail(`cannot read a standalone staged dsh-tool-fs-search (${error.message})`);
  }
  if (manifest?.name !== HARNESS_SEARCH_PATCH.package || manifest?.version !== HARNESS_SEARCH_PATCH.version)
    fail(`expected ${HARNESS_SEARCH_PATCH.package} ${HARNESS_SEARCH_PATCH.version}; review the patch before staging a different version`);
  if (sha256(before) !== HARNESS_SEARCH_PATCH.beforeSha256) fail('unexpected lib/index.js SHA-256; review the dependency source before staging');
  const after = Buffer.from(patchedSearchSource(before.toString('utf8')));
  writeFileSync(target, after);
  return { ...HARNESS_SEARCH_PATCH, afterSha256: sha256(after) };
}
