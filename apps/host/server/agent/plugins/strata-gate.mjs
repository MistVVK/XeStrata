/**
 * The Strata approval gate: a small Harness plugin (plain ESM, no imports, loaded by absolute path from the profile).
 *
 * Why it exists. The Harness only asks a person when a tool escalates its sandbox. With the sandbox read-only that
 * produced a blocked attempt first (a card full of sandbox and PowerShell errors) and the real Approve/Deny card
 * only on the model's second try. In Manual mode this plugin asks BEFORE a change runs, through the Harness's own
 * approval service, so the person sees exactly one card per change and nothing has run when it appears.
 *
 * How it works. It listens to `tools/pre-execute` (the same seam the Harness's own Auto-review layer uses). For a
 * session whose effective preset is Manual it returns an `ask` decision for every call that is not known to be
 * read-only. The Harness then asks, runs the tool only after an approval, and records the answer. Manual runs with
 * the sandbox open, so an approved change can really happen anywhere the person agreed to; the approval is the
 * control, which is why anything this file cannot prove is read-only asks (fail closed).
 *
 * Children. A subagent's approval policy is pinned to `never` by the Harness (its requests are rejected on the
 * spot). For a child of a Manual chat that would hide the card and the child could not do anything. The gate
 * writes `ask` into that child's log, so the child's request reaches the same single Approve/Deny card.
 * Plan and Bypass children are untouched.
 *
 * The status command `/strata-gate` lets the host confirm the gate is really loaded before it lets a Manual turn
 * start (a Manual chat with no gate would be a chat with the sandbox open and nobody asking).
 */

export const name = 'strata-gate';
export const inject = ['approval', 'permissionPresets', 'sessions', 'tools'];

/** Preset key the host writes for Manual (apps/host/server/agent/profile.mjs MODE_PRESETS). */
export const MANUAL_PRESET = 'strata-manual';
/** Preset key the host writes for Plan (read-only, nothing may change). */
export const PLAN_PRESET = 'strata-plan';
/** The app's own look-only tool server (the person's instruction tools): it only hands back text. */
const LOOK_ONLY_SERVER = 'mcp__strata-instructions__';
/** The tools a person wrote as commands (Settings -> Tools): the card says which one runs and with what text. */
const COMMANDS_SERVER = 'mcp__strata-commands__';
/** Run as administrator (#5968, admin-tool-mcp.mjs): Manual asks first, and Windows asks again for every call; Plan refuses it. */
const ADMIN_TOOL = 'mcp__strata-admin__run_as_administrator';
/** The browser tool server's name in the agent (Settings -> Tools). Looking at the page does not ask; acting on it does. */
const BROWSER_SERVER = 'mcp__browser__';
const BROWSER_LOOKING = new Set(['browser_snapshot', 'browser_take_screenshot', 'browser_console_messages', 'browser_network_requests', 'browser_wait_for']);

/** Tools that cannot change anything on this computer. Everything else asks. */
export const READ_ONLY_TOOLS = new Set([
  'read', 'glob', 'grep', 'read_image', 'working_directory', 'todo_write', 'skill', 'present', 'lsp',
  'get_goal', 'create_goal', 'update_goal', 'list_agents', 'list_subagent_models', 'job_list', 'job_output',
  'ask_user_question', 'exit_plan_mode',
  // (web_search and web_fetch are not here: the first use in a chat asks, see webDecision.)
  // Delegation: the child it starts is governed by this same gate.
  'subagent', 'subagent_fork', 'send_message', 'interrupt_agent', 'workflow',
]);

const SHELL_TOOLS = new Set(['pwsh', 'bash']);

/**
 * Which shell commands run without a card in Manual mode, and with which options. Safe by construction (#3975): a command is on the list WITH the
 * options that are known to only look. Every other option, long or short, abbreviated or attached, makes the command ask. So a write mode
 * (`sort -o`, `tree -o`, `uniq in out`), a program-execution mode (`sort --compress-program`, `rg --pre`, `git grep -O`), a setting mode
 * (`date -s`) or one that never ends (`tail -f`) cannot slip in, and a word added to the list later starts with no options at all.
 *
 * A spec says what one command accepts:
 *   f   short options without a value (letters, so `-la` is `-l -a`)       v   short options that need a value (`-k2`, `-k 2`)
 *   vn  the same, and the value must be a number                          o   short options with an optional value that must be attached (`-Iseconds`)
 *   n   a bare `-5` is allowed                                            l   long options (`--name`, `--name=value`), spelled in full
 *   lv  long options that need a value (`--name=value`, `--name value`)   ps  PowerShell parameter names, lower case, spelled in full
 *   max how many operands (files, patterns) are allowed                   op  a check for each operand
 *   post a last check from the options seen and the operand count
 */
const words = (text) => (text ? text.trim().split(/\s+/) : []);
const spec = ({ f = '', v = '', vn = '', o = '', n = false, l = '', lv = '', ps = '', max = Infinity, op = null, post = null }) =>
  ({ f, v, vn, o, n, l: new Set(words(l)), lv: new Set(words(lv)), ps: ps ? new Set([...words(ps), ...PS_COMMON]) : null, max, op, post });
/** Parameters every cmdlet has that only change how errors and warnings show. (-Debug asks a question and -OutVariable sets one: not listed.) */
const PS_COMMON = ['erroraction', 'ea', 'warningaction', 'wa', 'informationaction', 'ia', 'verbose', 'vb'];
const NUMBER = /^[+-]?\d+[A-Za-z]{0,3}$/;

/** What `ls`, `dir`, `gci` and `Get-ChildItem` take. */
const LS = { f: 'aAbBcdDFgGhHiklLmnNopqQrRsStuUvxX1CZ', v: 'ITw', ps: 'path literalpath filter include exclude recurse depth name force file directory hidden readonly system attributes',
  l: 'all almost-all human-readable recursive directory classify reverse size inode dereference color group-directories-first sort time time-style block-size indicator-style si numeric-uid-gid no-group literal hide ignore quote-name escape author file-type full-time width tabsize' };
/** What `cat`, `type`, `gc` and `Get-Content` take. (-Wait never ends, so it is not here.) */
const CAT = { f: 'AbeEnstTuv', ps: 'path literalpath totalcount tail head raw encoding readcount force filter include exclude delimiter stream asbytestream',
  l: 'show-all number-nonblank show-ends number squeeze-blank show-tabs show-nonprinting' };
const MEASURE = { ps: 'property sum average maximum minimum line word character ignorewhitespace inputobject standarddeviation allstats' };
const SELECT = { ps: 'first last skip skiplast property expandproperty unique index excludeproperty inputobject' };
const WHERE = { ps: 'property value inputobject eq ne gt lt ge le like notlike match notmatch contains notcontains in notin is isnot ceq cne cgt clt cge cle clike cnotlike cmatch cnotmatch ccontains cnotcontains cin cnotin' };
const GROUP = { ps: 'property noelement ashashtable asstring casesensitive inputobject culture' };
const FT = { ps: 'property autosize wrap hidetableheaders groupby view force expand repeatheader displayerror showerror inputobject' };
const FL = { ps: 'property view groupby expand force displayerror showerror inputobject' };
const SLS = { ps: 'pattern path literalpath simplematch casesensitive context list notmatch quiet allmatches include exclude encoding raw inputobject culture noemphasis' };
const GETITEM = { ps: 'path literalpath name filter include exclude force stream' };
const GETLOC = { ps: 'psprovider psdrive stack stackname' };
const ECHO = { f: 'neE', ps: 'inputobject noenumerate' };
/** The GNU/BSD grep family. */
const GREP = { f: 'aiIlLcnvwHhEFGPqrRsTUyzZbxo', v: 'efdD', vn: 'ABCm', n: true,
  l: 'extended-regexp fixed-strings basic-regexp perl-regexp ignore-case no-ignore-case word-regexp line-regexp invert-match count color colour files-with-matches files-without-match max-count only-matching quiet silent no-messages byte-offset with-filename no-filename label line-buffered initial-tab null after-context before-context context text recursive dereference-recursive include exclude exclude-dir binary-files line-number null-data no-group-separator group-separator regexp file directories devices binary' };

/** Commands that only look, each with the options that are known to be harmless. Matched against the first word of every pipeline segment. */
const SPECS = new Map();
const add = (names, s) => { for (const name of words(names)) SPECS.set(name, spec(s)); };
// PowerShell
add('get-childitem gci', { ps: LS.ps });
add('ls dir', LS);
add('get-content gc', { ps: CAT.ps });
add('cat type', CAT);
add('get-item gi get-itemproperty', GETITEM);
add('get-location gl', GETLOC);
add('pwd', { ...GETLOC, f: 'LP' });
add('select-string sls', SLS);
add('measure-object measure', MEASURE);
add('select-object select', SELECT);
add('where-object where', WHERE);
add('sort-object', { ps: 'property descending unique top bottom casesensitive culture stable inputobject' });
add('format-table ft', FT);
add('format-list fl', FL);
add('out-string', { ps: 'stream width inputobject' });
add('test-path', { ps: 'path literalpath pathtype isvalid filter include exclude olderthan newerthan' });
add('resolve-path', { ps: 'path literalpath relative relativebasepath force' });
add('get-filehash', { ps: 'path literalpath algorithm' });
add('get-date', { ps: 'format uformat date asutc displayhint year month day hour minute second millisecond' });
add('get-command', { ps: 'name commandtype module all syntax listimported totalcount noun verb fullyqualifiedmodule parametername parametertype' });
add('get-process', { ps: 'name id includeusername module fileversioninfo inputobject' });
add('write-output', { ps: ECHO.ps });
add('echo', ECHO);
add('get-host', { ps: '' });
add('get-psdrive', { ps: 'name psprovider scope literalname' });
add('get-help', { ps: 'name full detailed examples parameter category component functionality role' });
add('group-object group', GROUP);
add('convertto-json', { ps: 'depth compress enumsasstrings inputobject asarray escapehandling' });
add('convertfrom-json', { ps: 'ashashtable depth inputobject noenumerate' });
add('select-xml', { ps: 'xml path literalpath xpath content namespace' });
add('compare-object', { ps: 'referenceobject differenceobject property includeequal excludedifferent passthru syncwindow casesensitive culture' });
add('get-unique', { ps: 'inputobject asstring caseinsensitive onetype' });
// POSIX. `sort` and `date` take no PowerShell names here: `sort -o FILE` writes, `date -s` sets the clock, and a PowerShell-looking `-Second` or `-Top`
// must not be read by a real sort or date as one of those. (Sort-Object and Get-Date above are the PowerShell forms.)
add('head', { f: 'qvz', vn: 'nc', n: true, l: 'quiet silent verbose zero-terminated lines bytes' });
add('tail', { f: 'qvz', vn: 'nc', n: true, l: 'quiet silent verbose zero-terminated lines bytes' }); // no -f, -F, --follow, --retry, --pid, -s: those never end
add('wc', { f: 'lwcmL', l: 'lines words bytes chars max-line-length' });
add('grep egrep fgrep', GREP);
add('rg', { // no --pre, --pre-glob, --hostname-bin (they run a program) and no -z/--search-zip (it runs decompressors)
  f: 'iIsSnNHlcwFxvUoLaquhVpbP', v: 'efgtTr', vn: 'ABCjdMm',
  l: 'ignore-case no-ignore-case smart-case case-sensitive fixed-strings word-regexp line-regexp invert-match count count-matches files-with-matches files-without-match files type-list hidden no-ignore no-ignore-vcs no-ignore-parent no-ignore-dot no-ignore-global no-ignore-messages follow line-number no-line-number with-filename no-filename column heading no-heading null only-matching quiet text multiline multiline-dotall pcre2 no-messages json stats trim unrestricted binary no-config vimgrep pretty byte-offset passthru crlf one-file-system no-unicode regexp file glob iglob type type-not max-count max-depth maxdepth context after-context before-context color replace threads sort sortr max-filesize max-columns path-separator encoding engine ignore-file glob-case-insensitive no-glob-case-insensitive' });
add('stat', { f: 'LtxlnqrsFf', v: 'c', l: 'dereference file-system terse format printf' });
add('file', { f: 'bhikLnNrvI', l: 'brief dereference no-dereference mime mime-type mime-encoding keep-going no-pad no-buffer raw' }); // no -C, -m, -f, -P (they write or read other files), no -z/-s/-p (they run decompressors, read devices, reset times)
add('du', { f: 'abchHklmsxLPSD0', v: 'tB', vn: 'd', l: 'all apparent-size bytes total dereference-args human-readable si summarize one-file-system dereference no-dereference separate-dirs count-links inodes null max-depth threshold block-size time time-style exclude' });
add('df', { f: 'ahHiklmPTv', v: 'txB', l: 'all human-readable si inodes local portability print-type total sync no-sync output block-size type exclude-type' });
add('uname', { f: 'asnrvmpio', max: 0, l: 'all kernel-name nodename kernel-release kernel-version machine processor hardware-platform operating-system' });
add('which', { f: 'as', l: 'all' });
add('basename', { f: 'az', v: 's', l: 'multiple zero suffix' });
add('dirname', { f: 'z', l: 'zero' });
add('realpath', { f: 'emsqzLP', l: 'canonicalize-existing canonicalize-missing no-symlinks quiet zero strip logical physical relative-to relative-base' });
add('readlink', { f: 'femnqsvz', l: 'canonicalize canonicalize-existing canonicalize-missing no-newline quiet silent verbose zero' });
add('sort', { f: 'bdfghiMnRrsuVzmcC', v: 'kt', l: 'ignore-leading-blanks dictionary-order ignore-case general-numeric-sort human-numeric-sort ignore-nonprinting month-sort numeric-sort random-sort reverse stable unique version-sort zero-terminated check merge sort field-separator key' });
// uniq's second operand is its OUTPUT file, so it takes at most one.
add('uniq', { f: 'cdDuiz', vn: 'fsw', max: 1, l: 'count repeated all-repeated unique ignore-case skip-fields skip-chars check-chars zero-terminated group' });
add('cut', { f: 'snz', v: 'bcfd', l: 'complement only-delimited zero-terminated bytes characters fields delimiter output-delimiter' });
add('tr', { f: 'cCdst', max: 2 });
add('date', { f: 'uR', o: 'I', v: 'dr', lv: 'date reference', l: 'utc universal rfc-email rfc-2822 iso-8601 rfc-3339',
  op: (word) => word.startsWith('+') }); // a bare operand like 1026120026 SETS the clock; only +FORMAT is a format
add('id', { f: 'ugGnrazZ', l: 'user group groups name real context zero' });
add('hostname', { f: 'sfiIdaAyv', max: 0 }); // `hostname NAME` renames the machine
add('whoami', { l: 'help version', op: (word) => /^\/(all|user|groups|priv|upn|fqdn|logonid|nh)$/i.test(word) });
add('tree', { f: 'adfilpsugDFqNQrtvxCnASUchJX', v: 'LPIHT', vn: '', l: 'help version filelimit charset sort timefmt noreport dirsfirst filesfirst prune du si gitignore matchdirs ignore-case inodes device nolinks' }); // no -o, no -R
// `less` and `more` are not here: they wait for keys, and `less` can run commands.

/** The names that run without a card (for the tests). */
export const readOnlyCommandNames = () => [...SPECS.keys()];

/** git subcommands that only read, with the options they may take. */
const GIT_VIEW = 'oneline graph decorate no-decorate all branches tags remotes stat shortstat numstat dirstat name-only name-status patch no-patch raw summary compact-summary abbrev-commit no-abbrev-commit abbrev reverse follow no-merges merges first-parent full-history simplify-by-decoration simplify-merges ancestry-path left-right left-only right-only cherry-pick cherry-mark cherry topo-order date-order author-date-order relative-date all-match invert-grep regexp-ignore-case extended-regexp fixed-strings perl-regexp no-color color color-words word-diff word-diff-regex minimal patience histogram diff-algorithm check exit-code quiet cached staged merge-base no-index no-ext-diff no-textconv find-renames find-copies find-copies-harder no-renames ignore-space-change ignore-all-space ignore-space-at-eol ignore-blank-lines ignore-cr-at-eol function-context show-function unified inter-hunk-context src-prefix dst-prefix no-prefix full-index binary text source use-mailmap mailmap no-mailmap log-size notes no-notes show-signature diff-merges boundary children parents no-walk sparse date format pretty author committer grep since until after before max-count skip min-parents max-parents diff-filter encoding stat-width stat-name-width stat-graph-width stat-count count';
const LISTING = ['l', 'list', 'contains', 'no-contains', 'merged', 'no-merged', 'points-at'];
const GIT = new Map(Object.entries({
  status: spec({ f: 'sbvz', o: 'u', l: 'short branch porcelain long ignored untracked-files ignore-submodules renames no-renames find-renames show-stash ahead-behind no-ahead-behind column no-column verbose null' }),
  log: spec({ f: 'pswbMCizuacmrtqEFPR', vn: 'nU', v: 'SGL', n: true, l: GIT_VIEW }),
  diff: spec({ f: 'pswbMCizuacmrtqEFPR', vn: 'nU', v: 'SGL', n: true, l: GIT_VIEW }),
  show: spec({ f: 'pswbMCizuacmrtqEFPR', vn: 'nU', v: 'SGL', n: true, l: GIT_VIEW }),
  'rev-list': spec({ f: 'pswbMCizuacmrtqEFPR', vn: 'nU', v: 'SGL', n: true, l: GIT_VIEW }),
  shortlog: spec({ f: 'sneciw', n: false, l: `${GIT_VIEW} summary numbered email committer group` }),
  // Listing forms only. A name with no list option would CREATE a branch or a tag, and the options that change things are not here.
  branch: spec({ f: 'avri', l: 'list all remotes verbose show-current ignore-case contains no-contains merged no-merged points-at sort format abbrev no-abbrev color no-color column no-column omit-empty',
    post: (seen, count) => count === 0 || LISTING.some(name => seen.has(name)) }),
  tag: spec({ f: 'li', o: 'n', l: 'list sort contains no-contains merged no-merged points-at format column no-column ignore-case omit-empty color no-color',
    post: (seen, count) => count === 0 || LISTING.some(name => seen.has(name)) }),
  remote: spec({ f: 'v', l: 'verbose', max: 0 }),
  // `git config KEY` reads and `git config KEY VALUE` writes, so a key is only taken with --get; no --edit, --add, --unset, --file.
  config: spec({ f: 'lz', l: 'list get get-all get-regexp show-origin show-scope local global system worktree bool int bool-or-int path expiry-date type null name-only includes no-includes', lv: 'default',
    post: (seen, count) => (count > 0 ? ['get', 'get-all', 'get-regexp'].some(name => seen.has(name)) : seen.has('list') || seen.has('l')) }),
  'rev-parse': spec({ f: 'q', l: 'show-toplevel git-dir git-common-dir absolute-git-dir abbrev-ref symbolic symbolic-full-name short verify is-inside-work-tree is-inside-git-dir is-bare-repository is-shallow-repository show-prefix show-cdup show-superproject-working-tree all branches tags remotes quiet revs-only no-revs flags no-flags local-env-vars show-object-format path-format', lv: 'default git-path' }),
  'ls-files': spec({ f: 'cdmoiskuzvtfD', l: 'cached deleted modified others ignored stage directory unmerged killed no-empty-directory error-unmatch exclude-standard exclude full-name eol deduplicate recurse-submodules sparse abbrev resolve-undo format' }),
  'ls-tree': spec({ f: 'rtdlz', l: 'name-only name-status long full-name full-tree object-only abbrev format' }),
  blame: spec({ f: 'wMCespltbfnc', v: 'L', l: 'porcelain line-porcelain incremental show-email show-name show-number date root abbrev' }),
  describe: spec({ l: 'all tags always long dirty broken contains abbrev match exclude first-parent exact-match candidates' }),
  'cat-file': spec({ f: 'tsep', l: 'batch batch-check batch-all-objects allow-unknown-type buffer unordered follow-symlinks mailmap use-mailmap' }), // no --textconv, --filters
  // No -O / --open-files-in-pager: they run a program on the matches.
  grep: spec({ f: 'aiIlLcnvwHhEFGPqpWrzUo', v: 'e', vn: 'ABCm', n: true, l: 'no-index untracked cached no-exclude-standard exclude-standard recurse-submodules text ignore-case word-regexp invert-match full-name extended-regexp basic-regexp perl-regexp fixed-strings line-number column files-with-matches name-only files-without-match count null only-matching quiet no-color color break heading show-function function-context all-match and or not max-depth max-count threads after-context before-context context recursive no-recursive' }),
}));

/**
 * One command's words against its spec: every option has to be one the spec knows (anything else is not understood, so it asks).
 * @param {ReturnType<typeof spec>} sp
 * @param {{t:string,q:boolean}[]} args the words after the command; q marks a quoted text, which is data and never an option
 */
function argumentsAreSafe(sp, args) {
  const seen = new Set();
  let operands = 0, ended = false;
  const operand = (arg) => { operands++; return operands <= sp.max && (!sp.op || sp.op(arg.t, arg.q)); };
  const valueOf = (i, numeric) => { // the next word as an option's value; it can be a quoted text, a number, or a word that is not an option
    const arg = args[i + 1];
    if (!arg) return false;
    return numeric ? NUMBER.test(arg.t) : arg.q || !arg.t.startsWith('-') || NUMBER.test(arg.t);
  };
  for (let i = 0; i < args.length; i++) {
    const { t, q } = args[i];
    if (q || ended || t === '-' || t[0] !== '-') { if (!operand(args[i])) return false; continue; }
    if (t === '--') { ended = true; continue; }
    if (t.startsWith('--')) {
      const eq = t.indexOf('=');
      const name = eq < 0 ? t.slice(2) : t.slice(2, eq);
      if (sp.l.has(name)) { seen.add(name); continue; }
      if (sp.lv.has(name)) { seen.add(name); if (eq < 0) { if (!valueOf(i, false)) return false; i++; } continue; }
      return false;
    }
    if (sp.ps) {
      const named = /^-([A-Za-z][A-Za-z0-9]*)(?::|$)/.exec(t);
      if (named && sp.ps.has(named[1].toLowerCase())) { seen.add(named[1].toLowerCase()); continue; }
    }
    if (sp.n && /^-\d+$/.test(t)) continue;
    for (let j = 1; j < t.length; j++) {
      const c = t[j];
      if (sp.f.includes(c)) { seen.add(c); continue; }
      if (sp.o.includes(c)) { seen.add(c); break; }
      const numeric = sp.vn.includes(c);
      if (!numeric && !sp.v.includes(c)) return false;
      seen.add(c);
      const rest = t.slice(j + 1);
      if (rest) { if (numeric && !NUMBER.test(rest)) return false; } else { if (!valueOf(i, numeric)) return false; i++; }
      break;
    }
  }
  return sp.post ? sp.post(seen, operands) : true;
}

const SAFE_VERSION = /^(node|python|python3|py|npm|pnpm|git|dotnet|cargo|rustc|go|java|pip|pip3)\s+(--version|-v|-version|version)$/i;

/** Anything that can chain, redirect, expand or start something else is not read-only. (`sh` as a word, not inside an option like `du -sh`.) */
const UNSAFE_SYNTAX = /[;&<>`${}()]|\r|\n|--%|\bout-file\b|\bset-content\b|\badd-content\b|\binvoke-|\biex\b|\bstart-|\bnew-|\bremove-|\bcopy-|\bmove-|\brename-|\bclear-|\bset-|\bpush-|\bexpand-|\bsave-|\binstall-|\bcurl\b|\bwget\b|\bsudo\b|(?<![-\w])sh(?![-\w])|\bbash\b|\bpwsh\b|\bpowershell\b|\bcmd\b|-exec\b|-delete\b|-execdir\b/i;

/**
 * The command with every quoted text replaced by a plain stand-in, or null when it cannot be read safely. What is inside quotes is data
 * (a path like "C:\Users\x\Q&A", a pattern like 'a|b'), so an & ; | ( ) < > or { in it must not decide anything. Two things still can:
 * inside double quotes a $ or a backtick expands (a variable, a sub-command), and an unterminated quote means the text was not understood.
 * Either one gives null, and the command asks as before.
 * @param {string} text
 * @param {string[]} [quoted] when given, receives the text inside each quote, in order (the checks on operands read it)
 * @returns {string|null}
 */
export function maskQuotes(text, quoted) {
  let out = '';
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c !== '"' && c !== "'") { out += c; continue; }
    let end = text.indexOf(c, i + 1);
    // PowerShell writes a quote inside a quoted text by doubling it ('it''s'): the text goes on past the pair.
    while (end >= 0 && text[end + 1] === c) end = text.indexOf(c, end + 2);
    if (end < 0) return null;
    if (c === '"' && /[$`]/.test(text.slice(i + 1, end))) return null;
    // A quoted text is a whole word. One glued to other characters (--out"put"=f, "a"b) or one that starts with a dash could hide an option from the
    // word checks below (git diff "--output=f" writes a file), so those are not understood either.
    const before = text[i - 1], after = text[end + 1];
    if ((before !== undefined && !/[\s|]/.test(before)) || (after !== undefined && !/[\s|]/.test(after)) || text[i + 1] === '-') return null;
    out += '"_"';
    quoted?.push(text.slice(i + 1, end));
    i = end;
  }
  return out;
}

/**
 * Whether a shell command only reads. Conservative by design: one unrecognised word or option and the answer is no.
 * Quoted text is data and is looked at after it has been set aside (maskQuotes): Get-ChildItem "C:\Users\x\Q&A" only lists.
 * @param {string} command
 * @returns {boolean}
 */
export function isReadOnlyShell(command) {
  const text = String(command ?? '').trim();
  if (!text || text.length > 600) return false;
  if (SAFE_VERSION.test(text)) return true;
  const quoted = [];
  const masked = maskQuotes(text, quoted);
  if (masked === null || UNSAFE_SYNTAX.test(masked)) return false;
  let next = 0;
  const tokens = masked.split('|').map(segment => segment.trim()).map(segment => words(segment).map(w => (w === '"_"' ? { t: quoted[next++] ?? '', q: true } : { t: w, q: false })));
  return tokens.every((segment) => {
    if (!segment.length || segment[0].q) return false;
    const first = segment[0].t.toLowerCase().replace(/^\.\\/, '');
    if (first === 'git') {
      const sub = segment[1];
      const sp = sub && !sub.q ? GIT.get(sub.t) : null;
      return !!sp && argumentsAreSafe(sp, segment.slice(2));
    }
    const sp = SPECS.get(first);
    return !!sp && argumentsAreSafe(sp, segment.slice(1));
  });
}

const clip = (value, max) => {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
};

/** The tools that reach the internet. In Manual the first one in a chat asks once; the answer holds for the rest of that chat. */
export const WEB_TOOLS = new Set(['web_search', 'web_fetch']);
/** What the model reads when the person says no. Plain words, and it tells the model what to do next. */
export const WEB_REFUSAL = 'The person did not allow web access, so nothing was searched or opened. Go on without it and say so.';

/**
 * The question for a web call, in plain words: what is about to leave this computer and where it goes.
 * @param {{name:string, arguments:any}} exec
 * @param {{provider?: string}} [options] provider is the search service's name as the person knows it
 */
export function webDecision(exec, options = {}) {
  const tool = String(exec.name ?? '');
  const args = exec.arguments && typeof exec.arguments === 'object' ? exec.arguments : {};
  if (tool === 'web_search') {
    const queries = Array.isArray(args.queries) ? args.queries : [args.query ?? args.q];
    const words = clip(queries.filter(q => typeof q === 'string' && q.trim()).join('; '), 160);
    return { kind: 'ask', web: true, reason: `The assistant wants to search the web for: ${words}. Search words go to ${options.provider || 'the search service'}.` };
  }
  return { kind: 'ask', web: true, reason: `The assistant wants to open the web page ${clip(args.url, 200)}. The page's site sees the request.` };
}

/**
 * The decision for one call in Manual mode, or null when it needs no question.
 * @param {{name:string, arguments:any}} exec
 * @param {{provider?: string}} [options]
 * @returns {{kind:'ask', reason:string, web?:true}|null}
 */
export function manualDecision(exec, options = {}) {
  const tool = String(exec.name ?? '');
  const args = exec.arguments && typeof exec.arguments === 'object' ? exec.arguments : {};
  if (WEB_TOOLS.has(tool)) return webDecision(exec, options);
  if (READ_ONLY_TOOLS.has(tool)) return null;
  if (SHELL_TOOLS.has(tool)) {
    if (isReadOnlyShell(args.command)) return null;
    const what = args.description ? clip(args.description, 160) : clip(args.command, 160);
    return { kind: 'ask', reason: `Run a command that may change things: ${what}` };
  }
  if (tool === 'write') return { kind: 'ask', reason: `Create or replace the file ${clip(args.file_path ?? args.path, 200)}` };
  if (tool === 'edit') return { kind: 'ask', reason: `Change the file ${clip(args.file_path ?? args.path, 200)}` };
  if (tool === 'job_kill') return { kind: 'ask', reason: 'Stop a background task' };
  if (tool.startsWith(LOOK_ONLY_SERVER)) return null;
  if (tool.startsWith(BROWSER_SERVER)) return browserDecision(tool.slice(BROWSER_SERVER.length), args);
  if (tool === ADMIN_TOOL) return { kind: 'ask', reason: `Run as administrator (Windows will ask too): ${clip(args.command, 160)}${args.reason ? ` (${clip(args.reason, 120)})` : ''}` };
  if (tool.startsWith(COMMANDS_SERVER)) return { kind: 'ask', reason: `Run your tool “${tool.slice(COMMANDS_SERVER.length).replace(/_/g, ' ')}”${args.input ? ` with: ${clip(args.input, 120)}` : ''}` };
  if (tool.startsWith('mcp__')) return { kind: 'ask', reason: `Use the connected tool ${tool.split('__').slice(1).join(' / ')}` };
  return { kind: 'ask', reason: `Use the tool ${tool}` };
}

/** The browser in plain words: opening a page, clicking or typing asks; looking at what is already open does not. */
function browserDecision(name, args) {
  if (BROWSER_LOOKING.has(name) || (name === 'browser_tabs' && args.action === 'list')) return null;
  const url = clip(args.url, 200);
  if (name === 'browser_navigate') return { kind: 'ask', reason: `Open the web page ${url}` };
  if (name === 'browser_click' || name === 'browser_hover' || name === 'browser_drag') return { kind: 'ask', reason: `Click on the page${args.element ? `: ${clip(args.element, 120)}` : ''}` };
  if (name === 'browser_type' || name === 'browser_fill_form' || name === 'browser_press_key' || name === 'browser_select_option') return { kind: 'ask', reason: `Type into the page${args.element ? `: ${clip(args.element, 120)}` : ''}` };
  if (name === 'browser_file_upload') return { kind: 'ask', reason: 'Send a file from this computer to the page' };
  if (name === 'browser_evaluate' || name === 'browser_run_code') return { kind: 'ask', reason: 'Run a small program inside the web page' };
  return { kind: 'ask', reason: `Use the browser (${name.replace(/^browser_/, '').replace(/_/g, ' ')})` };
}

/**
 * Plan mode changes nothing. A connected tool (a person's tool server, a command they wrote, the browser) is not under the sandbox, so
 * Plan refuses it instead of letting it run. The look-only instruction tools are fine.
 * @returns {{kind:'deny', reason:string}|null}
 */
export function planDecision(exec) {
  const tool = String(exec.name ?? '');
  if (!tool.startsWith('mcp__') || tool.startsWith(LOOK_ONLY_SERVER)) return null;
  return { kind: 'deny', reason: 'Plan mode only looks and changes nothing, so it cannot use this tool. Switch to Manual or Bypass to use it.' };
}

/**
 * Blocked folders (#5737, Settings -> Tools): every tool call that names a file or folder inside one is refused, in every mode and for
 * helpers too, before anything runs. `blocked` comes from the host (profile.mjs blockedForms): per folder the name the person sees and
 * the spellings that reach the same place (lower case, forward slashes: the real path, the Windows short name, ~ / %USERPROFILE% /
 * $HOME forms, /mnt/d and /d forms). Every string argument is searched, so a command that names the folder is refused too. A program the
 * assistant starts could still reach a folder without naming it; Strata does not change Windows permissions (Settings says so).
 */
export const BLOCKED_REASON = 'This folder is blocked in Settings → Tools. Do not try to reach it another way; tell the person it is blocked.';
/** A string as the matcher reads it: lower case, forward slashes, no doubled slashes, no "/./", "dir/../" folded once per level. */
export function pathForm(text) {
  let s = String(text).toLowerCase().replace(/\\/g, '/').replace(/\/{2,}/g, '/').replace(/\/\.(?=\/|$)/g, '');
  for (let i = 0; i < 20 && /\/[^/]+\/\.\.(?=\/|$)/.test(s); i++) s = s.replace(/\/(?!\.\.\/)[^/]+\/\.\.(?=\/|$)/, '');
  return s;
}
const strings = (value, out = [], depth = 0) => {
  if (typeof value === 'string') out.push(value);
  else if (value && typeof value === 'object' && depth < 6) for (const v of Object.values(value)) strings(v, out, depth + 1);
  return out;
};
/** The blocked folder a call names, or null. A spelling matches at a word start and ends at the end, a slash, a quote, a space or a `;`. */
export function blockedFolderIn(exec, blocked = []) {
  if (!Array.isArray(blocked) || !blocked.length) return null;
  const texts = strings(exec?.arguments).map(pathForm);
  for (const folder of blocked) {
    for (const form of folder?.forms ?? []) {
      if (!form) continue;
      for (const t of texts) {
        let at = t.indexOf(form);
        while (at >= 0) {
          const before = at === 0 ? '' : t[at - 1], after = t[at + form.length] ?? '';
          if (!/[a-z0-9_]/.test(before) && (after === '' || /[/\s"'`;,|)>]/.test(after))) return folder.shown ?? form;
          at = t.indexOf(form, at + 1);
        }
      }
    }
  }
  return null;
}
export function blockedDecision(exec, blocked) {
  const shown = blockedFolderIn(exec, blocked);
  return shown ? { kind: 'deny', reason: `${BLOCKED_REASON} (${shown})` } : null;
}

/**
 * @param {object} ctx
 * @param {{web?: {provider?: string, skipAsk?: boolean}, blocked?: {shown: string, forms: string[]}[]}} [config] `web.provider`: the search service's name for the question;
 *        `web.skipAsk`: the person's switch in Settings -> Tools, "search the web without asking" (off unless they turned it on)
 */
export function apply(ctx, config = {}) {
  const web = config.web ?? {};
  /** Chats (their first session) where the person said yes to the web. One yes holds for the rest of the chat. */
  const webAllowed = new Set();
  /** A question already on the screen for a chat, so two searches started together share one card. */
  const webAsking = new Map();
  const chatOf = (session, depth = 0) => {
    const parentId = session.header?.parentSession;
    if (session.header?.origin === 'subagent' && parentId && depth < 6) {
      const parent = ctx.sessions.get(parentId);
      if (parent) return chatOf(parent, depth + 1);
    }
    return session.id;
  };
  async function askWeb(exec, gate, agent) {
    const chat = chatOf(agent.session);
    let question = webAsking.get(chat);
    if (!question) {
      question = ctx.approval.request({ agent, toolName: exec.name, callId: exec.callId, reason: gate.reason, displayReason: { en: gate.reason }, signal: exec.signal });
      webAsking.set(chat, question);
      const forget = () => { if (webAsking.get(chat) === question) webAsking.delete(chat); };
      question.then(forget, forget);
    }
    const outcome = await question;
    if (outcome === 'allowed-once') { webAllowed.add(chat); return null; }
    if (outcome === 'cancelled') return { kind: 'cancel' };
    return { kind: 'deny', reason: WEB_REFUSAL };
  }

  /** The preset a session is really working under: a child follows the chat that started it. */
  const effectivePreset = (session, depth = 0) => {
    const parentId = session.header?.parentSession;
    if (session.header?.origin === 'subagent' && parentId && depth < 6) {
      const parent = ctx.sessions.get(parentId);
      if (parent) return effectivePreset(parent, depth + 1);
    }
    return ctx.permissionPresets.current(session);
  };

  ctx.on('tools/pre-execute', async (exec, next) => {
    const agent = exec.agent;
    // Blocked folders hold in every mode, for every tool (run_code included) and for helpers, before anything else decides.
    const blocked = blockedDecision(exec, config.blocked);
    if (blocked) return blocked;
    if (!agent || exec.name === 'run_code') return next();
    // Fail closed (T4): a session whose preset cannot be read (a reset in progress, a helper whose chat is gone) is treated as Manual, so
    // anything this file cannot prove is read-only still asks. It used to run unasked.
    let preset;
    try { preset = effectivePreset(agent.session); } catch (e) {
      preset = MANUAL_PRESET;
      try { console.warn(`[strata-gate] could not read the permission mode for this call (${e?.message ?? e}); treating it as Manual`); } catch { /* logging must never decide a permission */ }
    }
    // A helper that was set to ask while the chat was in Manual must stop asking the moment the chat leaves Manual,
    // so a mode change applies to the very next tool call of a running goal and its helpers.
    if (preset !== MANUAL_PRESET && agent.session.header?.origin === 'subagent' && ctx.approval.overrideOf(agent.session) === 'ask') {
      agent.session.append('approval/policy', { policy: 'never' });
    }
    if (preset === PLAN_PRESET) return planDecision(exec) ?? next();
    if (preset !== MANUAL_PRESET) return next();
    const gate = manualDecision(exec, { provider: web.provider });
    if (!gate) return next();
    if (gate.web) {
      // The person's switch, or a yes already given in this chat: no question.
      if (web.skipAsk === true || webAllowed.has(chatOf(agent.session))) return next();
      // A helper's own approval policy is `never`; it must be `ask` for the question to reach the person.
      if (agent.session.header?.origin === 'subagent' && ctx.approval.overrideOf(agent.session) !== 'ask') {
        agent.session.append('approval/policy', { policy: 'ask' });
      }
      return (await askWeb(exec, gate, agent)) ?? next();
    }
    // A child's approval policy is pinned to `never` by the Harness; asking is how its request reaches the person.
    if (agent.session.header?.origin === 'subagent' && ctx.approval.overrideOf(agent.session) !== 'ask') {
      agent.session.append('approval/policy', { policy: 'ask' });
    }
    const downstream = await next();
    if (downstream.kind !== 'allow') return downstream;
    return { kind: 'ask', reason: gate.reason, displayReason: { en: gate.reason } };
  }, { prepend: true });

  // Lets the host prove the gate is live before it starts a Manual turn.
  ctx.inject(['commands'], (commandCtx) => {
    commandCtx.commands.register({
      definitionId: 'strata-gate',
      name: 'strata-gate',
      description: 'Reports that the Strata approval gate is active',
      handler: () => ({ kind: 'success', text: 'strata-gate active' }),
    });
  });
}
