/**
 * The agent preset the app's chats run under: the Harness's `standard` preset (pinned 57190202,
 * packages/bundle/web-app/presets/standard.patch.yml) with the differences the app needs, and nothing else.
 *
 * A patch replaces the whole preset row, so the list is restated here. `tests/agent-tools.test.mjs` compares this list
 * with the pinned Harness's own file (when STRATA_HARNESS_DIR is set), so a Harness update that adds or renames a
 * plugin fails a test instead of silently dropping a tool.
 *
 * Differences from `standard`, each with its reason:
 *   - `tool-schedule` removed: the app has one scheduler (server/tasks.mjs, the Tasks screen); a second set of
 *     reminder tools would run behind its back.
 *   - `web_search` removed (tool-web with `search:false`): every search backend the Harness has needs a service key
 *     the app does not have, so the tool could only fail. `web_fetch` stays. (`searchProvider` can switch it back on.)
 *   - `tool-ask-user` stays (it is the group "Questions for you" in Settings -> Tools): the app shows the agent's own questions in a card above
 *     the message box and sends the answer back (#4244). Switched off, the tool is not offered and the agent asks in plain text.
 *   - the `planning` group (Harness plan mode and `exit_plan_mode`) removed: the app's Plan mode is the permission
 *     preset (nothing can change); the Harness's plan mode needs an interactive review the app does not have.
 *   - `tool-bash` / `tool-pwsh`: exactly one, by operating system (the Harness chooses the same at load).
 *   - `tool-plugin-manager` stays disabled, as in `standard`.
 *   - The goal, skill, workflow and fork tools are kept (owner decision): the first message is slower, and the short plain system prompt below keeps it as small as it can be.
 */

import { withoutDisabled } from './tools-catalog.mjs';

const COMPACTION_PLUGIN = new URL('./plugins/strata-compaction.mjs', import.meta.url).href;

/** Plugin ids of the pinned Harness `standard` preset, in its order (the drift check). */
export const STANDARD_PLUGIN_IDS = [
  'persona', 'agent-instructions', 'time-context', 'tool-bash', 'tool-pwsh', 'tool-fs', 'tool-fs-search', 'tool-jobs',
  'tool-schedule', 'skill-filesystem', 'tool-skill', 'command-goal', 'tool-goal', 'planning', 'compaction', 'delegation',
  'tool-ask-user', 'tool-todo', 'tool-web', 'present', 'tool-plugin-manager',
];
/** What we leave out of it. */
export const REMOVED_PLUGIN_IDS = ['tool-schedule', 'planning'];
/** Plugins removed from inside the `delegation` group (the group stays: helper agents are a feature). */
export const REMOVED_DELEGATION_IDS = [];

/**
 * The whole system prompt of an app chat. The Harness's own text is about 4.8 KB (about 1,200 tokens) of guidance for working on
 * the Harness itself (its checkout, its web page, its plugins): a local model pays for it on every first message and the person
 * can see its words in the chat ("DSH", "sandbox"). This one says what a chat needs and nothing else. The tools state their own rules.
 */
export const AGENT_PROMPT = [
  'You are a helpful assistant that can work on this computer through your tools. You are running as {{model}}.',
  'Answer directly and briefly; use a tool only when needed. Use plain words, not tool names or jargon.',
  'Read a file before editing it and prefer small edits to rewrites. Use glob, grep and read for files; use the shell to run programs.',
  'Starting a program, folder or page only confirms it started. Never say a window opened, a page rendered or that you checked how it looks unless a tool confirmed it. On failure, state what went wrong in one sentence and what you will try next.',
  'To show an image, SVG, HTML or PDF from local folders, use the present tool (browsers cannot open file://). Start programs detached without waiting: Start-Process on Windows, open on macOS, xdg-open on Linux.',
  'To continue delegated work (including after a pause or update), call list_agents to find your helper and send_message with its agent_id; inactive helpers resume. Use subagent for a new assignment ("ask a subagent", "ask model X") and set its model field when asked. Never look for a model with shell commands or wait with sleep: end your turn so its answer arrives.',
  'Tokens starting with @ are files or folders the person named. A failed or refused action changed nothing; say so and never claim undone work.',
].join('\n\n');

/**
 * The one sentence that says which computer this is. The Harness's own runtime context is off (see `includeRuntimeContext` below), so without it
 * a model on Windows spent its first steps on `ls`, `uname` and `wc` before trying PowerShell commands (and some ran out of steps first).
 */
export const OS_LINES = {
  win32: 'This computer runs Windows and the shell tool is PowerShell: uname, wc and grep do not exist here; use Get-ChildItem, Get-Content, Select-String and Measure-Object. There is no need to check which system this is.',
  darwin: 'This computer runs macOS and the shell tool is a Unix shell. There is no need to check which system this is.',
  linux: 'This computer runs Linux and the shell tool is a Unix shell. There is no need to check which system this is.',
};

/** AGENT_PROMPT with the sentence for `platform` after its first line. */
export function agentPromptFor(platform = process.platform, { modelNames = false } = {}) {
  // #5971: {{model_name}} is registered by plugins/strata-models.mjs (the model's name as the person knows it); without that plugin in the
  // profile (no model routes) the Harness's own {{model}} stays, since an unregistered variable fails the prompt.
  const prompt = modelNames ? AGENT_PROMPT.replace('You are running as {{model}}.', 'You are running as {{model_name}}.') : AGENT_PROMPT;
  const line = OS_LINES[platform];
  if (!line) return prompt;
  const [first, ...rest] = prompt.split('\n\n');
  return [first, line, ...rest].join('\n\n');
}

/**
 * @param {{platform?: string, search?: boolean}} [options]
 * @returns {object} the `config` of the `preset-standard` row
 */
export function standardPreset({ platform = process.platform, search = false, off = new Set(), skillsDir = null, compaction = [], modelNames = false } = {}) {
  const windows = platform === 'win32';
  const plugins = [
    { id: 'persona', name: '@deepseek-ai/dsh-persona', config: { prefix: agentPromptFor(platform, { modelNames }), complete: true, includeRuntimeContext: false } },
    { id: 'agent-instructions', name: '@deepseek-ai/dsh-agent-instructions', config: { maxBytes: 65536 } },
    { id: 'time-context', name: '@deepseek-ai/dsh-time-context' },
    windows
      ? { id: 'tool-pwsh', name: '@deepseek-ai/dsh-tool-pwsh' }
      : { id: 'tool-bash', name: '@deepseek-ai/dsh-tool-bash' },
    { id: 'tool-fs', name: '@deepseek-ai/dsh-tool-fs' },
    { id: 'tool-fs-search', name: '@deepseek-ai/dsh-tool-fs-search', config: { sampleOverCapGlobResults: false } },
    { id: 'tool-jobs', name: '@deepseek-ai/dsh-tool-jobs' },
    // The skills of the Skills page, written out as folders the Harness scans (and watches: a skill saved in the app is there at once).
    { id: 'skill-filesystem', name: '@deepseek-ai/dsh-skill-filesystem', ...(skillsDir ? { config: { customSkillDirs: [skillsDir] } } : {}) },
    { id: 'tool-skill', name: '@deepseek-ai/dsh-tool-skill' },
    { id: 'command-goal', name: '@deepseek-ai/dsh-command-goal' },
    { id: 'tool-goal', name: '@deepseek-ai/dsh-tool-goal' },
    {
      id: 'compaction', name: 'cordis:group', group: true, isolate: { compaction: true, toolResultPruner: true },
      config: [
        { id: 'compaction-basic', name: '@deepseek-ai/dsh-compaction-basic', ...(compaction.length ? { config: { modelPolicies: compaction } } : {}) },
        { id: 'strata-compaction', name: COMPACTION_PLUGIN, config: {} },
        { id: 'command-compact', name: '@deepseek-ai/dsh-command-compact' },
        { id: 'tool-result-pruner', name: '@deepseek-ai/dsh-compaction-tool-result-pruner', config: { thresholdChars: 8192, headChars: 4096, tailChars: 1024 } },
      ],
    },
    {
      id: 'delegation', name: 'cordis:group', group: true, isolate: { workflowEngine: true },
      config: [
        { id: 'tool-subagent-control', name: '@deepseek-ai/dsh-tool-subagent-control' },
        { id: 'tool-subagent-list-agents', name: '@deepseek-ai/dsh-tool-subagent-control/list-agents' },
        { id: 'tool-subagent', name: '@deepseek-ai/dsh-tool-subagent', config: { provider: 'spawn', toolName: 'subagent', modelSelectionSettings: true } },
        { id: 'tool-subagent-fork', name: '@deepseek-ai/dsh-tool-subagent', config: { provider: 'fork', toolName: 'subagent_fork' } },
        { id: 'workflow-ptc', name: '@deepseek-ai/dsh-workflow-ptc', config: { provider: 'spawn' } },
        { id: 'tool-workflow', name: '@deepseek-ai/dsh-tool-workflow' },
      ],
    },
    { id: 'tool-ask-user', name: '@deepseek-ai/dsh-tool-ask-user' },
    { id: 'tool-todo', name: '@deepseek-ai/dsh-tool-todo', config: { allowParallelInProgress: true } },
    { id: 'tool-web', name: '@deepseek-ai/dsh-tool-web', config: { fetch: !off.has('fetch'), search: search && !off.has('search'), searchTimeoutMs: 60000 } },
    { id: 'present', name: '@deepseek-ai/dsh-tool-present' },
    { id: 'tool-plugin-manager', name: '@deepseek-ai/dsh-plugin-manager/tools', disabled: true },
  ];
  // The groups the person switched off in Settings -> Tools; a web plugin with both halves off is not loaded at all.
  const kept = withoutDisabled(plugins, off).filter(p => p.id !== 'tool-web' || p.config.fetch || p.config.search);
  return { id: 'standard', order: 1, plugins: kept };
}
