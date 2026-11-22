import { createHash } from 'node:crypto';

const RECENT_RESULTS = 64;
const REPEATED_RESULT_LIMIT = 4;
const STEPS_WITHOUT_PROGRESS = 12;
const WAIT_TOOLS = new Set(['job_wait', 'wait_for_job', 'wait_for_subagent']);
const THINKING_WINDOW = 4096;
const THINKING_CHECK_EVERY = 768;
const TEST_REPORTERS = [
  {
    command: /^node(?:\.exe)?[ \t]+--test(?:[ \t]|$)/i,
    timing: [
      /^([ \t]*[✔✖﹣] .*) \(\d+(?:\.\d+)?ms\)(?=[ \t]*$)/gmu,
      /^([ \t]*(?:(?:#|ℹ) duration_ms|duration_ms:)[ \t]+)\d+(?:\.\d+)?(?=[ \t]*$)/gmu,
    ],
  },
  {
    command: /^(?:pytest(?:\.exe)?|python(?:\.exe)?[ \t]+-m[ \t]+pytest)(?:[ \t]|$)/i,
    timing: [
      /^([ \t]*(?:=+[ \t]+)?\d+ (?:passed|failed|skipped|xfailed|xpassed|errors?|warnings?|deselected)(?:, \d+ (?:passed|failed|skipped|xfailed|xpassed|errors?|warnings?|deselected))* in )\d+(?:\.\d+)?s(?=[ \t=]*$)/gmu,
    ],
  },
  {
    command: /^ctest(?:\.exe)?(?:[ \t]|$)/i,
    timing: [
      /^([ \t]*\d+\/\d+[ \t]+Test[ \t]+#\d+:.*(?:Passed|Failed|Timeout|Not Run)[ \t]+)\d+(?:\.\d+)? sec(?=[ \t]*$)/gmu,
      /^([ \t]*Total Test time \(real\)[ \t]*=[ \t]*)\d+(?:\.\d+)? sec(?=[ \t]*$)/gmu,
    ],
  },
];

/** Catch a clearly periodic suffix in the model's reasoning, without reading tool drafts or answer text. */
function repeatsPassage(text) {
  const words = text.toLowerCase().replace(/\s+/g, ' ').trim();
  if (words.length < 1500) return false;
  for (let length = 80; length <= 768 && length * 3 <= words.length; length++) {
    const passage = words.slice(-length);
    if (words.endsWith(passage.repeat(3)) && /[\p{L}\p{N}]/u.test(passage)) return true;
  }
  return false;
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  }
  return value;
}

function argumentsOf(args) {
  if (typeof args !== 'string') return args;
  try { return JSON.parse(args); } catch { return args; }
}

/** Ignore the reporter's timing fields for a standalone test command. Keep
 * output from scripts, monitors and compound shell commands verbatim. */
function testResult(call, content, fileChanges) {
  if (!['bash', 'pwsh'].includes(call.name) || !Array.isArray(content)) return null;
  const command = call.args?.command;
  if (typeof command !== 'string' || /[&;|<>\r\n]/u.test(command)) return null;
  const reporter = TEST_REPORTERS.find(r => r.command.test(command.trim()));
  if (!reporter) return null;
  let changed = false;
  const result = content.map(block => {
    if (block?.type !== 'text' || typeof block.text !== 'string') return block;
    const text = reporter.timing.reduce((value, pattern) => value.replace(pattern, '$1<time>'), block.text);
    if (text === block.text) return block;
    changed = true;
    return { ...block, text };
  });
  if (!changed) return null;
  const { description, ...args } = call.args;
  return [{ ...call, args }, result, fileChanges];
}

function fingerprint(call, message, fileChanges) {
  return createHash('sha256')
    .update(JSON.stringify(canonical([message.isError === true, testResult(call, message.content, fileChanges) ?? [call, message.content]])))
    .digest('hex');
}

/** One per agent, observing durable events. Model prose and tool traffic alone do not reset the budget.
 *  New successful results do reset it so a long sequence of distinct reads, edits or tests can finish. */
export class ProgressGuard {
  calls = new Map();
  recent = new Map();
  withoutProgress = 0;
  progressed = false;
  waited = false;
  repeated = false;
  attempt = null;
  thinking = '';
  sinceThinkingCheck = 0;
  fileChanges = 0;

  newInstruction() {
    this.recent.clear();
    this.progress();
  }

  observeStream(frame) {
    if (frame.type !== 'chunk' || frame.chunk?.type !== 'reasoning-delta' || !frame.chunk.text) return null;
    if (frame.attemptId !== this.attempt) {
      this.attempt = frame.attemptId;
      this.thinking = '';
      this.sinceThinkingCheck = 0;
    }
    this.thinking = (this.thinking + frame.chunk.text).slice(-THINKING_WINDOW);
    this.sinceThinkingCheck += frame.chunk.text.length;
    if (this.sinceThinkingCheck < THINKING_CHECK_EVERY) return null;
    this.sinceThinkingCheck = 0;
    return repeatsPassage(this.thinking) ? 'repeated-thinking' : null;
  }

  progress() {
    this.progressed = true;
    this.repeated = false;
    this.withoutProgress = 0;
  }

  observe(event) {
    const data = event.data ?? {};
    switch (event.type) {
      case 'tool/call':
        if (data.callId) this.calls.set(data.callId, { name: data.name, args: argumentsOf(data.arguments) });
        break;
      case 'tool/result': {
        const message = data.message;
        const call = this.calls.get(message?.toolCallId);
        if (!call) break;
        this.calls.delete(message.toolCallId);
        if (WAIT_TOOLS.has(call.name)) { this.waited = true; break; }
        const key = fingerprint(call, message, this.fileChanges);
        const prior = this.recent.get(key) ?? 0;
        this.recent.delete(key);
        this.recent.set(key, prior + 1);
        if (this.recent.size > RECENT_RESULTS) this.recent.delete(this.recent.keys().next().value);
        if (prior + 1 >= REPEATED_RESULT_LIMIT) this.repeated = true;
        else if (prior === 0 && !message.isError) {
          if (call.name === 'edit' || call.name === 'write') this.fileChanges++;
          this.progress();
        }
        break;
      }
      case 'step/end': {
        if (this.repeated && !this.progressed) return 'repeated-work';
        if (!this.progressed && !this.waited) this.withoutProgress += 1;
        this.progressed = false;
        this.waited = false;
        this.repeated = false;
        if (this.withoutProgress >= STEPS_WITHOUT_PROGRESS) return 'no-progress';
        break;
      }
      case 'turn/end':
        this.calls.clear();
        break;
    }
    return null;
  }
}
