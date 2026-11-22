import { resolve } from 'node:path';
import { parseArguments } from './events.mjs';

const marker = /\[verify-read\b/i;
const suffix = /\s*\[verify-read ([^|\]\r\n]+) \| ([^\]\r\n]+)\]\s*$/iu;
const clip = (text, size) => String(text ?? '').slice(0, size);
const fileKey = (cwd, path) => {
  const full = resolve(cwd, path);
  return process.platform === 'win32' ? full.toLowerCase() : full;
};
const relativePath = value => {
  const path = value.trim().replaceAll('\\', '/');
  return path.length <= 120 && path.split('/').every(part => /^[\p{L}\p{N}_. -]+$/u.test(part) && part !== '.' && part !== '..') ? path : null;
};

// Only an explicit file switch changes the check. Status requests and extra
// reading instructions can arrive during the same step without changing it.
const fileReference = String.raw`(?:\x60([^\x60\r\n]+)\x60|"([^"\r\n]+)"|([^\s\x60",;!?]+))`;
const directionStart = String.raw`(?:^|[.!?]\s+|\n)\s*(?:please\s+|bitte\s+)?`;
const replacement = new RegExp(String.raw`${directionStart}(?:use|read|take|nimm|nutze|verwende|lies)\s+${fileReference}\s+(?:instead of|statt|anstelle von)\s+${fileReference}`, 'iu');
const switchTo = new RegExp(String.raw`${directionStart}(?:switch to|wechsle zu)\s+${fileReference}`, 'iu');
const reference = (match, index) => relativePath(match[index] ?? match[index + 1] ?? match[index + 2]?.replace(/\.$/u, '') ?? '');

function changedTarget(text, cwd, current) {
  if (!current?.path || !cwd) return null;
  const pair = replacement.exec(text);
  const from = pair && reference(pair, 4);
  const next = pair && from && fileKey(cwd, from) === fileKey(cwd, current.path) ? pair : switchTo.exec(text);
  if (!next) return null;
  const path = reference(next, 1);
  if (!path || !/\.[\p{L}\p{N}]+$/u.test(path) || fileKey(cwd, path) === fileKey(cwd, current.path)) return null;
  return { path };
}

/** Opt-in steps may declare a literal line to verify through the Harness read tool. */
export function readRequirement(text) {
  if (!marker.test(text)) return null;
  const match = suffix.exec(text);
  if (!match || marker.test(text.slice(0, match.index))) return { invalid: true };
  const path = relativePath(match[1]);
  const line = match[2];
  if (!path || line.length > 120) return { invalid: true };
  return { path, line };
}

function readInstruction(requirement) {
  const target = JSON.stringify(requirement.path);
  return typeof requirement.line === 'string'
    ? 'Use read on ' + target + ' and verify this exact line: ' + JSON.stringify(requirement.line) + '.'
    : 'Use read on ' + target + ' to verify its current contents.';
}

/** Preserve the verification suffix before the plan editor clips its prose. */
export function splitReadRule(value, enabled) {
  const raw = String(value ?? '');
  const start = enabled ? raw.search(marker) : -1;
  return start < 0 ? { body: raw, rule: '' } : { body: raw.slice(0, start).trimEnd(), rule: raw.slice(start, start + 272).trimEnd() };
}

export function requirementOf(step) {
  if (step.verification) return step.verification;
  const inTitle = readRequirement(step.text);
  const inDetail = readRequirement(step.detail);
  return inTitle && inDetail ? { invalid: true } : inTitle ?? inDetail;
}

export function stepBrief(plan, index) {
  const step = plan.steps[index];
  const previous = plan.steps.slice(0, index);
  const done = previous.filter(item => item.state === 'done');
  const latest = done.at(-1);
  const requirement = requirementOf(step);
  return [
    'Overall aim: ' + clip(plan.title, 80),
    'Current step ' + (index + 1) + ' of ' + plan.steps.length + ': ' + clip(String(step.text ?? '').replace(suffix, ''), 140),
    step.detail ? 'Step detail: ' + clip(step.detail.replace(suffix, ''), 220) : '',
    index > 0 ? done.length + ' earlier steps completed. Keep their results.' : '',
    latest?.summary ? 'Latest completed result: ' + clip(latest.summary, 96) : '',
    plan.briefing ? 'Workspace instructions: ' + clip(plan.briefing, 600) : '',
    step.verificationDirection && step.verificationDirection !== step.correction ? 'User direction that changed the check: ' + clip(step.verificationDirection, 600) : '',
    step.correction ? 'Latest user direction for this step: ' + clip(step.correction, 600) : '',
    requirement?.path ? 'Before marking this step complete: ' + readInstruction(requirement) : '',
    'Work on the current step. Apply the latest live user direction if it changes the step. Verify the result before finishing the goal.',
  ].filter(Boolean);
}

/** Proof is local to the active step and to the most recent live direction.
 * Observe the parent session's actual tool events, rather than the model's summary. */
export class StepEvidence {
  read = null;
  receipt = null;

  constructor(cwd, requirement) {
    this.cwd = cwd;
    this.requirement = requirement;
  }

  correct(text) {
    const requirement = readRequirement(text) ?? changedTarget(text, this.cwd, this.requirement);
    if (!requirement) return false;
    this.requirement = requirement;
    this.receipt = null;
    this.read = null;
    return true;
  }

  observe(event) {
    const { type, seq, time, data } = event;
    if (!this.requirement?.path || !this.cwd || !data) return;
    const target = fileKey(this.cwd, this.requirement.path);
    if (type === 'tool/call') {
      // A later shell or code action may change the file without a file argument.
      if (['pwsh', 'bash', 'run_code'].includes(data.name)) {
        this.receipt = null;
        this.read = null;
        return;
      }
      const path = parseArguments(data.arguments).file_path;
      if (typeof path !== 'string' || fileKey(this.cwd, path) !== target) return;
      if (data.name === 'read' && typeof data.callId === 'string' && Number.isSafeInteger(seq)) {
        // A newer inspection supersedes the previous observation of this file.
        this.receipt = null;
        this.read = { id: data.callId, seq };
      } else if (data.name === 'write' || data.name === 'edit') {
        this.receipt = null;
        this.read = null;
      }
      return;
    }
    if (type !== 'tool/result') return;
    const callId = data.message?.toolCallId;
    const read = this.read;
    if (!read || callId !== read.id) return;
    this.read = null;
    if (data.message.isError !== false || seq <= read.seq || typeof data.meta?.path !== 'string'
      || fileKey(this.cwd, data.meta.path) !== target || !Array.isArray(data.meta.lines)) return;
    const expected = this.requirement.line;
    const line = data.meta.lines.find(entry => typeof entry?.text === 'string' && (expected === undefined || entry.text === expected)
      && Number.isSafeInteger(entry.number) && entry.number > 0);
    if (line) this.receipt = { path: this.requirement.path, ...(expected === undefined ? {} : { line: line.number }), callId, seq, at: time };
  }
}
