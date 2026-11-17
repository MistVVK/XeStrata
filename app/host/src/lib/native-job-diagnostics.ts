import type { NativeJob } from './native';

/** What the host says about a job a restart cut short. */
export const INTERRUPTED = 'The host restarted before completion. Check diagnostics, then retry; completed files are kept.';
const updateKinds = new Set(['update-strata', 'update-engine']);

/**
 * A later finished update of the same component settles the restart interruption of an earlier one (ported from #3578 by medking82).
 * The records are never changed: a running retry, another component or an ordinary failure stays a plain failure.
 */
export function recoveredInterruptions(jobs: readonly NativeJob[]): Set<string> {
  const time = (job: NativeJob) => Date.parse(job.startedAt);
  return new Set(jobs.filter(job => job.state === 'error' && updateKinds.has(job.kind) && job.error === INTERRUPTED && Number.isFinite(time(job))
    && jobs.some(later => later.state === 'done' && later.kind === job.kind && Number.isFinite(time(later)) && time(later) > time(job))).map(job => job.id));
}

/** The text of a failed job's alert. The alert stays red; an interruption a later update has settled says so in plain words. */
export function jobErrorText(job: NativeJob, jobs: readonly NativeJob[]): string {
  return recoveredInterruptions(jobs).has(job.id)
    ? 'This update was cut short when the app restarted. A later update finished, so nothing needs doing.'
    : job.error ?? job.summary;
}

/**
 * The failed update Settings > About shows under its buttons: per component, the last failed one, and only while no later update of the same
 * component has finished. A failure a later update made right (the first-run engine card’s “A model is running … then press Continue”
 * after Continue installed the engine, QAGUI-STRIX 11-09) is history: it stays in Diagnostics, not in About. Jobs are kept across restarts
 * (native-jobs.json), so without this the line stayed there for good.
 */
export function currentUpdateFailure(jobs: readonly NativeJob[]): NativeJob | undefined {
  const time = (job: NativeJob) => Date.parse(job.startedAt);
  const failures = jobs.filter(job => job.state === 'error' && updateKinds.has(job.kind)
    && !jobs.some(later => later.state === 'done' && later.kind === job.kind && Number.isFinite(time(later)) && (!Number.isFinite(time(job)) || time(later) > time(job))));
  return failures.at(-1);
}

/**
 * A Strata or engine update that finished after the update check shown was made: that check is out of date (it still says “Update available”,
 * and its cache is keyed to the Control version that runs until a restart). Settings > About then checks again. The job’s id, or undefined.
 */
export function updateFinishedAfter(jobs: readonly NativeJob[], checkedAt: string | undefined): string | undefined {
  const checked = Date.parse(checkedAt ?? '');
  if (!Number.isFinite(checked)) return undefined;
  const ended = (job: NativeJob) => Date.parse((job as NativeJob & { finishedAt?: string }).finishedAt ?? job.startedAt);
  return jobs.filter(job => job.state === 'done' && updateKinds.has(job.kind) && ended(job) > checked).at(-1)?.id;
}
