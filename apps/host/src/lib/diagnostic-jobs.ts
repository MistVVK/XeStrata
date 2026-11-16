/**
 * #5489: the Diagnostics card listed the last three failed jobs, and since the job history is kept across restarts (#5802) a failure stayed
 * there for good: "UD-IQ4_XS-v31 needs engine 0.1.57 or newer; this one is 0.1.56" was still shown with engine 0.1.58 in use. A failure is
 * shown only while nothing later has settled it: a later job for the same model that succeeded (a load, an install), a later job of the same
 * kind that succeeded for jobs without a model, or a later successful change of the engine (an update, a rollback, an engine file, a Strata
 * update), after which the engine the failure ran on is gone. The full history stays in the engine logs.
 */
import type { NativeJob } from './native';

const ENGINE_CHANGES = new Set(['update-engine', 'rollback', 'install-engine-file', 'update-strata']);

export function openFailures(jobs: readonly NativeJob[] | undefined, max = 3): NativeJob[] {
  const ordered = [...(jobs ?? [])].sort((a, b) => String(a.startedAt).localeCompare(String(b.startedAt)));
  let open: NativeJob[] = [];
  for (const job of ordered) {
    if (job.state === 'error') { open.push(job); continue; }
    if (job.state !== 'done') continue;
    if (ENGINE_CHANGES.has(job.kind)) open = [];
    else if (job.model) open = open.filter(f => f.model !== job.model);
    else open = open.filter(f => f.model || f.kind !== job.kind);
  }
  return open.slice(-max);
}
