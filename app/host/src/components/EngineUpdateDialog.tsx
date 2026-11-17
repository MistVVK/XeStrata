import { useEffect, useState } from 'react';
import { localeTag, tr } from '../lib/i18n';
import { Bar, Btn, Dialog } from './ui';
import { nativeAction, useNative, type NativeJob } from '../lib/native';

/** "A", "A and B", "A, B and C" in the language of the app. */
const listOf = (names: string[]) => {
  try { return new Intl.ListFormat(localeTag(), { style: 'long', type: 'conjunction' }).format(names); } catch { return names.join(', '); }
};

/**
 * Update engine with a model loaded (Settings > About). Strata Control does the work (`strata update --stop-model --output jsonl`, through the host); this is the
 * conversation around it, in four plain steps:
 *  1. Warn, naming the model, and ask. Nothing happens before the yes. With no model loaded the older dialog of Settings is used instead.
 *  2. Progress in words: the new engine downloads while the model keeps working; then the model stops; then the engine is updated.
 *  3. Offer to start the model again with the same settings. Never automatic.
 *  4. A failed update leaves the old engine as it was and offers to start the model again on it.
 * The dialog follows the host's state, so it also shows the offer when the update finished while Settings was closed.
 */
export function EngineUpdateDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const native = useNative();
  const [jobId, setJobId] = useState('');
  const [restartId, setRestartId] = useState('');
  const [error, setError] = useState('');
  const [hidden, setHidden] = useState(false);
  const loaded = native?.instances ?? [];
  const offer = native?.stoppedForUpdate ?? null;
  const job: NativeJob | undefined = native?.jobs.find(j => j.id === jobId);
  const restart: NativeJob | undefined = native?.jobs.find(j => j.id === restartId);
  const offerKey = offer?.at ?? '';
  useEffect(() => { setHidden(false); }, [offerKey]);

  const restarting = restart?.state === 'running';
  const updating = job?.state === 'running';
  const names = listOf(offer?.models.map(m => m.label) ?? loaded.map(i => i.label));
  const showOffer = !!offer && !hidden && !updating && !restarting;
  const failedRestart = restart?.state === 'error' && !offer ? restart : undefined;
  const ended = job?.state === 'done' || job?.state === 'error';
  const visible = open || updating || restarting || showOffer || !!failedRestart || ended;
  if (!visible) return null;

  async function start() {
    setError('');
    try {
      const started = await nativeAction<NativeJob>('operations', { kind: 'update-engine', args: { stopModel: true } });
      setJobId(started.id);
      onClose();                                  // the dialog now follows the job; the caller's "open" is spent
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }
  async function startAgain() {
    setError('');
    try {
      const started = await nativeAction<NativeJob>('operations', { kind: 'restart-models' });
      setRestartId(started.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }
  async function notNow() {
    setHidden(true);
    try { await nativeAction('engine-restart/dismiss'); } catch { /* the offer stays hidden here either way */ }
    setJobId('');
    onClose();
  }
  function close() {
    if (showOffer) { void notNow(); return; }
    if (failedRestart) setRestartId('');
    onClose();
  }

  const percent = Number(job?.result?.install?.download?.percent);
  const details = (j: NativeJob) => j.details ? <details className="native-settings-diagnostic"><summary className="small">{tr("Details")}</summary><p className="small muted">{j.details}</p></details> : null;

  // 3 and 4: the model was stopped for the update.
  if (showOffer && offer) {
    const done = offer.updated;
    return <Dialog open compact title={tr(done ? 'The engine is updated' : "The engine was not updated")} onClose={close}
      actions={<><Btn onClick={() => void notNow()}>{tr("Not now")}</Btn><Btn variant="primary" onClick={() => void startAgain()}>{tr("Start again")}</Btn></>}>
      <p>{done ? tr("The engine is updated. Start {0} again with the same settings?", [names]) : tr("The update didn't finish. Your old engine is unchanged and still works. Start {0} again on it?", [names])}</p>
      {!done && job ? details(job) : null}
      {error ? <p className="small inline-error" role="alert">{tr(error)}</p> : null}
    </Dialog>;
  }
  // The model is starting again.
  if (restarting) {
    return <Dialog open compact title={tr("Starting {0} again", [names])} onClose={onClose}>
      <p role="status">{tr("Starting {0}. Loading can take a few minutes.", [names])}</p>
      <div className="engine-update-progress"><Bar value={0} indeterminate label={tr("Starting {0} again", [names])}/></div>
    </Dialog>;
  }
  if (failedRestart) {
    return <Dialog open compact title={tr("The model did not start again")} onClose={close} actions={<Btn variant="primary" onClick={close}>{tr("Close")}</Btn>}>
      <p>{tr("You can load it again from Models.")}</p>
      {failedRestart.error ? <p className="small muted">{failedRestart.error}</p> : null}
    </Dialog>;
  }
  // 2: the update is running.
  if (updating && job) {
    const step = job.step, label = listOf(job.models ?? loaded.map(i => i.label));
    return <Dialog open compact title={tr("Updating the engine")} onClose={onClose}>
      <p role="status" aria-live="polite">{step === 'stopping' ? tr("Stopping {0}. Answers in progress finish first.", [label])
        : step === 'stopped' ? tr("{0} is stopped. Updating the engine…", [label])
        : step === 'getting-ready' ? tr("Getting the new engine ready. {0} keeps working until it is ready.", [label])
        : tr(job.summary || 'Starting…')}</p>
      <div className="engine-update-progress">{step === 'getting-ready' && Number.isFinite(percent) ? <Bar value={percent} label={tr("Downloading…")}/> : <Bar value={0} indeterminate label={tr("Updating the engine")}/>}</div>
    </Dialog>;
  }
  // The update ended without stopping anything (declined, refused, or failed before the model was touched).
  if (job?.state === 'error') {
    return <Dialog open compact title={tr("The engine was not updated")} onClose={() => { setJobId(''); onClose(); }} actions={<Btn variant="primary" onClick={() => { setJobId(''); onClose(); }}>{tr("Close")}</Btn>}>
      <p>{tr(job.error ?? job.summary)}</p>
      {details(job)}
    </Dialog>;
  }
  if (job?.state === 'done') {
    return <Dialog open compact title={tr("The engine is updated")} onClose={() => { setJobId(''); onClose(); }} actions={<Btn variant="primary" onClick={() => { setJobId(''); onClose(); }}>{tr("Close")}</Btn>}>
      <p>{tr(job.summary)}</p>
    </Dialog>;
  }
  // 1: warn and ask.
  return <Dialog open compact title={tr("Update engine?")} onClose={close}
    actions={<><Btn onClick={close}>{tr("Cancel")}</Btn><Btn variant="primary" disabled={!loaded.length} onClick={() => void start()}>{tr("Stop and update")}</Btn></>}>
    <p>{tr("To update the engine, Strata has to stop the model that is loaded: {0}.", [names])}</p>
    <p className="small muted">{tr("The new engine downloads first while the model keeps answering. Then the model stops (answers in progress finish first), the engine is updated, and you can start the model again with the same settings. Your models and settings are kept.")}</p>
    {error ? <p className="small inline-error" role="alert">{tr(error)}</p> : null}
  </Dialog>;
}
