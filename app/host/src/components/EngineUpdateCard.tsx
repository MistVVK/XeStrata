import { useState } from 'react';
import { tr } from '../lib/i18n';
import { Btn, Icon } from './ui';
import { nativeAction, type NativeState } from '../lib/native';

/** Strata Control's last check found a newer engine than the one installed: its numbers come from the
 *  check kept on this PC (`update_available` of `strata components engine`), so showing the card reads no network. After an app or Control update this is how a
 *  person with an older engine is told (#4848). */
export const engineUpdateOffered = (native: NativeState | null | undefined) => {
  const offer = native?.engine?.updateAvailable;
  return !!(offer && offer.engine);
};

/** One card, plain words, one button. The update itself is the existing "Update engine" job (its progress shows where every job's does); models are not touched. */
export function EngineUpdateCard({ native, busy }: { native: NativeState | null | undefined; busy: boolean }) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  if (!engineUpdateOffered(native)) return null;
  const running = (native?.instances.length ?? 0) > 0;
  async function update() {
    setPending(true);
    setError('');
    try {
      await nativeAction('operations', { kind: 'update-engine' });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }
  return <div className="engine-warning engine-update" role="status">
    <Icon name="info" size={18}/>
    <div>
      <p>{tr("A newer engine is available for this computer. Your models and settings stay as they are.")}</p>
      {running ? <small>{tr("Unload your running models first.")}</small> : null}
      {error ? <small className="inline-error" role="alert">{error}</small> : null}
    </div>
    <Btn size="sm" variant="primary" disabled={busy || pending || running} onClick={() => void update()}>{tr("Update engine")}</Btn>
  </div>;
}
