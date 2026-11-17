import { tr } from '../lib/i18n';
import { Btn, Icon } from './ui';
import type { NativeState } from '../lib/native';

/** The app could not find or recognise an engine for this computer (none installed and none in the feed, another card family, a driver that is too old). It still downloads the model
 *  (owner decision 2026-10-18) and says so before and during the download; loading says the same with the next step. */
export const engineMissing = (native: NativeState | null | undefined) => !!native?.engine && native.engine.usable === false;

/** `onSetup` (the Models page passes it): a way to the guided setup, whose first card downloads the engine (#5529: someone who chose "Only connect to Strata on another computer" adds the engine later from here). */
export function EngineWarning({ native, onSetup }: { native: NativeState | null | undefined; onSetup?: () => void }) {
  if (!engineMissing(native)) return null;
  if (native?.engine?.wrongMake && native.engine.notice) return <div className="engine-warning" role="status"><Icon name="info" size={18}/><div><p>{native.engine.notice}</p></div></div>;
  return <div className="engine-warning" role="status"><Icon name="info" size={18}/><div><p>{tr("Strata didn’t recognise an engine for this computer. The model will download, but it may not run until the right engine is installed (Settings → About).")}</p>{native?.engine?.notice ? <small className="muted">{native.engine.notice}</small> : null}{onSetup ? <p><Btn size="sm" onClick={onSetup}>{tr("Run guided setup")}</Btn></p> : null}</div></div>;
}
