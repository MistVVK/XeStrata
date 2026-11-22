import { tr } from '../lib/i18n';
import { Btn, Icon } from './ui';
import type { NativeState } from '../lib/native';

const SHAPE_BY_FAMILY: Record<string, string> = {
  'inkling-2-mini': 'inkling2',
  'hy4-flash': 'hy4',
};

/** True when `family` runs on its own shape engine (e.g. `strata-inkling` or `strata-hy4`) and that shape engine or a ready model of that family is already installed. */
export const shapeEngineUsable = (native: NativeState | null | undefined, family?: string | null): boolean => {
  if (!family) return false;
  const shape = SHAPE_BY_FAMILY[family];
  if (!shape) return false;
  if ((native?.engine?.shapes ?? []).some(s => s.shape === shape && s.usable === true)) return true;
  const prefix = `${family}-`;
  return (native?.installed?.models ?? []).some(m => m.ready === true && m.model.toLowerCase().startsWith(prefix));
};

/** The app could not find or recognise an engine for this computer (none installed and none in the feed, another card family, a driver that is too old). It still downloads the model
 *  (owner decision 2026-10-18) and says so before and during the download; loading says the same with the next step.
 *  Suppressed for shape-engine families (`inkling-2-mini`, `hy4-flash`) when their shape engine is already installed and usable. */
export const engineMissing = (native: NativeState | null | undefined, family?: string | null) =>
  !!native?.engine && native.engine.usable === false && !shapeEngineUsable(native, family);

/** `onSetup` (the Models page passes it): a way to the guided setup, whose first card downloads the engine (#5529: someone who chose "Only connect to Strata on another computer" adds the engine later from here). */
export function EngineWarning({ native, family, onSetup }: { native: NativeState | null | undefined; family?: string | null; onSetup?: () => void }) {
  if (!engineMissing(native, family)) return null;
  if (native?.engine?.wrongMake && native.engine.notice) return <div className="engine-warning" role="status"><Icon name="info" size={18}/><div><p>{native.engine.notice}</p></div></div>;
  return <div className="engine-warning" role="status"><Icon name="info" size={18}/><div><p>{tr("Strata didn’t recognise an engine for this computer. The model will download, but it may not run until the right engine is installed (Settings → About).")}</p>{native?.engine?.notice ? <small className="muted">{native.engine.notice}</small> : null}{onSetup ? <p><Btn size="sm" onClick={onSetup}>{tr("Run guided setup")}</Btn></p> : null}</div></div>;
}
