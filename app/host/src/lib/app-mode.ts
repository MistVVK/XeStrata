import { useEffect, useState } from 'react';
import { api } from './client';

/** Which window shows the page: the app window ('window'), a browser tab ('browser'), or neither (a host nobody started: null). */
export interface AppMode {
  mode: 'window' | 'browser' | null;
  reason: string | null;
  canQuit: boolean;
  appUpdate: 'available' | 'unavailable';
}

const NONE: AppMode = { mode: null, reason: null, canQuit: false, appUpdate: 'unavailable' };
let loading: Promise<AppMode> | null = null;
let known: AppMode | null = null;
/** The answer if it has already arrived (a click handler that must act at once uses it), otherwise null. */
export const appModeKnown = (): AppMode | null => known;

/** Asked once per page load; an older host without the route (404) or a failed call means "not the app": nothing extra is shown. */
export function loadAppMode(): Promise<AppMode> {
  loading ??= api.get<AppMode>('/api/app/mode').catch(() => NONE).then(m => (known = m));
  return loading;
}

export function useAppMode(): AppMode {
  const [mode, setMode] = useState<AppMode>(NONE);
  useEffect(() => { let live = true; void loadAppMode().then(m => { if (live) setMode(m); }); return () => { live = false; }; }, []);
  return mode;
}
