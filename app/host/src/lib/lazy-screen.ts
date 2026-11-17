/**
 * Screens that are not needed to show the start screen load the first time they are used.
 *
 * `lazyScreen(load, pick)` is React.lazy over a dynamic import, for the routed screens.
 * `useLoadedModule(load, wanted)` is for surfaces that stay mounted and animate open (the Settings and set-up
 * dialogs): it returns the module once its code is here (null until then), and the caller opens the surface at
 * that point, so it appears exactly as before (same first frame, same focus target) instead of opening empty and
 * filling in. If a chunk cannot be fetched the error is thrown into render, and reaches the app's boundary as for
 * any other screen that fails to show.
 */
import { lazy, useEffect, useState, type ComponentType } from 'react';

export function lazyScreen<M, P extends object>(load: () => Promise<M>, pick: (module: M) => ComponentType<P>) {
  return lazy(() => load().then((module) => ({ default: pick(module) })));
}

const loaded = new WeakMap<object, unknown>();

export function useLoadedModule<M>(load: () => Promise<M>, wanted: boolean): M | null {
  const [module, setModule] = useState<M | null>(() => (loaded.get(load) as M | undefined) ?? null);
  const [failure, setFailure] = useState<unknown>(null);
  useEffect(() => {
    if (!wanted || module) return;
    let live = true;
    load().then((value) => { loaded.set(load, value); if (live) setModule(value); }, (error) => { if (live) setFailure(error ?? new Error('chunk failed')); });
    return () => { live = false; };
  }, [wanted, module, load]);
  if (failure) throw failure;
  return module;
}
