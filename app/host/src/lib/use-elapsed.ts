import { useEffect, useState } from 'react';

/** Milliseconds since `since` (an ISO time), re-read every second while `on`; null when there is no start time. */
export function useElapsed(since: string | undefined, on = true): number | null {
  const start = since ? Date.parse(since) : NaN;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!on || !Number.isFinite(start)) return undefined;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [on, start]);
  return Number.isFinite(start) ? Math.max(0, now - start) : null;
}
