/** The host's design-mode placeholder for the engine version; it is not an engine anyone installed. */
const PLACEHOLDER_ENGINE = /^strata-engine 0\.1\.0$/;
/** The engine version the app shows: the installed engine's once the host has read it; until then nothing (never the placeholder). */
export function shownEngineVersion(native: { available?: boolean; installed?: { engine?: { version?: string | null } | null } | null } | null | undefined, storeVersion: string | undefined): string | undefined {
  if (native?.available) return native.installed?.engine?.version ?? undefined;
  return storeVersion && !PLACEHOLDER_ENGINE.test(storeVersion) ? storeVersion : undefined;
}

