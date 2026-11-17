/**
 * "Only connect to Strata on another computer" (#5529): the first-run engine card's way out for someone who will use a model on another
 * computer or an API. Nothing is downloaded: setup ends dismissed (the same as "Go straight to the app"), and the app opens Settings >
 * Models with a "Custom model API" connection already started. The engine card comes back by itself while setup is dismissed but not
 * complete and the engine is still missing (Setup.tsx), so adding the engine later needs no extra step.
 */

let wanted = false;

/** Ask the Models settings, when they open next, to start a Custom model API connection. */
export function requestCustomConnection() { wanted = true; }

/** True once for each request: the Models settings take it and start the connection. */
export function takeCustomConnectionRequest(): boolean {
  const was = wanted;
  wanted = false;
  return was;
}

export interface ConnectOnlyDeps {
  /** completeSetupStep(step, choices, dismissed) of the store. */
  completeSetupStep(step: number, choices: { onboarding: { step: number; manual: boolean } }, dismissed: boolean): Promise<void>;
  openSetup(open: boolean): void;
  setSettingsSection(section: 'Models'): void;
  /** navigate('chat', undefined, { settings: '1' }) of the router. */
  openSettings(): void;
}

/** The click: end setup dismissed, then open Settings > Models with the custom connection started. No engine download is started. */
export async function connectOnly(deps: ConnectOnlyDeps): Promise<void> {
  await deps.completeSetupStep(1, { onboarding: { step: 1, manual: true } }, true);
  deps.openSetup(false);
  requestCustomConnection();
  deps.setSettingsSection('Models');
  deps.openSettings();
}
