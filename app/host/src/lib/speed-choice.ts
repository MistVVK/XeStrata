/**
 * A speed switch with a value to choose instead of on / off (owner, 0.2.16: STRATA_ROUTE_TAIL_SKIP in Expert settings). The choices come from
 * strata_control/speed_switches.py through policy-generated.mjs, so the app's dropdown and the Terminal's sentence are one list.
 * The default ('') writes no entry at all: the engine's own rule. Every other choice writes its value. A value the list does not know (a file
 * edited by hand) shows as the default and is left as it is until a choice is made.
 */
export interface ChoiceSwitch {
  env: Record<string, string>;
  words?: { on: string; off: string };
  choices?: [string, string][];
}

const nameOf = (sw: ChoiceSwitch) => Object.keys(sw.env)[0];

/** The picked choice for this environment: its value, or '' for the default. */
export function choiceValue(sw: ChoiceSwitch, env: Record<string, unknown> | undefined): string {
  const now = env?.[nameOf(sw)];
  const text = now == null ? '' : String(now);
  return (sw.choices ?? []).some(([v]) => v === text) ? text : '';
}

/** A choice whose word is its own value (a number), which carries the switch's note. */
export const isNumberChoice = (sw: ChoiceSwitch, value: string) => !!value && (sw.choices ?? []).some(([v, w]) => v === value && w === v);

/** The dropdown: the default first (the switch's "on" word, "Automatic (recommended)"), then the choices in their order. */
export const choiceOptions = (sw: ChoiceSwitch) => [{ value: '', label: sw.words?.on ?? 'Automatic (recommended)' }, ...(sw.choices ?? []).map(([value, label]) => ({ value, label }))];

/** The environment with this choice: the default removes the entry, a choice writes its value; nothing else changes. */
export function withChoice(sw: ChoiceSwitch, env: Record<string, unknown> | undefined, value: string): Record<string, unknown> {
  const next = { ...(env ?? {}) };
  if (value && (sw.choices ?? []).some(([v]) => v === value)) next[nameOf(sw)] = value;
  else delete next[nameOf(sw)];
  return next;
}
