import { tr } from '../lib/i18n';
import { Select } from './ui';
import { CHAT_BUDGET_CHOICES, CHAT_EFFORT_CHOICES, setChatDefaults, useStore } from '../lib/store';

/** The composer's own names, so a default and the chat's own menu never show two words for one thing. */
const MODES: { value: string; label: string }[] = [
  { value: 'manual', label: 'Manual' },
  { value: 'plan', label: 'Plan' },
  { value: 'bypass', label: 'Bypass permissions' },
];
const BUDGET_LABELS: Record<number, string> = { 0: 'No limit', 2048: '2K', 8192: '8K', 32768: '32K' };

/** What every new conversation starts with (Settings → General). Each chat can still be changed inside itself. */
export function ChatDefaultsSetting() {
  const defaults = useStore(s => s.chatDefaults);
  return <div className="stack stack--sm">
    <Select ariaLabel={tr("What a new chat may do")} value={defaults.mode} options={MODES.map(m => ({ value: m.value, label: tr(m.label) }))} onChange={mode => setChatDefaults({ mode: mode as typeof defaults.mode })} />
    <Select ariaLabel={tr("How deeply a new chat thinks")} value={defaults.thinking} options={CHAT_EFFORT_CHOICES.map(e => ({ value: e, label: tr(e) }))} onChange={thinking => setChatDefaults({ thinking })} />
    <Select ariaLabel={tr("How long a new chat's answer may be")} value={String(defaults.thinkingBudget)} options={CHAT_BUDGET_CHOICES.map(v => ({ value: String(v), label: tr(BUDGET_LABELS[v]) }))} onChange={value => setChatDefaults({ thinkingBudget: Number(value) })} />
  </div>;
}
