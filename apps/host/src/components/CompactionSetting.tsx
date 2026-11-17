import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';
import { Select } from './ui';

/**
 * When the assistant summarises the older part of a chat (the compaction point). What the person chooses here is written
 * into the assistant's profile as `thresholdRatio` / `headroomTokens`; the number that really answers is the smaller of the
 * two, so the line under the controls says it for the model that is loaded now.
 */
interface CompactionView {
  choice: { thresholdRatio: number | null; headroomFraction: number | null };
  effective: { thresholdRatio: number; headroomFraction: number };
  defaults: { thresholdRatio: number; headroomFraction: number };
  source: { thresholdRatio: string; headroomFraction: string };
  model: { name: string; context: number; thresholdTokens: number; percent: number } | null;
}

/** The choices offered as whole percents; the default of 80 is named so the person can find their way back. */
const AT = [60, 70, 75, 80, 85, 90, 95] as const;
/** How much of the room the conversation has is kept free before summarising: less room means it starts later. */
const RESERVE = [{ value: '0.25', label: 'Much more' }, { value: '0.125', label: 'Strata’s usual' }, { value: '0.06', label: 'Less' }, { value: '0.02', label: 'Almost none' }] as const;
const grouped = (n: number) => n.toLocaleString('en-US');

export function CompactionSetting() {
  const [view, setView] = useState<CompactionView | null>(null);
  const [problem, setProblem] = useState('');
  const [advanced, setAdvanced] = useState(false);
  useEffect(() => {
    let alive = true;
    api.get<CompactionView>('/api/agent/compaction').then(v => { if (alive) setView(v); }).catch(() => undefined);
    return () => { alive = false; };
  }, []);
  if (!view) return <span className="xs muted">{tr("Loading…")}</span>;
  const save = async (patch: Record<string, number | null>) => {
    try { setProblem(''); setView(await api.post<CompactionView>('/api/agent/compaction', patch)); }
    catch (error) { setProblem(error instanceof Error ? error.message : tr("The choice was not saved.")); }
  };
  const atValue = view.choice.thresholdRatio === null ? 'auto' : String(Math.round(view.choice.thresholdRatio * 100));
  const reserveValue = view.choice.headroomFraction === null ? 'auto' : String(view.choice.headroomFraction);
  /** A saved number that is not one of the offered ones (an environment value, an older choice) still has to show. */
  const options = (offered: { value: string; label: string }[], current: number | null, label: (n: number) => string) =>
    current !== null && !offered.some(o => o.value === String(current)) ? [...offered, { value: String(current), label: label(current) }] : offered;
  const atOptions = options(AT.map(p => ({ value: String(p), label: `${p}%` })), view.choice.thresholdRatio, n => `${Math.round(n * 100)}%`);
  const reserveOptions = options(RESERVE.map(r => ({ value: r.value, label: tr(r.label) })), view.choice.headroomFraction, n => `${Math.round(n * 100)}%`);
  return <span className="stack">
    <span className="row">
      <Select translateLabels={false} ariaLabel={tr("Start summarizing older messages")} value={atValue}
        options={[{ value: 'auto', label: `${tr("Automatic")} (${Math.round(view.defaults.thresholdRatio * 100)}%)` }, ...atOptions]}
        onChange={value => void save({ thresholdRatio: value === 'auto' ? null : Number(value) / 100 })}/>
    </span>
    <details className="xs" open={advanced} onToggle={e => setAdvanced(e.currentTarget.open)}>
      <summary>{tr("Advanced: room kept free")}</summary>
      <span className="row" style={{ marginTop: 6 }}>
        <Select translateLabels={false} ariaLabel={tr("Room kept free before summarizing")} value={reserveValue}
          options={[{ value: 'auto', label: `${tr("Automatic")} (${Math.round(view.defaults.headroomFraction * 100)}%)` }, ...reserveOptions]}
          onChange={value => void save({ headroomFraction: value === 'auto' ? null : Number(value) })}/>
      </span>
      <span className="xs muted">{tr("Strata keeps some of the window free so a summary and an answer still fit. Less room starts summarizing later.")}</span>
    </details>
    <span className="xs muted">
      {view.model
        ? tr("On {0} ({1} tokens) that starts at about {2} tokens ({3}% of the window).", [view.model.name, grouped(view.model.context), grouped(view.model.thresholdTokens), view.model.percent])
        : tr("Load a model to see the number for your window.")}
    </span>
    {problem ? <span className="xs" role="alert">{problem}</span> : null}
  </span>;
}
