import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';
import { Select } from './ui';
import { validHelperLimit } from '../lib/helper-limit';

/** Settings > General > Helper time limit (#5996): Off (the default), 30 minutes, 1 hour or 2 hours. The host reads it live, so it applies to helpers already running. */
export function HelperLimitSetting() {
  const [value, setValue] = useState<string | null>(null);
  const [problem, setProblem] = useState('');
  useEffect(() => {
    let alive = true;
    api.get<{ helperLimitMin: number | null }>('/api/agent/helper-limit').then(v => { if (alive) setValue(String(validHelperLimit(v.helperLimitMin) ?? 'off')); }).catch(() => { if (alive) setValue('off'); });
    return () => { alive = false; };
  }, []);
  if (value === null) return <span className="xs muted">{tr("Loading…")}</span>;
  const save = async (next: string) => {
    try { setProblem(''); const saved = await api.post<{ helperLimitMin: number | null }>('/api/agent/helper-limit', { helperLimitMin: next === 'off' ? null : Number(next) }); setValue(String(validHelperLimit(saved.helperLimitMin) ?? 'off')); }
    catch (error) { setProblem(error instanceof Error ? error.message : tr("The choice was not saved.")); }
  };
  return <span className="stack">
    <Select ariaLabel={tr("Helper time limit")} value={value} options={[{ value: 'off', label: 'Off' }, { value: '30', label: '30 minutes' }, { value: '60', label: '1 hour' }, { value: '120', label: '2 hours' }]} onChange={v => void save(v)}/>
    {problem ? <span className="xs" role="alert">{problem}</span> : null}
  </span>;
}
