import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { useImeKeys } from '../lib/ime-keys';
import { tr } from '../lib/i18n';
import { Btn } from './ui';

interface ModelPortState {
  port: number | null;
  defaultPort: number;
  activePorts?: number[];
}

/** Preferred TCP port for local model servers (#6475). Empty uses the automatic 18500+ range and avoids port 8080 conflicts. */
export function ModelPortSetting() {
  const [state, setState] = useState<ModelPortState | null>(null);
  const [text, setText] = useState('');
  const [pending, setPending] = useState(false);
  const [notice, setNotice] = useState(false);
  const [problem, setProblem] = useState('');
  const ime = useImeKeys();

  useEffect(() => {
    let active = true;
    api.get<ModelPortState>('/api/native/model-port').then(value => {
      if (!active) return;
      setState(value);
      setText(value.port ? String(value.port) : '');
    }).catch(e => {
      if (active) setProblem(e instanceof Error ? e.message : String(e));
    });
    return () => { active = false; };
  }, []);

  const savedText = state?.port ? String(state.port) : '';
  async function commit() {
    if (!state || pending) return;
    const trimmed = text.trim();
    if (trimmed === savedText) { setProblem(''); return; }
    let next: number | null = null;
    if (trimmed !== '') {
      const value = Number(trimmed);
      if (!Number.isInteger(value) || value < 1024 || value > 65535) {
        setProblem(tr('Choose a port between 1024 and 65535, or leave it empty.'));
        return;
      }
      next = value;
    }
    setPending(true);
    setProblem('');
    setNotice(false);
    try {
      const updated = await api.post<ModelPortState>('/api/native/model-port', { port: next });
      setState(updated);
      setText(updated.port ? String(updated.port) : '');
      setNotice(true);
    } catch (e) {
      setProblem(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }

  return <span className="stack stack--sm">
    <span className="row">
      <input
        className="st-input"
        inputMode="numeric"
        size={7}
        aria-label={tr('Model server port (empty = automatic)')}
        placeholder={String(state?.defaultPort ?? 18500)}
        value={text}
        disabled={!state || pending}
        onChange={e => { setText(e.target.value.replace(/[^0-9]/g, '').slice(0, 5)); setNotice(false); setProblem(''); }}
        onBlur={() => void commit()}
        {...ime.events}
        onKeyDown={e => { if (ime.owns(e)) return; if (e.key === 'Enter') void commit(); }}
      />
      <Btn size="sm" disabled={!state || pending || text.trim() === savedText} onClick={() => void commit()}>{tr('Save')}</Btn>
      {notice && <span className="xs muted" role="status">{tr('Saved')}</span>}
    </span>
    {state?.activePorts?.length ? <span className="xs muted">{tr('Active model port: {0}.', [state.activePorts.join(', ')])}</span> : null}
    {problem ? <span className="xs inline-error" role="alert">{problem}</span> : null}
  </span>;
}
