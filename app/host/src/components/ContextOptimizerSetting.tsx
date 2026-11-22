import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';
import { Btn } from './ui';

type OptimizerSettings = { enabled: boolean; baseUrl: string; model: string; timeoutMs: number };
const path = '/api/agent/context-optimizer';

export function ContextOptimizerSetting() {
  const [draft, setDraft] = useState<OptimizerSettings | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    api.get<OptimizerSettings>(path, controller.signal).then(settings => {
      setDraft(settings);
      setEnabled(settings.enabled);
    }).catch(() => {
      if (!controller.signal.aborted) setError(tr('Context Optimizer settings are unavailable.'));
    });
    return () => controller.abort();
  }, []);

  if (!draft) return <span className="xs muted" role="status">{error || tr('Loading…')}</span>;
  const edit = (patch: Partial<OptimizerSettings>) => {
    setDraft({ ...draft, ...patch });
    setNotice('');
    setError('');
  };
  const submit = async (test: boolean) => {
    setBusy(true);
    setError('');
    setNotice('');
    try {
      if (test) {
        const result = await api.post<{ ok: boolean; durationMs: number }>(path + '/test', draft);
        if (result.ok) setNotice(tr('Model test passed ({0} ms).', [result.durationMs]));
        else setError(tr('The local model did not return a usable selection. Check the server, model ID and timeout.'));
      } else {
        const saved = await api.post<OptimizerSettings>(path, draft);
        setDraft(saved);
        setEnabled(saved.enabled);
        setNotice(tr('Saved. The next assistant request applies this after active chats finish.'));
      }
    } catch (e) { setError(e instanceof Error ? tr(e.message) : tr('Context Optimizer settings are unavailable.')); }
    finally { setBusy(false); }
  };

  return <details>
    <summary>{tr('Configure local model')} · {tr(enabled ? 'On' : 'Off')}</summary>
    <fieldset className="stack" disabled={busy}>
      <label className="row">
        <input type="checkbox" checked={draft.enabled} onChange={e => edit({ enabled: e.target.checked })}/>
        {tr('Enable Context Optimizer')}
      </label>
      <label className="input-field"><span>{tr('Base URL')}</span>
        <input className="st-input" type="url" value={draft.baseUrl} placeholder="http://127.0.0.1:8080/v1" spellCheck={false}
          onChange={e => edit({ baseUrl: e.target.value.trim() })}/>
      </label>
      <label className="input-field"><span>{tr('Model ID')}</span>
        <input className="st-input" value={draft.model} spellCheck={false} onChange={e => edit({ model: e.target.value.trim() })}/>
      </label>
      <label className="input-field"><span>{tr('Timeout in seconds')}</span>
        <input className="st-input" type="number" min={1} max={60} step={1} value={draft.timeoutMs ? draft.timeoutMs / 1000 : ''}
          onChange={e => edit({ timeoutMs: Number(e.target.value) * 1000 })}/>
      </label>
      <span className="row">
        <Btn size="sm" disabled={!draft.baseUrl || !draft.model} onClick={() => void submit(true)}>{tr('Test model')}</Btn>
        <Btn size="sm" onClick={() => void submit(false)}>{tr('Save')}</Btn>
      </span>
      {notice && <span className="xs muted" role="status">{notice}</span>}
      {error && <span className="xs" role="alert">{error}</span>}
    </fieldset>
  </details>;
}
