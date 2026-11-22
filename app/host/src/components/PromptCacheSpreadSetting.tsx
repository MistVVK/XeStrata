import { useEffect, useState } from 'react';
import { api } from '../lib/client';
import { tr } from '../lib/i18n';

interface PromptCacheSpreadState {
  enabled: boolean;
  customized: boolean;
}

/** Toggles --prompt-cache-spread for desktop local models so agent compaction resumes near the cut (#6469). */
export function PromptCacheSpreadSetting() {
  const [state, setState] = useState<PromptCacheSpreadState | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    api.get<PromptCacheSpreadState>('/api/native/prompt-cache-spread').then(value => {
      if (active) setState(value);
    }).catch(e => {
      if (active) setError(e instanceof Error ? e.message : String(e));
    });
    return () => { active = false; };
  }, []);

  async function toggle() {
    if (!state || pending) return;
    setPending(true);
    setError('');
    try {
      setState(await api.post<PromptCacheSpreadState>('/api/native/prompt-cache-spread', { enabled: !state.enabled }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }

  return <span className="stack stack--sm">
    <button
      className="st-toggle"
      role="switch"
      aria-checked={state?.enabled ?? false}
      aria-label={tr('Spread prompt cache checkpoints')}
      disabled={!state || pending}
      onClick={() => void toggle()}
    />
    {error ? <span className="xs inline-error" role="alert">{error}</span> : null}
  </span>;
}
