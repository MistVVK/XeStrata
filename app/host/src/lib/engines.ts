import { useCallback, useEffect, useState } from 'react';
import { api } from './client';
import type { EngineList, InstalledEngine } from './engine-list';

export * from './engine-list';

/** The engine list from the host, read when the page opens and again after every switch or delete. */
export function useEngineList() {
  const [list, setList] = useState<EngineList | null>(null);
  const [failed, setFailed] = useState(false);
  const reload = useCallback(async () => {
    try {
      setList(await api.get<EngineList>('/api/engines'));
      setFailed(false);
    } catch {
      setFailed(true);
    }
  }, []);
  useEffect(() => { void reload(); }, [reload]);
  return { list, failed, reload };
}

/** Switch to (`use`) or delete (`remove`) an installed engine. Rejects with the host's plain sentence. */
export async function changeInstalledEngine(action: 'use' | 'remove', engine: InstalledEngine): Promise<void> {
  await api.post(`/api/engines/${action}`, { family: engine.family, folder: engine.folder });
}
