import { useEffect, useState } from 'react';
import { api } from './client';

export interface HelperMessage {
  id: string;
  text: string;
  at: string;
  delivery: 'queue' | 'steer';
}

export interface HelperRecord {
  archived?: boolean;
  messages?: HelperMessage[];
}

type HelperRecords = Record<string, HelperRecord>;
export type HelperAction = 'archive' | 'restore';
const empty: HelperRecords = {};
const changed = new EventTarget();
let channel: BroadcastChannel | undefined;

function helperChannel() {
  // Initialize only in the app. The same row is also rendered in server-side tests.
  if (!channel && typeof window !== 'undefined' && typeof BroadcastChannel === 'function') {
    channel = new BroadcastChannel('strata-helper-records');
    channel.onmessage = event => changed.dispatchEvent(new CustomEvent('change', { detail: event.data }));
  }
  return channel;
}

export function helperRecordsChanged(sessionId: string) {
  changed.dispatchEvent(new CustomEvent('change', { detail: sessionId }));
  helperChannel()?.postMessage(sessionId);
}

export function recordOf(records: HelperRecords, id: string): HelperRecord | undefined {
  return Object.hasOwn(records, id) ? records[id] : undefined;
}

export function useHelperRecords(sessionId?: string) {
  const [snapshot, setSnapshot] = useState<{ sessionId: string; records: HelperRecords }>();
  useEffect(() => {
    if (!sessionId) return;
    const id = sessionId;
    let request: AbortController | undefined;
    function read() {
      request?.abort();
      const current = new AbortController();
      request = current;
      void api.get<HelperRecords>('/api/chat/helpers?sessionId=' + encodeURIComponent(id), current.signal)
        .then(records => { if (!current.signal.aborted) setSnapshot({ sessionId: id, records }); })
        .catch(() => {});
    }
    function onChange(event: Event) {
      if ((event as CustomEvent).detail === sessionId) read();
    }
    helperChannel();
    changed.addEventListener('change', onChange);
    window.addEventListener('focus', read);
    read();
    return () => {
      request?.abort();
      changed.removeEventListener('change', onChange);
      window.removeEventListener('focus', read);
    };
  }, [sessionId]);
  return snapshot && snapshot.sessionId === sessionId ? snapshot.records : empty;
}

export async function updateHelperRecords(sessionId: string, action: HelperAction, helperIds: string[]) {
  await api.post('/api/chat/helpers', { sessionId, action, helperIds });
  helperRecordsChanged(sessionId);
}
