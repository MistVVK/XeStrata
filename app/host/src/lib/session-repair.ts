import type { Session } from './types';

/**
 * A chat read from a saved file or the browser's old copy, as the page can safely work with it. A damaged, hand-edited or older file may lack fields the lists read
 * (a missing `updatedAt` made `localeCompare` throw and blanked the whole chat page; a missing title or message text made a `.trim()` throw). Every text field is
 * made a string, a missing date takes the other date (or stays empty and sorts last), a missing title is the existing "New conversation", and a message that is
 * not an object is dropped. Whatever is well-formed is returned as it was.
 */
const text = (value: unknown) => (typeof value === 'string' ? value : '');
const isObject = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);

export function repairSession(session: Session): Session {
  const raw = session as unknown as Record<string, unknown>;
  const createdAt = text(raw.createdAt), updatedAt = text(raw.updatedAt) || createdAt;
  const hasTitle = typeof raw.title === 'string' && raw.title.trim() !== '';
  const messages = (Array.isArray(raw.messages) ? raw.messages : []).filter(isObject).map((message, i) => {
    const m = message as Record<string, unknown>;
    const fixed: Record<string, unknown> = { ...m, id: typeof m.id === 'string' ? m.id : `m${i}`, role: typeof m.role === 'string' ? m.role : 'assistant', content: text(m.content), at: text(m.at) };
    if (m.model !== undefined && typeof m.model !== 'string') delete fixed.model;
    if (m.reasoning !== undefined && typeof m.reasoning !== 'string') delete fixed.reasoning;
    return fixed;
  }) as unknown as Session['messages'];
  return {
    ...session,
    title: hasTitle ? (raw.title as string) : 'New conversation',
    ...(hasTitle ? {} : { defaultTitle: true }),
    createdAt: createdAt || updatedAt,
    updatedAt,
    workspace: text(raw.workspace),
    messages,
    ...(isObject(raw.preferences) || raw.preferences === undefined ? {} : { preferences: undefined }),
  };
}
