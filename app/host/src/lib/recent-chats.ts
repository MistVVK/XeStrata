import type { Session } from './types';

/** The time a chat was last used, as text that sorts: a missing or odd value is empty, which sorts last (a damaged or older chat file must never take the page down). */
const activityOf = (session: Session) => (typeof session?.updatedAt === 'string' ? session.updatedAt : '');

export function recentChats(sessions: Session[], activeId?: string | null) {
  const ordered = sessions.filter(session => session && !session.archived)
    .sort((a, b) => activityOf(b).localeCompare(activityOf(a)));
  const current = ordered.find(session => session.id === activeId);
  const latest = ordered.filter(session => session.id !== current?.id).slice(0, 5);
  const ids = new Set([...latest, ...(current ? [current] : [])].map(session => session.id));
  // Activity order keeps number shortcuts stable when simply switching chats.
  return ordered.filter(session => ids.has(session.id));
}

export function recentChatDescription(session: Session) {
  const messages = Array.isArray(session?.messages) ? session.messages.filter(message => !!message && typeof message === 'object') : [];
  const message = messages.filter(message => message.role === 'user').at(-1) ?? messages.at(-1);
  const text = (typeof message?.content === 'string' ? message.content : '').replace(/\s+/g, ' ').trim() || 'No messages yet';
  return text.length > 160 ? `${text.slice(0, 157).trimEnd()}…` : text;
}
