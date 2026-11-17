import type { Session, Workspace } from './types';

/** The workspace a chat was made in: the sidebar groups chats by this same rule (the saved path, or an earlier folder of the workspace). */
export function workspaceIdForSession(workspaces: Workspace[], session: Pick<Session, 'workspace'> | null | undefined): string | null {
  const path = session?.workspace;
  if (!path) return null;
  return workspaces.find(w => w.path === path || !!w.aliases?.includes(path))?.id ?? null;
}

export type MoveRefusal = 'working' | 'no-chat' | 'no-workspace' | 'same';

/**
 * Move a chat to another workspace (#6051): its saved workspace becomes the destination's folder, so the sidebar lists it there
 * and its next turn works there (a folder chosen for the chat itself stays). A working chat is refused, so a running turn
 * never changes folder under its feet; an unknown chat or destination, or the workspace it is already in, changes nothing.
 */
export function moveSessionToWorkspace<S extends Pick<Session, 'id' | 'workspace'>>(
  sessions: readonly S[], chatId: string, workspaces: readonly Workspace[], workspaceId: string, working: ReadonlySet<string> = new Set(),
): { sessions: S[]; refused: MoveRefusal | null } {
  const unchanged = (refused: MoveRefusal) => ({ sessions: [...sessions], refused });
  const chat = sessions.find(s => s.id === chatId);
  if (!chat) return unchanged('no-chat');
  const target = workspaces.find(w => w.id === workspaceId);
  if (!target) return unchanged('no-workspace');
  if (working.has(chatId)) return unchanged('working');
  if (workspaceIdForSession([...workspaces], chat) === target.id) return unchanged('same');
  return { sessions: sessions.map(s => (s.id === chatId ? { ...s, workspace: target.path } : s)), refused: null };
}
