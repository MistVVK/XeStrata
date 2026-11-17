import type { ToolCall } from './types';

/** The actions of one chat that wait for an answer, oldest first (the approval dock above the message box shows them, #4725). */
export function pendingApprovals(session: { messages: { id: string; tools?: ToolCall[] }[] } | null | undefined): { messageId: string; tool: ToolCall }[] {
    const out: { messageId: string; tool: ToolCall }[] = [];
    for (const m of session?.messages ?? [])
        for (const tool of m.tools ?? [])
            if (tool.state === 'proposed')
                out.push({ messageId: m.id, tool });
    return out;
}

/**
 * What the approval card's Details shows: every byte the agent asks to run (#5856). The card's preview is a short summary (cut at 600
 * characters, whitespace as given); for a command the full command comes from the call's own arguments, so ten long steps can all be
 * read before Approve. Anything else keeps the preview.
 */
export function approvalDetail(t: Pick<ToolCall, 'scope' | 'args' | 'preview'>): string {
    const command = t.scope === 'shell' ? t.args?.command : undefined;
    return typeof command === 'string' && command.length > 0 ? command : t.preview;
}
