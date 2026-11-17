/** A file that is not a picture or plain text (a PDF) is saved by the host in the chat's folder and named in the next message, so the assistant can open it (#3838). */
export async function uploadChatFile(file: File, sessionId: string, fetchFn: typeof fetch = fetch): Promise<{ name: string }> {
  const host = (globalThis as { __STRATA_HOST__?: string }).__STRATA_HOST__ ?? '';
  const res = await fetchFn(`${host}/api/chat/upload?sessionId=${encodeURIComponent(sessionId)}&name=${encodeURIComponent(file.name)}`, { method: 'POST', headers: { 'content-type': file.type || 'application/octet-stream' }, body: file });
  const data = await res.json().catch(() => ({})) as { files?: { name: string; attached?: boolean }[]; error?: string };
  if (!res.ok) throw new Error(data.error || 'The file could not be added. Try again.');
  return { name: data.files?.find(f => f.attached)?.name ?? file.name };
}

/** Types the Linux file dialog needs next to the extensions (WebKitGTK reads only the type part of `accept`). */
export const FILE_ACCEPT = 'image/*,application/pdf,text/*,application/json,application/xml,.pdf,.txt,.md,.markdown,.csv,.ts,.tsx,.js,.jsx,.py,.json,.css,.rs,.go,.sh,.html,.htm,.yaml,.yml,.toml,.xml,.log';
