import { closeSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { join } from 'node:path';
import { validChatId } from './chat-store.mjs';

export const validHelperId = id => typeof id === 'string' && /^[\w.:@-]{1,160}$/.test(id);
export const busyHelper = row => row?.status === 'working' || row?.status === 'queued';
export const recordFor = (records, id) => Object.hasOwn(records, id) ? records[id] : undefined;

/** Helper actions belong to the host. They survive chat saves from a window that has not seen them yet. */
export function createHelperRecords(root, now = () => new Date()) {
  const dir = join(root, 'helpers');
  const file = chatId => {
    if (!validChatId(chatId)) throw new Error('That is not a chat id.');
    return join(dir, chatId + '.json');
  };
  function read(chatId) {
    try { return JSON.parse(readFileSync(file(chatId), 'utf8')); }
    catch (error) { if (error.code !== 'ENOENT') throw error; return {}; }
  }
  function update(chatId, change) {
    const records = Object.assign(Object.create(null), read(chatId));
    change(records);
    mkdirSync(dir, { recursive: true, mode: 0o700 });
    const temp = file(chatId) + '.' + randomUUID() + '.tmp';
    try {
      const handle = openSync(temp, 'wx', 0o600);
      try { writeFileSync(handle, JSON.stringify(records), 'utf8'); fsyncSync(handle); }
      finally { closeSync(handle); }
      renameSync(temp, file(chatId));
    } finally {
      rmSync(temp, { force: true });
    }
    return records;
  }
  return {
    read,
    message(chatId, helperId, { messageId, delivery }, text) {
      const message = { id: messageId || randomUUID(), text, at: now().toISOString(), delivery };
      update(chatId, records => {
        const previous = recordFor(records, helperId);
        const messages = previous?.messages ?? [];
        if (!messages.some(item => item.id === message.id)) {
          records[helperId] = { ...previous, messages: [...messages, message] };
        }
      });
      return message;
    },
    archive(chatId, ids, archived) {
      return update(chatId, records => {
        for (const id of ids) {
          const previous = recordFor(records, id);
          if (!archived && !previous?.messages?.length) delete records[id];
          else records[id] = {
            ...(previous?.messages ? { messages: previous.messages } : {}),
            ...(archived ? { archived: true } : {}),
          };
        }
      });
    },
    forget: chatId => rmSync(file(chatId), { force: true }),
  };
}
