import { isAbsolute, resolve } from 'node:path';

export const name = 'strata-edit-recovery';
export const inject = ['workingDirectory'];

const missingText = /^(?:Error: )?old_string was not found in "([^"\r\n]+)"/u;
const guidance = 'Use Read on this file now. If the intended change is already present, verify it. Otherwise build the edit from the current contents and preserve other changes.';

const pathKey = path => process.platform === 'win32' ? resolve(path).toLowerCase() : resolve(path);

/** Look for a real Edit conflict or a later successful Read in this session.
 * Match the resolved path reported by the filesystem, even after a cwd change. */
function needsRead(log, target) {
  const outcomes = new Map();
  for (let i = log.length - 1; i >= 0; i--) {
    const { type, data } = log[i];
    if (type === 'tool/result') {
      const message = data?.message;
      const id = message?.toolCallId;
      if (!id) continue;
      if (message.isError === false && Array.isArray(data.meta?.lines)
        && typeof data.meta.path === 'string' && isAbsolute(data.meta.path) && pathKey(data.meta.path) === target) {
        outcomes.set(id, 'read');
      } else if (message.isError === true && Array.isArray(message.content)) {
        const file = message.content.find(part => part?.type === 'text' && missingText.test(part.text))?.text?.match(missingText)?.[1];
        if (file && isAbsolute(file) && pathKey(file) === target) outcomes.set(id, 'edit');
      }
    } else if (type === 'tool/call' && outcomes.get(data?.callId) === data?.name && outcomes.has(data?.callId)) {
      return data.name === 'edit';
    }
  }
  return false;
}

export function apply(ctx) {
  ctx.on('tools/pre-execute', (exec, next) => {
    const session = exec.agent?.session;
    const file = exec.arguments?.file_path;
    if (!session || !['edit', 'write'].includes(exec.name) || typeof file !== 'string' || !file.trim() || exec.signal.aborted) return next();
    const target = pathKey(resolve(ctx.workingDirectory.get(session), file));
    if (!needsRead(session.log, target)) return next();
    return { kind: 'deny', reason: 'A previous edit could not find its original text. ' + guidance };
  });

  ctx.on('tools/post-execute', async (exec, result, next) => {
    const decision = await next();
    if (decision.kind !== 'accept' || exec.name !== 'edit' || !result.isError || exec.signal.aborted
      || !missingText.test(result.error?.message)) return decision;
    return { ...decision, content: [...(decision.content ?? result.content), { type: 'text', text: guidance }] };
  });
}
