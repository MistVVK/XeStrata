/**
 * The pictures of earlier messages that go back to the model with a new message are read from the browser's own storage (IndexedDB).
 * When one of them is gone (browser data cleared, the chat opened in another browser or another app window profile), the whole send
 * used to fail with "... is no longer in browser storage. Attach it again.", and in that chat no message could ever be sent again,
 * because an earlier message cannot be given its picture back (GUI-TUI 0.2.17 QA). A picture that cannot be read is now left out of
 * the history and named, and the message goes on.
 */
export async function readHistoryPictures<T extends { name: string }>(
  photos: readonly T[],
  read: (photo: T) => Promise<{ url: string }>,
): Promise<{ found: { photo: T; url: string }[]; lost: string[] }> {
  const found: { photo: T; url: string }[] = [];
  const lost: string[] = [];
  for (const photo of photos) {
    try {
      found.push({ photo, url: (await read(photo)).url });
    } catch {
      lost.push(photo.name);
    }
  }
  return { found, lost };
}

/** The line added to an earlier message's text for pictures the page could not read, so the model knows they were there. */
export function lostPicturesNote(lost: readonly string[]): string {
  return lost.length ? `\n[Earlier pictures: ${lost.join(', ')}. They are no longer stored in this browser, so their pixels are not part of this turn.]` : '';
}
