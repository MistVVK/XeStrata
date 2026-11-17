/** The pictures on the clipboard, named so each one is told apart in the composer (a pasted screenshot arrives as "image.png"). */
export function pastedFiles(data: DataTransfer | null | undefined): File[] {
  const out: File[] = [];
  const found: File[] = Array.from(data?.files ?? []);
  // A screenshot from the clipboard (Win+Shift+S) often comes as an item and not in `files` (#3748).
  if (!found.some(f => f.type.startsWith('image/'))) {
    for (const item of Array.from(data?.items ?? [])) {
      if (item.kind !== 'file' || !item.type.startsWith('image/')) continue;
      const file = item.getAsFile();
      if (file) found.push(file);
    }
  }
  for (const file of found) {
    if (!file.type.startsWith('image/')) continue;
    const extension = file.type.split('/')[1]?.replace('jpeg', 'jpg').replace(/[+].*/, '') || 'png';
    out.push(new File([file], `Pasted picture ${new Date().toLocaleTimeString().replace(/[^0-9]+/g, '-').replace(/-$/, '')}${out.length ? `-${out.length + 1}` : ''}.${extension}`, { type: file.type }));
  }
  return out;
}
