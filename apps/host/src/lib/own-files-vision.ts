/**
 * #4926: "Use a local GGUF" prepares the model with picture reading on or off. It used to be on only when the chosen folder itself held the pinned
 * picture file, so a person whose reader sits in Strata's models folder (it is shared by every model) got `--vision no` on every switch: the launch
 * config dropped the pictures and the first re-enable ran setup with `--vision no` again.
 *
 * Picture reading is on when the engine has a reader and either the folder holds a reader of the right kind (the checker's words for a file with
 * another name: Strata downloads the official one when it is missing), or a reader is already on this computer (an installed model has its picture files).
 */
export interface OwnFilesProjector { status: string; pinned?: boolean }
export interface OwnFilesPictures { visionPrepared?: boolean }

export function ownFilesVision(projectors: OwnFilesProjector[] | undefined, visionReader: boolean | undefined, installed: OwnFilesPictures[] | undefined): 'yes' | 'no' {
  if (!visionReader) return 'no';
  if ((projectors ?? []).some(p => (p.status === 'ok' && p.pinned) || p.status === 'other_name')) return 'yes';
  return (installed ?? []).some(m => m.visionPrepared === true) ? 'yes' : 'no';
}
