/**
 * The Read pictures checkbox on the Models screen: what it shows is the person's choice. The host's reports (the saved settings, whether the picture files
 * are prepared, whether the engine has a reader) only give it its first value for a version, and never overwrite a choice made by hand (GUI feedback 2, item 4:
 * the box used to flip off while Load ran and back on afterwards, because it was re-derived from the host's report as that report changed under the load).
 */
export interface InstalledPictures { visionEnabled?: boolean; visionPrepared?: boolean; visionDevice?: string; images?: string; /** the saved choice was made with a picture reader present (ui_vision_chosen, 0.2.14) */ visionChosen?: boolean }

/** The value a version opens with: off when the person turned it off and the files exist, or when they turned it off in 0.2.14 or later (the
 *  marker: QAGUI-STRIX 11-09, a model set up without the picture files came back ticked after a restart and Load fetched the 866 MB reader);
 *  otherwise on (the device they chose, or Automatic). An older off without the marker and without the files may be the one 0.2.9 saved while
 *  the engine had no reader: it still opens on, as before. */
export function openingVision(installed: InstalledPictures | undefined): string {
  if (!installed) return 'yes';
  if (installed.visionEnabled === false && (installed.visionPrepared !== false || installed.visionChosen === true)) return 'no';
  return installed.visionDevice ?? (installed.images && installed.images !== 'off' ? installed.images : 'yes');
}

/** The value to show for a version: the person's own choice for it when there is one, else what it opens with. */
export const shownVision = (picked: string | undefined, installed: InstalledPictures | undefined): string => picked ?? openingVision(installed);

/**
 * Whether the engine has no picture reader, as the row should show it. While a job runs the host is busy and its reader probe can drop out for a moment:
 * the answer given before the job started is kept until it ends, so the row never changes under the person's hands.
 */
export function steadyNoReader(now: boolean, jobRunning: boolean, last: { id: string; value: boolean } | undefined, id: string): { id: string; value: boolean } {
  return jobRunning && last && last.id === id ? last : { id, value: now };
}
