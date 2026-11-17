/** The default reading size and interface size (owner, 11-01: 12 px and 90%; they were 14 px and 100%). */
export const DEFAULT_CONTENT_FS = 12;
export const DEFAULT_UI_SCALE = 0.9;
export const UI_SCALE_CHOICES = [0.9, 1, 1.1, 1.25, 1.5, 1.75, 2] as const;
/** Saves made while the old defaults were in force carry no `sizeDefaults` mark. */
export const SIZE_DEFAULTS_VERSION = 2;
const OLD_CONTENT_FS = 14;
const OLD_UI_SCALE = 1;

/** The two sizes to use for a saved preferences record. A save without the mark is not a choice when it holds the old default (14 px, 100%): that
 *  becomes the new default once. Any other value is a size the person picked and stays. A save with the mark is kept as it is. Nothing saved: the defaults. */
export function savedSizes(saved: { contentFontSize?: unknown; uiScale?: unknown; sizeDefaults?: unknown } | null | undefined, min: number, max: number) {
  const fs = Number(saved?.contentFontSize), scale = Number(saved?.uiScale);
  const marked = saved?.sizeDefaults === SIZE_DEFAULTS_VERSION;
  const fsOk = Number.isFinite(fs) && fs > 0, scaleOk = (UI_SCALE_CHOICES as readonly number[]).includes(scale);
  return {
    contentFontSize: !fsOk || (!marked && fs === OLD_CONTENT_FS) ? DEFAULT_CONTENT_FS : Math.min(max, Math.max(min, fs)),
    uiScale: !scaleOk || (!marked && scale === OLD_UI_SCALE) ? DEFAULT_UI_SCALE : scale,
  };
}
