/**
 * #5066: how wide the text of an answer may run. Comfortable is what the app always did (56ch, about 66 characters a line); Wide is about 72ch, the top of the
 * 45-75 band the stylesheet aims at; Full width lets the text fill the column the way tables and code already do. The choice sets `data-measure` on the page
 * root, and the stylesheet moves its one token, --st-measure, so Focus reading (which sizes its column from the same token) follows it too.
 */
export type ReplyWidth = 'comfortable' | 'wide' | 'full';
export const REPLY_WIDTHS: readonly ReplyWidth[] = ['comfortable', 'wide', 'full'];
export const DEFAULT_REPLY_WIDTH: ReplyWidth = 'comfortable';
export const validReplyWidth = (value: unknown): ReplyWidth => REPLY_WIDTHS.includes(value as ReplyWidth) ? value as ReplyWidth : DEFAULT_REPLY_WIDTH;
