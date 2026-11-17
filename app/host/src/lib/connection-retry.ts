/**
 * The connection banner's Retry only asks the same address again. When Retry has failed this many times in a row the host process itself is most likely gone (it was
 * stopped or crashed; the Windows and macOS app does not start it again), and the banner says so with the one step that helps: close Strata and open it again.
 */
export const STOPPED_AFTER_FAILED_RETRIES = 2;

export const hostSeemsStopped = (failedRetries: number): boolean => failedRetries >= STOPPED_AFTER_FAILED_RETRIES;

/** The count after one more Retry finished: a connection that is live again clears it, one that is still down adds one. */
export const failedRetriesAfter = (before: number, connection: string): number => (connection === 'live' ? 0 : before + 1);
