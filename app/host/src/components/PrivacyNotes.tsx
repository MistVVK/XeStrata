import { tr } from '../lib/i18n';

/** What Strata collects and where things go: one compact paragraph (the six sentences read together, translated as one in every language). */
const NOTES = "Strata sends no usage telemetry. Conversations stay on this computer, in Strata's data folder. When you choose an API model, that conversation and enabled skill instructions are sent to its provider. The UI request log keeps client, timing and token metadata in memory. Native API monitoring can retain request content in memory when enabled. Connected tools access the resources you configure.";

export function PrivacyNotes() {
  return <div className="privacy-notes" aria-label={tr("Collection and logs")}>
    <p>{tr(NOTES)}</p>
  </div>;
}
