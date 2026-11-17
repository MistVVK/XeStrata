import type { ApiModel, ModelConnection, ModelProviders, RemoteModelSelection } from './types';

/**
 * The model of an API connection the person picked in the chat's header (a custom model API, or a provider), with the connection it belongs to; null when none is
 * picked or the pick is gone (its connection or model was removed). The picked remote model is a model to chat with the same as a local one: the sidebar chip,
 * the composer's placeholder and the "No model is loaded" line all ask this before they say there is none.
 */
export function chosenRemote(providers: ModelProviders | null | undefined, remote: RemoteModelSelection | null | undefined): { model: ApiModel; connection: ModelConnection } | null {
  if (!remote) return null;
  const connection = providers?.connections.find(c => c.id === remote.connectionId);
  const model = connection?.models.find(m => m.id === remote.modelId);
  return connection && model ? { model, connection } : null;
}

/** A model to chat with: a picked API model, the older runtime's loaded model, or a ready instance of the native engine. */
export function hasModelToChat(input: { remote: unknown; localLoaded: unknown; nativeReady: boolean }): boolean {
  return !!input.remote || !!input.localLoaded || input.nativeReady;
}

/** The plain line under the message box when text is typed and there is no model to chat with (never while an API model is picked). */
export function showNoModelHint(input: { streaming: boolean; canChat: boolean; hasText: boolean; hasPhotos: boolean; bareSlash: boolean; isTimeSkip: boolean }): boolean {
  return !input.streaming && !input.canChat && (input.hasText || input.hasPhotos) && !input.bareSlash && !input.isTimeSkip;
}
