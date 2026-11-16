/**
 * #5661: the Serving switch that lets apps on this computer use a model's MCP tools through the API (serving.mjs mcpToolsRefusal holds the
 * other guards). Off by default; it needs an API key, so without one it is off and disabled and says to create a key first.
 */
import type { ServingState } from './types';

export interface McpToolsSwitch { checked: boolean; disabled: boolean; hint: 'help' | 'needs-key' }

export function mcpToolsSwitch(serving: Pick<ServingState, 'access' | 'mcpTools'> | null | undefined, model: string): McpToolsSwitch {
  const keySet = !!serving?.access?.apiKeySet;
  return { checked: keySet && !!serving?.mcpTools?.includes(model), disabled: !keySet, hint: keySet ? 'help' : 'needs-key' };
}

/** The PATCH /api/serving body that turns it on or off for one model. */
export const mcpToolsPatch = (model: string, on: boolean) => ({ mcpTools: { model, on } });
