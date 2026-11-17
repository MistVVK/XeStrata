/**
 * Import this FIRST in a test that builds the host's native control or starts the host: it points the per-user places the host writes to
 * (the settings file Control reads, the components folder, the app's local data) at a scratch folder.
 * On 2026-10-24 a host test run rewrote the owner's real %LOCALAPPDATA%\Strata\app-settings.json (its data_dir became a deleted test folder).
 */
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

if (!process.env.STRATA_TEST_KEEP_PROFILE) {
  const scratch = mkdtempSync(join(tmpdir(), 'strata-host-test-profile-'));
  process.env.STRATA_SETTINGS_FILE = join(scratch, 'app-settings.json');
  process.env.STRATA_COMPONENTS_DIR = join(scratch, 'components');
  process.env.LOCALAPPDATA = join(scratch, 'Local');
  process.env.APPDATA = join(scratch, 'Roaming');
  process.env.XDG_DATA_HOME = join(scratch, 'xdg-data');
  process.env.XDG_STATE_HOME = join(scratch, 'xdg-state');
  process.env.XDG_CONFIG_HOME = join(scratch, 'xdg-config');
}
