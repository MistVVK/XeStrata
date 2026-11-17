// Packaged-app walkthrough (GUI-VIS-1117 A10): the first screen with the update feed unreachable said "Strata can't pick an engine for this PC from its update list.
// The update didn't finish and nothing was changed ... Press Update Strata in Settings". A first run is not an update. Control's own words for "no internet"
// (strata_control/netpolicy.py update_offline / update_again) were recognised by neither looksOffline nor plainUpdateError, so the offline card that exists was never shown.
import test from 'node:test';
import assert from 'node:assert/strict';
import { engineFirstRun } from '../server/updates.mjs';
import { looksOffline, plainUpdateError } from '../server/update-messages.mjs';

// the words exactly as strata_control/netpolicy.py WORDS formats them
const UPDATE_OFFLINE = 'this computer could not connect to github.com (it looks offline), so the update check could not run. Strata keeps working with what is installed.';
const UPDATE_AGAIN = 'github.com did not answer a moment ago, so the update check was not tried again. Strata keeps working with what is installed.';
const UPDATE_NO_ANSWER = 'github.com did not answer within 15 seconds, so the update check could not run. Check that this computer can open https://github.com, then try again. Strata keeps working with what is installed.';
const gpus = [{ name: 'NVIDIA GeForce RTX 5070', vramGb: 12 }];
const failing = text => ({ updateCheck: async () => { throw Object.assign(new Error(text), { exitCode: 6 }); } });

test('Control\'s own offline and no-answer words are an offline person, not an update error', async () => {
  for (const words of [UPDATE_OFFLINE, UPDATE_AGAIN, UPDATE_NO_ANSWER]) {
    const info = await engineFirstRun({ control: failing(words), gpus });
    assert.equal(info.status, 'offline', words.slice(0, 50));
  }
  assert.ok(looksOffline(new Error(UPDATE_OFFLINE)));
  assert.equal(plainUpdateError(new Error(UPDATE_OFFLINE)).headline, 'No internet connection.');
  assert.match(plainUpdateError(new Error(UPDATE_AGAIN)).headline, /Couldn't reach the update server\./);
});

test('any other first-run error is status error, and its message carries no "Update Strata in Settings"', async () => {
  const info = await engineFirstRun({ control: failing('the feed says something nobody planned for'), gpus });
  assert.equal(info.status, 'error');
  assert.ok(!/Settings|Update Strata/.test(info.message), info.message);
  assert.match(info.details, /nobody planned for/, 'the raw reason is kept for Details');
});
