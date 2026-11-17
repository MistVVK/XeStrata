import assert from 'node:assert/strict';
import { statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { currentUserSid } from '../../server/private-file.mjs';

/** The effective protection of a file: mode 0600 on POSIX, the exact DACL (one protected entry for this user) on NTFS.
 *  The DACL is read through .NET (File.GetAccessControl), not Get-Acl: Get-Acl lives in a module Windows PowerShell has to load, and a
 *  Windows PowerShell started under PowerShell 7 (GitHub's Windows runners) failed to load it ("the module could not be loaded").
 *  The entries are compared by SID, not as SDDL text: SDDL writes some accounts as an alias (the built-in Administrator, which GitHub's
 *  runners run as, is "LA", not its S-1-5-21-...-500). */
export function assertPrivate(path) {
  if (process.platform !== 'win32') { assert.equal(statSync(path).mode & 0o777, 0o600); return; }
  const script = `$a = [System.IO.File]::GetAccessControl('${path.replace(/'/g, "''")}'); 'protected=' + $a.AreAccessRulesProtected; `
    + `$a.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier]) | ForEach-Object { $_.IdentityReference.Value + '|' + $_.FileSystemRights + '|' + $_.AccessControlType + '|inherited=' + $_.IsInherited }; `
    + `'sddl=' + $a.GetSecurityDescriptorSddlForm('Access')`;
  const lines = execFileSync(join(process.env.SystemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe'), ['-NoProfile', '-NonInteractive', '-Command', script],
    { encoding: 'utf8', timeout: 30000, windowsHide: true }).split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  const sddl = lines.find(line => line.startsWith('sddl=')) ?? '';
  const facts = lines.filter(line => !line.startsWith('sddl='));
  assert.deepEqual(facts, ['protected=True', `${currentUserSid()}|FullControl|Allow|inherited=False`], `ACL is not owner-only: ${sddl}`);
}
