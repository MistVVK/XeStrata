// 0217 lane FIX-APP, #3749: on a German Debian the models folder was Dokumente/Strata/Models and the workspace Documents/Strata/Workspace. The models folder follows
// the system's Documents folder; the default workspace now does on Linux too, and nowhere else changes.
import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultModelDirectory, documentsFolder, workspaceBases } from '../server/model-directory-default.mjs';

const german = { HOME: '/home/florian' };
const read = () => 'XDG_DOCUMENTS_DIR="$HOME/Dokumente"\nXDG_DOWNLOAD_DIR="$HOME/Downloads"\n';

test('on a German Linux desktop the workspace and the models folder share one Documents folder', () => {
  const models = defaultModelDirectory({ platform: 'linux', home: '/home/florian', env: german, read });
  const [first] = workspaceBases('/home/florian', { platform: 'linux', env: german, read, exists: () => false });
  assert.equal(models, '/home/florian/Dokumente/Strata/Models');
  assert.equal(first, '/home/florian/Dokumente');
});

test('the old places stay as fallbacks behind it', () => {
  assert.deepEqual(workspaceBases('/home/florian', { platform: 'linux', env: german, read, exists: () => false }), ['/home/florian/Dokumente', '/home/florian/Documents', '/home/florian']);
});

test('a workspace already made in <home>/Documents is kept while the system folder has none', () => {
  const had = p => p === '/home/florian/Documents/Strata/Workspace';
  assert.deepEqual(workspaceBases('/home/florian', { platform: 'linux', env: german, read, exists: had }), ['/home/florian/Documents', '/home/florian']);
  const both = p => p === '/home/florian/Documents/Strata/Workspace' || p === '/home/florian/Dokumente/Strata/Workspace';
  assert.equal(workspaceBases('/home/florian', { platform: 'linux', env: german, read, exists: both })[0], '/home/florian/Dokumente');
});

test('an English desktop, or one with no user-dirs file, is unchanged', () => {
  assert.deepEqual(workspaceBases('/home/ana', { platform: 'linux', env: { HOME: '/home/ana' }, read: () => { throw new Error('no file'); } }), ['/home/ana/Documents', '/home/ana']);
  assert.equal(documentsFolder({ platform: 'linux', home: '/home/ana', env: {}, read: () => 'XDG_DOCUMENTS_DIR="$HOME/Documents"\n' }), '/home/ana/Documents');
});

test('Windows and macOS keep <home>/Documents (a Documents folder moved to OneDrive does not move the default workspace)', () => {
  assert.deepEqual(workspaceBases('C:\\Users\\ana', { platform: 'win32', env: {} }), ['C:\\Users\\ana\\Documents', 'C:\\Users\\ana']);
  assert.deepEqual(workspaceBases('/Users/ana', { platform: 'darwin', env: {} }), ['/Users/ana/Documents', '/Users/ana']);
});

test('the models folder is what it was', () => {
  assert.equal(defaultModelDirectory({ platform: 'linux', home: '/home/ana', env: { STRATA_MODELS_DIR: '/data/m' }, read }), '/data/m');
  assert.equal(defaultModelDirectory({ platform: 'win32', home: 'C:\\Users\\ana', env: {}, windowsDocuments: () => 'D:\\Docs' }), 'D:\\Docs\\Strata\\Models');
});
