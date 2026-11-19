/**
 * #6245: a workspace skill's agent folder holds the whole skill, not only a regenerated SKILL.md. A skill whose instructions say "run scripts/build.py"
 * or "use the template in assets/" points at files that have to be there. No Harness is needed: the copy is plain files.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { copySkillExtras, materializeSkills } from '../server/skill-files.mjs';

const scratch = () => mkdtempSync(join(tmpdir(), 'strata-skill-extras-'));
const cleanup = dir => { try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } };
const put = (file, text) => { mkdirSync(join(file, '..'), { recursive: true }); writeFileSync(file, text); };

test('the files beside a workspace skill reach the agent folder in the same places', () => {
  const dir = scratch();
  try {
    const skill = join(dir, 'workspace', 'skills', 'demo');
    put(join(skill, 'SKILL.md'), '---\nname: demo\ndescription: d\n---\nRead assets/hello.txt and print it.');
    put(join(skill, 'assets', 'hello.txt'), 'hello');
    put(join(skill, 'scripts', 'build.py'), 'print(1)');
    put(join(skill, 'references', 'deep', 'notes.md'), 'notes');
    put(join(skill, '.git', 'config'), 'not copied');
    const live = join(dir, 'live');
    const list = [{ id: 'ws-demo', name: 'Demo', description: 'd', body: 'Read assets/hello.txt and print it.', enabled: true }];
    materializeSkills(live, list, new Map([['ws-demo', skill]]));
    assert.equal(readFileSync(join(live, 'demo', 'assets', 'hello.txt'), 'utf8'), 'hello');
    assert.equal(readFileSync(join(live, 'demo', 'scripts', 'build.py'), 'utf8'), 'print(1)');
    assert.equal(readFileSync(join(live, 'demo', 'references', 'deep', 'notes.md'), 'utf8'), 'notes');
    assert.equal(existsSync(join(live, 'demo', '.git')), false);
    assert.match(readFileSync(join(live, 'demo', 'SKILL.md'), 'utf8'), /^---\nname: demo\n/, 'the SKILL.md stays the regenerated one with the frontmatter');
    // without the folders argument (the old call) nothing is copied
    const bare = join(dir, 'bare');
    materializeSkills(bare, list);
    assert.deepEqual(readdirSync(join(bare, 'demo')), ['SKILL.md']);
  } finally { cleanup(dir); }
});

test('a changed file is copied again, an unchanged one is not touched, a file taken out of the skill is taken out here', () => {
  const dir = scratch();
  try {
    const skill = join(dir, 'skill'), live = join(dir, 'live', 'demo');
    mkdirSync(live, { recursive: true });
    put(join(skill, 'a.txt'), 'one');
    put(join(skill, 'b.txt'), 'two');
    assert.deepEqual(copySkillExtras(skill, live).sort(), ['a.txt', 'b.txt']);
    const before = statSync(join(live, 'a.txt')).mtimeMs;
    copySkillExtras(skill, live);
    assert.equal(statSync(join(live, 'a.txt')).mtimeMs, before, 'an unchanged file is not rewritten (the agent treats every write as a change)');
    writeFileSync(join(skill, 'a.txt'), 'three!');
    copySkillExtras(skill, live);
    assert.equal(readFileSync(join(live, 'a.txt'), 'utf8'), 'three!');
    rmSync(join(skill, 'b.txt'));
    copySkillExtras(skill, live);
    assert.equal(existsSync(join(live, 'b.txt')), false);
    // a file of the agent folder that was never copied from the skill (the person's own copy by hand) is left alone
    put(join(live, 'mine.txt'), 'by hand');
    copySkillExtras(skill, live);
    assert.equal(readFileSync(join(live, 'mine.txt'), 'utf8'), 'by hand');
  } finally { cleanup(dir); }
});

test('the limits hold: a big file, too many files, links and dot folders are not copied', () => {
  const dir = scratch();
  try {
    const skill = join(dir, 'skill'), live = join(dir, 'live');
    put(join(skill, 'small.txt'), 'x');
    put(join(skill, 'big.bin'), 'y'.repeat(2000));
    for (let i = 0; i < 5; i++) put(join(skill, 'many', `f${i}.txt`), String(i));
    put(join(skill, 'node_modules', 'pkg', 'index.js'), 'z');
    try { symlinkSync(join(skill, 'small.txt'), join(skill, 'link.txt')); } catch { /* no symlinks here */ }
    const found = copySkillExtras(skill, live, { files: 4, bytes: 1000, depth: 2, file: 1000 });
    assert.ok(!found.includes('big.bin'));
    assert.ok(found.length <= 4);
    assert.ok(!found.some(f => f.startsWith('node_modules')));
    assert.ok(!found.includes('link.txt'));
    assert.equal(existsSync(join(live, 'big.bin')), false);
  } finally { cleanup(dir); }
});
