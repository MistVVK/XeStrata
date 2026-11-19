/**
 * Skills on disk, for the app and for the agent.
 *
 * The app keeps the skills a person writes in its own folder (`<data>/skills/<id>/SKILL.md`). The agent (the Harness) finds skills by scanning
 * folders of `<name>/SKILL.md` files whose frontmatter has a kebab-case `name` and a `description`. So the app also writes the whole current
 * list (bundled, written, added, edited, switched on or off) into one folder the Harness is told to scan (`materializeSkills`), every time
 * anything changes. The Harness watches that folder, so a skill saved in the app is there for the next message without a restart.
 */
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

/** The note a skill's agent folder keeps of the files it copied from the skill's own folder, so a file taken out there is taken out here too (#6245). */
const COPIED = '.strata-copied.json';
export const SKILL_EXTRAS_LIMITS = { files: 500, bytes: 64 * 1024 * 1024, depth: 6, file: 16 * 1024 * 1024 };

/**
 * A skill's instructions refer to its own files ("run scripts/build.py", "use the template in assets/"). The agent reads `<agent skills>/<name>/`, so those
 * files go there beside the SKILL.md (#6245): every file under `from` except the SKILL.md at its top, in the same places. Not followed: links, folders whose
 * name starts with a dot, node_modules; not copied past the limits (500 files, 64 MB, 6 levels, 16 MB a file). A file that is already there with the same
 * size and a newer or equal time is left alone (the agent treats every write as a change); a file the person took out of the skill is taken out here.
 * @returns {string[]} the relative paths now copied
 */
export function copySkillExtras(from, to, limits = SKILL_EXTRAS_LIMITS) {
  const found = [];
  let bytes = 0;
  const walk = (dir, rel, depth) => {
    if (depth > limits.depth) return;
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (found.length >= limits.files) return;
      const name = entry.name, relative = rel ? `${rel}/${name}` : name;
      if (!rel && name.toLowerCase() === 'skill.md') continue;
      if (entry.isDirectory()) {
        if (name.startsWith('.') || name === 'node_modules') continue;
        walk(join(dir, name), relative, depth + 1);
      } else if (entry.isFile() && !name.startsWith('.')) {
        const size = statSync(join(dir, name)).size;
        if (size > limits.file || bytes + size > limits.bytes) continue;
        bytes += size;
        found.push(relative);
      }
    }
  };
  walk(from, '', 0);
  for (const relative of found) {
    const source = join(from, ...relative.split('/')), target = join(to, ...relative.split('/'));
    let current = false;
    try { const a = statSync(source), b = statSync(target); current = a.size === b.size && b.mtimeMs >= a.mtimeMs; } catch { /* not there yet */ }
    if (!current) { mkdirSync(dirname(target), { recursive: true }); copyFileSync(source, target); }
  }
  let before = [];
  try { before = JSON.parse(readFileSync(join(to, COPIED), 'utf8')); } catch { /* the first copy */ }
  const now = new Set(found);
  for (const relative of Array.isArray(before) ? before : []) {
    if (typeof relative === 'string' && !now.has(relative) && !relative.includes('..')) rmSync(join(to, ...relative.split('/')), { force: true });
  }
  const note = JSON.stringify(found);
  let same = false; try { same = readFileSync(join(to, COPIED), 'utf8') === note; } catch { /* not there yet */ }
  if (!same) writeFileSync(join(to, COPIED), note);
  return found;
}

/** The kebab-case name the Harness requires, from what a person typed ("Directory opener" -> "directory-opener"). */
export function skillSlug(name) {
  return String(name ?? '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60);
}

const yamlText = value => JSON.stringify(String(value ?? '').replace(/\s+/g, ' ').trim());

/** `SKILL.md` text: the frontmatter the Harness reads, then the instructions. */
export function skillFileText({ name, title = null, description, body, hidden = false }) {
  return `---\nname: ${skillSlug(name)}\n${title ? `title: ${yamlText(title)}\n` : ''}description: ${yamlText(description || name)}\n${hidden ? 'disable-model-invocation: true\n' : ''}---\n\n${String(body ?? '').trim()}\n`;
}

/** Name, description and instructions out of a SKILL.md (frontmatter optional). */
export function parseSkillFile(text) {
  const raw = String(text ?? '').replace(/^﻿/, '').replace(/\r\n/g, '\n');
  const m = /^---\n([\s\S]*?)\n---\n?/.exec(raw);
  if (!m) return { name: null, title: null, description: null, body: raw.trim() };
  const field = key => {
    const line = new RegExp(`^${key}:\\s*(.*)$`, 'mi').exec(m[1])?.[1]?.trim();
    if (line === undefined) return null;
    try { if (/^["']/.test(line)) return typeof JSON.parse(line.replace(/^'(.*)'$/, '"$1"')) === 'string' ? JSON.parse(line.replace(/^'(.*)'$/, '"$1"')) : line; } catch { /* plain scalar */ }
    return line;
  };
  return { name: field('name'), title: field('title'), description: field('description'), body: raw.slice(m[0].length).trim() };
}

/** Where a skill written in the app lives, and writing it there. */
export function writeCreatedSkill(root, { name, description, body }) {
  const id = skillSlug(name);
  if (!id) throw new Error('Give the skill a name with letters or numbers in it.');
  const dir = join(root, id);
  if (existsSync(dir)) throw new Error(`You already have a skill called "${id}". Choose another name, or edit the one you have.`);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'SKILL.md'), skillFileText({ name: id, title: String(name).trim(), description, body }));
  return id;
}

/**
 * Make `root` hold exactly the skills the agent may use: one folder each, a SKILL.md with frontmatter. A skill switched off is written
 * with `disable-model-invocation` (the agent does not offer it); folders of skills that are gone are removed. Only this folder is touched.
 * A skill that comes from a folder of its own (`folders`: skill id -> that folder) brings the rest of the folder along (`copySkillExtras`).
 * @param {string} root
 * @param {Array<{id: string, name: string, description?: string, body?: string, enabled?: boolean}>} skills
 * @param {Map<string, string>|Record<string, string>} [folders]
 */
export function materializeSkills(root, skills, folders = {}) {
  const folderOf = id => (folders instanceof Map ? folders.get(id) : folders?.[id]);
  mkdirSync(root, { recursive: true });
  const keep = new Set();
  for (const s of skills) {
    const slug = skillSlug(s.id.replace(/^(?:ws|my)-/, '') || s.name);
    if (!slug || !s.body?.trim()) continue;
    let unique = slug, n = 2;
    while (keep.has(unique)) unique = `${slug}-${n++}`;
    keep.add(unique);
    const dir = join(root, unique);
    mkdirSync(dir, { recursive: true });
    const text = skillFileText({ name: unique, description: `${s.name}. ${s.description ?? ''}`.trim(), body: s.body, hidden: s.enabled === false });
    const file = join(dir, 'SKILL.md');
    // Only a changed skill is rewritten: the agent watches this folder and treats every write as a change to its list.
    let same = false; try { same = readFileSync(file, 'utf8') === text; } catch { /* not there yet */ }
    if (!same) writeFileSync(file, text);
    const from = folderOf(s.id);
    if (from) { try { copySkillExtras(from, dir); } catch (e) { console.warn(`skills · could not copy the files of "${unique}":`, e.message); } }
  }
  for (const entry of readdirSync(root, { withFileTypes: true })) if (entry.isDirectory() && !keep.has(entry.name)) rmSync(join(root, entry.name), { recursive: true, force: true });
  return [...keep];
}
