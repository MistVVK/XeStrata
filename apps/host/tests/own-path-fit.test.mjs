/**
 * Qwen3.6-35B-A3B's context against the card (GitHub #5715). The app's estimate and its warning under the context chooser use the catalog's figures from
 * strata_control/own_path_fit.py, the rule setup's context menu uses; the Flash-Next int8 formula said 1.9 GB at 128K where the engine allocates 5.8, so a
 * 12 GB card got no warning and the engine stopped with a bare out-of-memory line. This test runs the Python rule and the app's on the same cards and needs
 * them to agree.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { transform } from 'esbuild';

const root = resolve(import.meta.dirname, '..');
const repo = resolve(root, '../..');
const load = async (f, deps = {}) => {
  let src = readFileSync(resolve(root, f), 'utf8');
  for (const [from, to] of Object.entries(deps)) src = src.replace(from, to);
  const { code } = await transform(src, { loader: 'ts', format: 'esm' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
};
const fit = await load('src/lib/native-fit.ts');
const fitUrl = `data:text/javascript;base64,${Buffer.from((await transform(readFileSync(resolve(root, 'src/lib/native-fit.ts'), 'utf8'), { loader: 'ts', format: 'esm' })).code).toString('base64')}`;
const ctx = await load('src/lib/default-context.ts', { "'./native-fit'": `'${fitUrl}'` });

const SIZES = ['UD-Q4_K_XL', 'UD-Q6_K_XL', 'Q8_0'];
const CARDS = [8, 10, 11.94, 12, 15.93, 16, 23.99, 32];
const CONTEXTS = [8192, 32768, 49152, 65536, 100000, 131072, 204800, 262144];

function python() {
  for (const exe of process.platform === 'win32' ? ['python', 'py'] : ['python3', 'python']) {
    const r = spawnSync(exe, ['--version'], { encoding: 'utf8' });
    if (r.status === 0) return exe;
  }
  return null;
}

test('the design catalog has Qwen3.6\'s conversation memory per token', () => {
  const src = readFileSync(resolve(root, 'server/catalog.mjs'), 'utf8');
  const kb = Number(/id:'qwen36'[^\n]*kvKbPerToken:(\d+)/.exec(src)?.[1]);
  assert.equal(kb, Math.round(47104 / 1024));
});

test('the app\'s Qwen3.6 context rule is own_path_fit.py\'s on 8 to 32 GB cards', { skip: !python() && 'no python here' }, () => {
  const script = 'import json,sys; sys.path.insert(0, sys.argv[1]); from strata_control import own_path_fit as F; '
    + 'sizes=json.loads(sys.argv[2]); cards=json.loads(sys.argv[3]); ctxs=json.loads(sys.argv[4]); '
    + 'out={m: {"in": F.inputs("qwen3.6-35b-a3b", m), "fits": [[F.fits(F.inputs("qwen3.6-35b-a3b", m), v, c) for c in ctxs] for v in cards], '
    + '"cache": [round(F.cache_gib(c), 6) for c in ctxs]} for m in sizes}; print(json.dumps(out))';
  const r = spawnSync(python(), ['-I', '-B', '-c', script, repo, JSON.stringify(SIZES), JSON.stringify(CARDS), JSON.stringify(CONTEXTS)], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  const py = JSON.parse(r.stdout);
  for (const m of SIZES) {
    const f = py[m].in;
    assert.ok(f && f.bytes_per_token === 47104, m);
    CARDS.forEach((v, i) => CONTEXTS.forEach((c, j) => {
      assert.equal(fit.ownPathFits(f, v, c), py[m].fits[i][j], `${m} ${v} GB ${c}`);
      assert.equal(ctx.contextIsTight(v, c, 'int8', f), !py[m].fits[i][j], `${m} ${v} GB ${c}: the warning`);
    }));
    CONTEXTS.forEach((c, j) => assert.ok(Math.abs(fit.contextCacheGb(c, 'int8', f) - py[m].cache[j]) < 1e-5, `${c}`));
  }
});

test('#5715: an RTX 5070 at 128K gets the warning with the real figure; 32K none; other models keep the Flash-Next rule', () => {
  const q4 = { bytes_per_token: 47104, fixed_gb: 8.55 + 1256 / 1024 + 0.07e9 / 2 ** 30 };
  assert.equal(ctx.contextIsTight(11.94, 131072, 'int8', q4), true);
  assert.equal(fit.contextCacheGb(131072, 'int8', q4).toFixed(1), '5.8');
  assert.equal(ctx.contextIsTight(11.94, 32768, 'int8', q4), false);
  assert.equal(ctx.contextIsTight(0, 131072, 'int8', q4), false);                       // no card known: nothing said
  // before: the Flash-Next int8 figure (1.9 GB at 128K) left "room" on the 12 GB card, so nothing was said
  assert.equal(fit.cacheGb(131072).toFixed(1), '1.9');
  assert.equal(ctx.contextIsTight(11.94, 131072, 'int8'), false);
  assert.equal(ctx.contextIsTight(11.94, 131072, 'int8', null), false);
  // the This computer card counts the real cache for Qwen3.6
  const v = { experts_gb: 2, context_fit: q4 };
  const a = fit.fitOn(v, { ram_gb: 64, vram_gb: 32 }, false, { context: 131072, kv: 'int8' });
  const b = fit.fitOn({ experts_gb: 2 }, { ram_gb: 64, vram_gb: 32 }, false, { context: 131072, kv: 'int8' });
  assert.ok(a.vramUseGb > b.vramUseGb, JSON.stringify([a, b]));
});

test('the Models page and the first-run page pass the catalog\'s figures to the warning', () => {
  const read = f => readFileSync(resolve(root, 'src', f), 'utf8');
  assert.match(read('components/NativeModels.tsx'), /contextIsTight\(fitHardware\.vram_gb, context, kv, selectedSize\?\.context_fit\)/);
  assert.match(read('components/NativeModels.tsx'), /contextCacheGb\(context, kv, selectedSize\?\.context_fit\)/);
  assert.match(read('components/GuidedSetup.tsx'), /contextIsTight\(topCardGb, context, 'int8', version\?\.context_fit\)/);
  assert.match(read('components/GuidedSetup.tsx'), /contextCacheGb\(context, 'int8', version\?\.context_fit\)/);
});
