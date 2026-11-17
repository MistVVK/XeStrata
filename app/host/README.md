# Strata app: developer guide

This folder is the app's Node host (`server/`) and its web interface (`src/`, React and Vite). The desktop shell that wraps
them is [`../desktop`](../desktop) (Tauri). The engine and Strata Control are Python and C++ and live elsewhere in the
repository; see [docs/SOURCE_INSTALL.md](../../docs/SOURCE_INSTALL.md) and [packaging/README.md](../../tools/packaging/README.md).

## Start from a fresh clone

You need **Node 24.12 or newer** (`engines` in `package.json`) and npm. Nothing else: no model, no engine, no GPU, no
sibling design folder.

```sh
cd apps/host
npm ci
npm run build          # makes dist/, which the preview and `npm run serve` need (dist/ is not checked in)
npm run preview:ui     # simulated hardware and models; open http://localhost:5272/#/models
```

`preview:ui` starts an isolated host and a fixture server on temporary data and prints "UI PREVIEW ONLY". Nothing it shows
is real: no engine is started, no model is loaded, no files outside its temporary folder are touched. Ctrl+C stops it.

| Command | What it does |
| --- | --- |
| `npm run build` | Builds the interface into `dist/`. |
| `npm run typecheck` | `tsc --noEmit`. |
| `npm test` | The host and interface tests (`tests/*.test.mjs`, `node --test`). They write to scratch copies of the per-user folders, never to your own profile. A test that hangs fails after 10 minutes (`STRATA_TEST_TIMEOUT_MS`). |
| `npm run dev` | The source host (port 8791) and the Vite dev server (port 5199) together; Vite proxies `/api` to the host. |
| `npm run serve` | The host with the built `dist/` (run `npm run build` first). |
| `npm run check` | Typecheck plus the interface audits (`audit`, `contrast`, `measure`, `luminance`). |

## Preview, source host and packaged Control

- **Simulated preview** (`npm run preview:ui`): fixtures only. Use it to look at screens.
- **Source host** (`npm run dev`, `npm run serve`): the real host code from this checkout. Without `STRATA_REAL_ENGINE=1` it
  runs a clearly labelled example engine; with it, the host looks for a Strata installation as a source checkout does.
  Loading a real model needs an engine and model files; none of the commands above do that.
- **Packaged app**: the installer ships the host next to Strata Control (Python plus `strata_control`), which does installs,
  updates and engine starts. A source checkout has no Control, so those paths keep their older behaviour; how Control is
  packaged is in [packaging/README.md](../../tools/packaging/README.md).

The version numbers in `package.json` are the app's. They are checked against `docs/data/release.json` by `python tools/release_manifest.py --check`.

## Styles, icons and fonts

The checked-in files are the source. There is no generator and no sibling design folder.

- Kit styles: `src/styles/kit/tokens.css` and `components.css`; app styles sit on top in `src/styles/*.css`.
- Icons: `public/icons/sprite.svg`. Add an icon as a `<symbol id="i-NAME">`; `npm run audit` fails on an icon that is used but missing.
- Fonts: `public/fonts`.
- The standalone server page (`serve/web/`) serves its own copy of `components.css`. The two copies are byte-identical and
  `tests/kit-copy.test.mjs` fails when they differ: change both together.

## Languages

The app speaks English, Chinese, Japanese, German, French, Spanish, Russian, Portuguese and Korean. The English text is the key: `src/locales/en.json`
(with `fix-en.json`, `tune-en.json` and `tools-en.json`) holds every string, and each language has the same four files (`fr.json`, `fix-fr.json`, ...).

- **A new string.** Write it with `tr("...")`, then add it to the English catalog (`node scripts/append-localization.mjs`, or the next `fx` number in
  `scripts/make-fix-locales.py`). The language tests fail until every language has it.
- **French.** `python scripts/fr_sync.py missing todo.tsv` lists the English strings that have no French yet, one per line (`id`, a tab, the text).
  Translate the second column ("vous", short plain words, one next step; a lone `=` keeps the English), then `python scripts/fr_sync.py apply todo.tsv` merges it
  (it also sets the French spacing before `: ; ? !` and the typographic apostrophe) and `python scripts/fr_sync.py check` confirms nothing is missing.
- **Screenshots in a language.** `npm run build`, then `node scripts/language-shots.mjs fr <folder> models settings setup` (hermetic preview host, scratch profile, headless Chrome).
