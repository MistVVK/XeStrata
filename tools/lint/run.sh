#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# tools/lint/run.sh [FILE...] - the lints of AGENTS.md on the given files, by kind; without files, on the files that
# differ from HEAD (changed or new, not deleted).  Run from the repository root.  Exits 1 when any lint reports.
# The tools come from apt, except ruff, lychee and stylelint, which are in .lint/ (see AGENTS.md, Lints).
# XESTRATA_LINT_BUILD names the build folder whose compile_commands.json clang-tidy reads (default build/free;
# configure it with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON).
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
C=tools/lint
BUILD=${XESTRATA_LINT_BUILD:-${STRATA_LINT_BUILD:-build/free}}
CLANG_TIDY=$(find /opt/intel/oneapi/compiler -path '*/bin/compiler/clang-tidy' 2>/dev/null | sort -V | tail -1)

if [ $# -gt 0 ]; then
  files=("$@")
else
  mapfile -t files < <( { git diff --name-only --diff-filter=d HEAD; git ls-files --others --exclude-standard; } | sort -u)
fi
[ ${#files[@]} -gt 0 ] || { echo "nothing to lint"; exit 0; }
all=("${files[@]}")
# Other projects' files, community benchmark records and license texts are kept as received: gitleaks and reuse only.
# XeStrata's patches in third_party are linted.
mapfile -t files < <(printf '%s\n' "${files[@]}" | while read -r f; do
  case "$f" in ""|third_party/*/*/*|bench/XeStrataBenchResults/*|LICENSES/*)
    case "$f" in third_party/*/*/patches/*) ;; *) continue ;; esac ;;
  esac
  echo "$f"
done)

status=0
run() {   # run NAME TOOL ARGS...: run a lint, or say it is not installed
  local name=$1 tool=$2; shift 2
  if ! command -v "$tool" >/dev/null 2>&1 && [ ! -x "$tool" ]; then
    echo "== $name: skipped ($tool not installed)"
    return
  fi
  echo "== $name"
  "$tool" "$@" || status=1
}
pick() {  # pick PATTERN: the files whose path matches the extended regex
  printf '%s\n' "${files[@]}" | grep -E "$1" || true
}
code() {  # code PATTERN: as pick, without the lint configurations here (each tool reads its own dialect)
  pick "$1" | grep -v "^$C/" || true
}

mapfile -t patches < <(pick '^third_party/[^/]+/[^/]+/patches/.*\.patch$')
mapfile -t py < <(code '\.py$')
mapfile -t sh < <(pick '\.sh$')
mapfile -t cpp < <(pick '\.(cpp|hpp)$')
mapfile -t cmake < <(pick '(^|/)CMakeLists\.txt$|\.cmake$')
mapfile -t md < <(pick '\.md$')
mapfile -t css < <(pick '\.css$')
mapfile -t js < <(code '\.js$')
mapfile -t html < <(pick '\.html$')
mapfile -t text < <(pick '\.(py|sh|cpp|hpp|cu|h|inc|cmake|md|css|js|html|jinja|txt|toml|ini|yaml|rb|mjs)$|CMakeLists\.txt$')

if [ ${#patches[@]} -gt 0 ]; then
  run "patch changes" python3 "$C/patches.py" "${patches[@]}"
fi
if [ ${#py[@]} -gt 0 ]; then
  run flake8 flake8 --config "$C/flake8.ini" "${py[@]}"
  run ruff .lint/venv/bin/ruff check --quiet --config "$C/ruff.toml" "${py[@]}"
  run mypy mypy --config-file "$C/mypy.ini" --no-error-summary "${py[@]}"
fi
[ ${#sh[@]} -gt 0 ] && run shellcheck shellcheck "${sh[@]}"
if [ ${#cpp[@]} -gt 0 ]; then
  if [ -n "$CLANG_TIDY" ] && [ -f "$BUILD/compile_commands.json" ]; then
    tidy=("$CLANG_TIDY" --quiet -p "$BUILD" --config-file="$C/clang-tidy.yaml")
    mapfile -t tus < <(python3 -c 'import json, sys; print("\n".join(sorted({e["file"] for e in json.load(open(sys.argv[1]))})))' \
      "$BUILD/compile_commands.json")
    # a source this build does not compile (another GPU maker's, as the AMD-only HIP files on a build without HIP):
    # clang-tidy would guess its command from a neighbor and report the headers it cannot find, so it is left out
    cxx=()
    for f in "${cpp[@]}"; do
      [[ $f == *.cpp ]] || continue
      if grep -qxF "$PWD/$f" <<<"$(printf '%s\n' "${tus[@]}")"; then   # not a pipe: grep -q ends it early
        cxx+=("$f")
      else
        echo "== clang-tidy: $f skipped (not compiled in $BUILD)"
      fi
    done
    mapfile -t hpp < <(printf '%s\n' "${cpp[@]}" | grep '\.hpp$')
    [ ${#cxx[@]} -gt 0 ] && run clang-tidy "${tidy[@]}" "${cxx[@]}"
    # A header is checked inside a source that includes it, with that source's command: alone it gets the command of
    # whichever source's path is nearest, often one without -fsycl, and misses the macros a source defines before the
    # include (strata/kernels/device_arch_tag.hpp).  --line-filter keeps the header's findings only.
    if [ ${#hpp[@]} -gt 0 ]; then
      for h in "${hpp[@]}"; do
        case "$h" in include/*) inc="[<\"]${h#include/}[>\"]" ;; *) inc="\"([^\"]*/)?${h##*/}\"" ;; esac
        tu=$(grep -l -E "^#include $inc" "${tus[@]}" 2>/dev/null | head -n 1)
        if [ -z "$tu" ]; then   # a header only another header includes (a shape's in layout.hpp): a source of that one
          while read -r p; do
            tu=$(grep -l -E "^#include [<\"]${p#include/}[>\"]" "${tus[@]}" 2>/dev/null | head -n 1)
            [ -n "$tu" ] && break
          done < <(grep -rl -E "^#include $inc" include 2>/dev/null)
        fi
        if [ -n "$tu" ]; then
          run "clang-tidy $h (in ${tu#"$PWD"/})" "${tidy[@]}" --header-filter='.*' --line-filter="[{\"name\":\"$h\"}]" "$tu"
        else
          run "clang-tidy $h" "${tidy[@]}" "$h"
        fi
      done
    fi
  else
    echo "== clang-tidy: skipped (needs oneAPI's clang-tidy and $BUILD/compile_commands.json)"
  fi
  run cppcheck cppcheck --quiet --error-exitcode=1 --std=c++20 --enable=warning,performance --library="$C/cppcheck.cfg" \
    --suppress=missingIncludeSystem --suppress=noCopyConstructor --suppress=noOperatorEq "${cpp[@]}"
fi
[ ${#cmake[@]} -gt 0 ] && run cmake-lint cmake-lint -c "$C/cmake-format.py" -- "${cmake[@]}"
if [ ${#md[@]} -gt 0 ]; then
  run markdownlint mdl --style "$C/mdl_style.rb" "${md[@]}"
  run lychee .lint/bin/lychee --offline --no-progress "${md[@]}"
fi
[ ${#css[@]} -gt 0 ] && run stylelint .lint/node_modules/.bin/stylelint --config "$C/stylelint.config.mjs" \
  --config-basedir .lint "${css[@]}"
[ ${#js[@]} -gt 0 ] && run eslint eslint --no-eslintrc -c "$C/eslintrc.js" "${js[@]}"
for f in "${html[@]}"; do run "tidy $f" tidy -config "$C/tidy.conf" -errors -quiet "$f"; done
[ ${#text[@]} -gt 0 ] && run codespell codespell --config "$C/codespell.cfg" "${text[@]}"
for f in "${all[@]}"; do
  [ -f "$f" ] || continue
  gitleaks detect --no-git --redact --no-banner --exit-code 1 --source "$f" >/dev/null 2>&1 \
    || { echo "== gitleaks: a possible secret in $f (gitleaks detect --no-git --redact --source $f)"; status=1; }
done
# upstream's name where a user reads XeStrata's (messages, environment variables, programs, documents)
if [ ${#files[@]} -gt 0 ]; then
  echo "== upstream's name"
  python3 "$C/upstream_name.py" "${files[@]}" || status=1
  # Each file's SPDX header against where its code comes from (AGENTS.md, Licenses).
  echo "== license classes"
  python3 tools/license_class.py "${files[@]}" || status=1
fi
REUSE=$(command -v reuse || echo .lint/venv/bin/reuse)
run reuse "$REUSE" lint --quiet
exit $status
