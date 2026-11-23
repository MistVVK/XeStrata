#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/lint/upstream_name.py [FILE...] - finds upstream's name where a user reads XeStrata's.

XeStrata's messages, logs, environment variables, programs, API fields and documents say xestrata (XESTRATA_*);
strata alone is upstream's name, and an upstream sync brings it back.  This lint reads the string literals of the
C++, Python and F# files, the shell scripts' code (not their comments), the documents (Markdown) and the web app
(JavaScript, HTML, the locales), and reports a `strata` that is not part of `xestrata`, outside the uses ALLOWED
keeps: the namespace and include paths, the file names, the CMake options and targets, the API fields and markers
shared with upstream, an old STRATA_<name> a script still reads after XESTRATA_<name>, and the credits.
In the documents, `Strata` is upstream's name and stays.  A line of code whose comment says upstream (or an earlier
name) speaks of upstream's engine and is not read; a document marks what it says of the names themselves with
<!-- upstream-name: off --> and <!-- upstream-name: on -->.  Without files it checks every tracked file it reads.
Exits 1 on a finding.
"""

import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

# A strata that is not the tail of xestrata (any case) and not inside a longer word.
WORD = re.compile(r'(?<![A-Za-z0-9])(?<![Xx][Ee])(?:strata|Strata|STRATA)(?![A-Za-z])')

# Uses that stay (AGENTS.md, the names that do not change): matched against the text around each find.
ALLOWED = [
    r'strata::', r'strata/', r'include/strata', r'\bnamespace strata\b',
    r'tools/strata_\w+\.py', r'strata_\w+\.py\b', r'strata_\w+\.(?:hpp|cpp)\b',
    # the API and markers shared with upstream
    r'\bstrata_(?:mcp|prefix|checkpoint|tune|rejected_tool_calls)\b', r'"service": "strata"',
    r'strata:(?:[a-z_{]|$)', r'(?:HELP|TYPE) strata:', r'\bstrata\.(?:ple|[a-z_]+)\b', r'strata\.r1:',
    r'\bstrata_(?:version|commit)\b',
    # the .strata model format (upstream's, Inkling-2-Mini's files): its extension, its magic and the format's name
    r'\.strata\b', r'\bstrata-engine\b', r'Inkling-2-Mini-Strata\b', r'STRATA(?:\\+x?0+)?\\+x01', r'format=strata\b',
    # the marks inside files and the state setup keeps beside them, which older files carry
    r'\bstrata-(?:ple|embd|pack)\b', r'strata-pack v', r'# strata pack index', r'\.strata-(?:pip\.json|mcp)\b',
    r'strata-shared-expert-arena', r'\bstrata_(?:tokenizer|onemath)\b',
    r'STRATA-ARENA-V1', r'# strata native experts', r'strata (?:native experts|hadamard)',
    # the MCP tools (tools/strata_mcp.py) and the fields shared with upstream
    r'\bstrata_(?:status|models|start|stop|logs|benchmark|connect_info|install|setup_for_mcp|folder|control|cli)\b',
    r'\bstrata_(?:caps_\w+|img_keys|piped_mtp)',
    # the fields between the server and its engine, which no user sends or reads
    r'\bstrata_(?:grammar\w*|strict_tools|route_prior|script)\b', r'strata-tool-media:', r'strata-prefix-end',
    # a log head that reads both XeStrata's engine and upstream's (the benchmarks)
    r'\(\?:xe\)\?strata',
    # the symbols a kernel module exports, read by name (dlsym)
    r'\bstrata_(?:kitchen|i8_gemm)\w*', r'(?<![\w.])strata_sdpa_aotriton(?![\w.])',
    # CMake's targets and variables, which keep upstream's names (the programs' names are xestrata-*)
    r'--target (?:strata[\w-]*(?:\s+|$))+', r'\bstrata_(?:hip_cxx|warnings)\b',
    r'targets="(?:strata[\w-]*\s+)+',
    # a shell script's old name, read after XeStrata's: ${XESTRATA_X:-${STRATA_X:-}}
    r'\$\{STRATA_[A-Z0-9_]+:-',
    # a tool that starts an engine clears the upstream-named measurement variables it may inherit
    r'"STRATA_"',
    # upstream's own things: its SYCL port, a variable of its that XeStrata did not take
    r'strata-sycl', r'build-sycl-aot/strata\b', r"f7eaf2de(?:'s| の) STRATA_EXPDEC",
    # the old names, where a document says they are read too
    r'`STRATA_\*`',
    # a heading's anchor (the headings of upstream's versions)
    r'#[\w%-]*strata-\d',
    # credits and upstream named
    r'(?i:sync-strata)',
    r'[Uu]pstream(?:\'s)? Strata', r'upstream の Strata', r'Upstream-Strata', r'Strata en amont', r'Strata \(upstream\)',
    r'Strata(?:-GLM)? (?:[0-9]|#|v[0-9])', r'Strata\'s', r'\boriginal-strata\b',
    r'Niko1221', r'the Strata contributors', r'Strata-GLM', r'github\.com/[\w.-]+/Strata',
    r'XeStrata', r'upstream\.name',
]
ALLOWED_RE = re.compile('|'.join(ALLOWED))

CPP = {'.cpp', '.hpp', '.h', '.cc', '.inc', '.cu'}
FS = {'.fs', '.fsx'}
SHELL = {'.sh'}
DOC = {'.md'}
WEB = {'.js', '.html', '.json'}


def cmake_options(root: Path) -> set[str]:
    """The STRATA_* names CMake defines (option, cache variables): they keep their names."""
    names: set[str] = set()
    for f in [root / 'CMakeLists.txt', *sorted((root / 'cmake').glob('*.cmake'))]:
        if f.exists():
            names.update(re.findall(r'\b(STRATA_[A-Z0-9_]+)\b', f.read_text(errors='replace')))
    return names


def cpp_strings(text: str):
    """(line, literal) for each string literal of a C++ file, comments, includes and raw strings aside."""
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == '\n':
            line += 1
            i += 1
        elif text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            line += text.count('\n', i, j)
            i = j
        elif c == '#' and re.match(r'#\s*include\b', text[i:i + 16]):
            j = text.find('\n', i)
            i = n if j < 0 else j
        elif c == "'" and not (i > 0 and (text[i - 1].isalnum())):
            j = i + 1
            while j < n and text[j] != "'" and text[j] != '\n':
                j += 2 if text[j] == '\\' else 1
            i = j + 1
        elif c == 'R' and text.startswith('R"', i) and not (i > 0 and (text[i - 1].isalnum() or text[i - 1] == '_')):
            k = text.find('(', i)
            delim = ')' + text[i + 2:k] + '"'
            j = text.find(delim, k)
            j = n if j < 0 else j + len(delim)
            yield line, text[k + 1:j]
            line += text.count('\n', i, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"' and text[j] != '\n':
                j += 2 if text[j] == '\\' else 1
            yield line, text[i + 1:j]
            i = j + 1
        else:
            i += 1


def py_strings(text: str):
    """(line, literal) for each string of a Python file (the text parts of f-strings included), docstrings aside."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
        starts = {tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT}
        for k, tok in enumerate(toks):
            if tok.type == tokenize.STRING and (k == 0 or toks[k - 1].type in starts) and \
                    k + 1 < len(toks) and toks[k + 1].type in (tokenize.NEWLINE, tokenize.ENDMARKER):
                continue                                     # a docstring: read by developers only
            if tok.type == tokenize.STRING or tok.type == getattr(tokenize, 'FSTRING_MIDDLE', -1):
                yield tok.start[0], tok.string
    except (tokenize.TokenError, SyntaxError):
        for i, ln in enumerate(text.splitlines(), 1):
            yield i, ln


def lines(text: str):
    """The lines of a document, without those between <!-- upstream-name: off --> and <!-- upstream-name: on -->
    (where a document speaks of upstream's names themselves)."""
    off = False
    for i, ln in enumerate(text.splitlines(), 1):
        if '<!-- upstream-name: off -->' in ln:
            off = True
        elif '<!-- upstream-name: on -->' in ln:
            off = False
        elif not off:
            yield i, ln


def js_lines(text: str):
    """The lines of a script without its comments (`// ...` after a blank, `/* ... */`)."""
    text = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group(0).count('\n'), text, flags=re.S)
    for i, ln in enumerate(text.splitlines(), 1):
        yield i, re.sub(r'(^|\s)//.*', '', ln)


def fs_strings(text: str):
    """(line, literal) for each string literal of an F# file: its (* ... *) comments aside, then read as C++ is (the
    // comments, "..." strings)."""
    text = re.sub(r'\(\*.*?\*\)', lambda m: '\n' * m.group(0).count('\n'), text, flags=re.S)
    return cpp_strings(text)


def sh_lines(text: str):
    """The lines of a shell script without its comments (a # at the start of a line or after a blank; $# and ${#x}
    are not comments)."""
    for i, ln in enumerate(text.splitlines(), 1):
        yield i, re.sub(r'(^|\s)#.*', '', ln)


def skipped(path: str) -> bool:
    """Files no user reads: other projects', the tests, the lint itself, the license texts."""
    p = Path(path)
    name = p.name
    return (path.startswith(('third_party/', 'LICENSES/', 'tests/', 'ref/', 'bench/', 'serve/fixtures/',
                             'launcher/Tests/'))
            or name.startswith('test_') or re.search(r'_test\.(cpp|py)$', name) is not None
            or path in {'tools/lint/upstream_name.py', 'tools/bench_upstream.py'}
            or name in {'NOTICE', 'COPYING', 'COPYING.LESSER'})


def check(path: str, options: set[str]) -> list[str]:
    p = Path(path)
    ext = p.suffix
    if skipped(path) or not p.is_file():
        return []
    try:
        text = p.read_text()
    except (UnicodeDecodeError, OSError):
        return []
    if ext in CPP:
        pieces = cpp_strings(text)
    elif ext == '.py':
        pieces = py_strings(text)
    elif ext == '.js':
        pieces = js_lines(text)
    elif ext in FS:
        pieces = fs_strings(text)
    elif ext in SHELL:
        pieces = sh_lines(text)
    elif ext in DOC or ext in WEB:
        pieces = lines(text)
    else:
        return []
    doc = ext in DOC
    src = text.splitlines()
    note = re.compile(r'(?://|#).*\b(?:upstream|earlier name)'   # a line on upstream's engine, or an old name
                      r'|["\']service["\']')          # /health's service, which upstream's clients check
    out = []
    for ln, s in pieces:
        if not doc and 0 < ln <= len(src) and note.search(src[ln - 1]):
            continue
        for m in WORD.finditer(s):
            if doc and s[m.start()] == 'S' and s[m.start():m.start() + 6] == 'Strata':
                continue                                    # upstream's name in prose
            tail = re.match(r'[A-Z0-9_]*', s[m.end():]).group(0)  # type: ignore[union-attr]
            if s[m.start():m.end()] == 'STRATA' and ('STRATA' + tail) in options:
                continue                                    # a CMake option keeps its name
            ctx = s[max(0, m.start() - 24):m.end() + 28]
            if any(a.start() <= m.start() - max(0, m.start() - 24) < a.end() for a in ALLOWED_RE.finditer(ctx)):
                continue
            out.append(f'{path}:{ln}: {ctx.strip()!r}')
    return out


def main(argv: list[str]) -> int:
    root = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True,
                               check=True).stdout.strip())
    files = argv or subprocess.run(['git', 'ls-files'], capture_output=True, text=True, cwd=root,
                                   check=True).stdout.split()
    options = cmake_options(root)
    found = [f for path in files for f in check(path, options)]
    for f in found:
        print(f)
    if found:
        print(f'{len(found)} uses of upstream\'s name strata where XeStrata\'s (xestrata, XESTRATA_*) belongs; '
              'a use that stays goes into ALLOWED in tools/lint/upstream_name.py')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
