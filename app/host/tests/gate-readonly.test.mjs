// #3975 (P0): Manual mode's "read-only" shell allowlist must be safe by construction. A command without a card may only use options that are known to
// look and nothing else. The first block is what the issue reported (write and program-execution modes of allowlisted commands), the rest is the
// same class: options, operands, spellings and quoting. The last block is the commands people really type, which must keep running unasked.
import test from 'node:test';
import assert from 'node:assert/strict';
import { isReadOnlyShell, maskQuotes, readOnlyCommandNames } from '../server/agent/plugins/strata-gate.mjs';

const denied = (list) => { for (const c of list) assert.equal(isReadOnlyShell(c), false, `must ask: ${c}`); };
const allowed = (list) => { for (const c of list) assert.equal(isReadOnlyShell(c), true, `must run unasked: ${c}`); };

test('#3975: the five confirmed cases and the rest of the issue table now ask', () => {
  denied([
    'sort --compress-program=sh a.txt', 'sort --compress-program sh a.txt', 'sort -o out.txt a.txt', 'sort --output=out.txt a.txt',
    'rg --pre sh pattern', 'rg --pre=sh pattern',
    'git grep -Osh pattern', 'git grep -O sh pattern', 'git grep --open-files-in-pager=sh pattern',
    'uniq in.txt out.txt',
    'tree -o out.txt', 'file -C -m x', 'date -s 12:00', 'date --set=12:00', 'tail -f app.log',
  ]);
});

test('#3975: other spellings of the same options (attached, clustered, abbreviated, repeated) ask', () => {
  denied([
    'sort -oout.txt a.txt', 'sort -ro out.txt a.txt', 'sort -nro out.txt a.txt', 'sort --out=out.txt a.txt', 'sort --outp out.txt a.txt',
    'sort --comp=sh a.txt', 'sort -T /tmp a.txt', 'sort --files0-from=list a.txt', 'sort -n -u -o out a',
    'rg --pre-glob "*" x', 'rg --hostname-bin=sh x', 'rg -z x', 'rg --search-zip x', 'rg -uz x', 'rg --pre sh --files',
    'git grep -O"sh" x', 'git grep -nOsh x', 'git grep -O x', 'git grep --open x', 'git grep -l --open-files-in-pager=sh x',
    'uniq - out.txt', 'uniq -c in.txt out.txt', 'uniq -i -u in out',
    'tree -oout.txt', 'tree -a -o out.txt', 'tree -R', 'tree --output=out',
    'file -C', 'file --compile x', 'file -m magic x', 'file -f list', 'file -bC x', 'file --magic-file=x y',
    'date -sMonth', 'date -u -s now', 'date --set now', 'date 102612002026', 'date -u 1026', 'date -d now 1026',
    'tail -F app.log', 'tail --follow app.log', 'tail --follow=name app.log', 'tail -fn5 app.log', 'tail -n 1 -f app.log', 'tail --retry x', 'tail -f -n 1 x',
    'tail --pid=1 -f x', 'tail -s 1 x',
    'head -f x', 'wc --files0-from=list', 'du --files0-from=list', 'du -X file', 'stat --foo x', 'ls --foo', 'ls -@', 'cat -@ x', 'grep --foo x y',
    'hostname newname', 'hostname -F file', 'uname -n box', 'uname box',
  ]);
});

test('#3975: git subcommands that write or run something when given the right option ask', () => {
  denied([
    'git diff --output=out.txt', 'git diff --out=out.txt', 'git log --output=out.txt', 'git show --output out HEAD', 'git log --ext-diff', 'git diff --ext-diff',
    'git diff --textconv', 'git log --textconv', 'git grep --textconv x', 'git cat-file --textconv HEAD:a', 'git cat-file --filters HEAD:a', 'git log --exec=sh',
    'git config -e', 'git config --edit', 'git config --global --edit', 'git config --global user.name x', 'git config user.name x', 'git config --unset user.name',
    'git config --add a.b c', 'git config --replace-all a.b c', 'git config --file x --list', 'git config --get', 'git config --get a --edit',
    'git branch --set-upstream-to=origin/main', 'git branch --unset-upstream', 'git branch -u origin/main', 'git branch -d x', 'git branch -D x', 'git branch -m new',
    'git branch --edit-description', 'git branch newbranch', 'git branch -a newbranch', 'git branch --track x origin/x',
    'git tag v1', 'git tag -a v1', 'git tag -d v1', 'git tag -s v1', 'git tag -v v1', 'git tag -a -m msg v1', 'git tag --edit',
    'git remote add o u', 'git remote show origin', 'git remote update', 'git remote prune origin', 'git remote set-url o u', 'git remote rename a b',
    'git -c core.pager=sh log', 'git --paginate log', 'git -C x status', 'git --git-dir=x log', 'git -p log',
    'git commit', 'git push', 'git checkout x', 'git stash', 'git reset', 'git clean -n', 'git fetch', 'git pull', 'git rev-parse --parseopt', 'git ls-files -x',
    'git status --foo', 'git log --foo', 'git blame --contents=x a', 'git diff -O orderfile', 'git diff --no-pager',
  ]);
});

test('#3975: PowerShell commands: parameters that wait forever, reach other machines or open windows ask', () => {
  denied([
    'Get-Content a.txt -Wait', 'gc a.txt -Wait', 'type a.txt -Wait', 'Get-Process -ComputerName other', 'Get-Help Get-ChildItem -Online', 'Get-Help Get-ChildItem -ShowWindow',
    'Select-String -Pattern x -Foo', 'Get-ChildItem -Rec', 'Get-ChildItem -Debug', 'Get-ChildItem -OutVariable x', 'Get-ChildItem -ErrorVariable x',
    'Sort-Object -Foo', 'Get-Date -Foo', 'Get-FileHash a.txt -InputStream x',
  ]);
});

test('#3975: pagers and anything not on the list ask', () => {
  denied(['less a.txt', 'more a.txt', 'less -o log a.txt', 'find . -name x', 'xargs ls', 'awk 1 a', 'sed s/a/b/ a', 'env', 'tee a', 'printenv', 'mv a b', 'cp a b', 'touch a', 'mkdir a', 'dd if=a of=b']);
});

test('#3975: quoting cannot hide an option or an operand', () => {
  denied([
    'sort "--output=out" a', 'sort --out"put"=out a', 'sort --"output"=out a', "sort '-o' out a", 'sort -"o" out a', 'sort -o"out" a', 'sort --compress-program="sh" a',
    'rg --pre "sh" x', "rg --pre 'sh' x", 'git grep -O"sh" x', 'git grep "-Osh" x',
    'uniq a "b"', 'uniq "a" "b"', "uniq 'a' 'b'", 'date "1026120026"', "date '1026'", 'date "-s" now', 'date "-s now"',
    'tail "-f" x', "tail '-f' x", 'tail -n 1 "-f" x',
    'tree "-o" x', 'file "-C" x', 'git diff "--output=out.txt"', 'git "diff" --output=x',
    // a quoted dash value is never taken for a value of a flag that wants a number
    'head -n "-f" x',
  ]);
});

test('#3975: operands that would write or set something ask (uniq output file, date setting, hostname)', () => {
  denied(['uniq a b', 'uniq a b c', 'date 1231', 'date 12311200', 'hostname a', 'whoami /foo', 'tr a b c', 'git tag v1.0', 'git branch topic']);
});

test('#3975: every command on the list asks for an unknown option, long or short', () => {
  const names = readOnlyCommandNames();
  assert.ok(names.length > 40, 'the list is not empty');
  for (const name of names) {
    assert.equal(isReadOnlyShell(`${name} -@`), false, `${name} -@`);
    assert.equal(isReadOnlyShell(`${name} --zz-no-such-option`), false, `${name} --zz-no-such-option`);
    assert.equal(isReadOnlyShell(`${name} --zz-no-such-option=1 x`), false, `${name} --zz-no-such-option=1 x`);
    assert.equal(isReadOnlyShell(`Get-ChildItem | ${name} -@`), false, `piped ${name} -@`);
  }
  for (const sub of ['status', 'log', 'diff', 'show', 'branch', 'rev-parse', 'ls-files', 'blame', 'describe', 'remote', 'tag', 'shortlog', 'rev-list', 'cat-file', 'grep', 'ls-tree', 'config']) {
    assert.equal(isReadOnlyShell(`git ${sub} --zz-no-such-option`), false, `git ${sub} --zz-no-such-option`);
    assert.equal(isReadOnlyShell(`git ${sub} -@`), false, `git ${sub} -@`);
  }
});

test('#3975: commands people really type still run without a card', () => {
  allowed([
    // sort, uniq, head, tail, wc, cut, tr
    'sort a.txt', 'sort -r a.txt', 'sort -n -k 2 a.txt', 'sort -u -t, -k2,2 a.txt', 'sort -nr a.txt', 'sort -k2,2nr a.txt', 'sort --reverse --numeric-sort a.txt',
    'uniq', 'uniq -c', 'uniq -c a.txt', 'uniq -d a.txt', 'uniq -f 1 a.txt', 'sort a.txt | uniq -c | sort -nr | head -5',
    'head a.txt', 'head -5 a.txt', 'head -n 5 a.txt', 'head -n5 a.txt', 'head -n -5 a.txt', 'head -c 100 a.txt', 'tail a.txt', 'tail -20 a.txt', 'tail -n 20 a.txt', 'tail -n +5 a.txt', 'tail -c 100 a.txt',
    'wc -l a.txt', 'wc -w a.txt', 'cat a.txt | wc -l', 'cut -d, -f1,3 a.txt', 'cut -c1-10 a.txt', 'tr a-z A-Z', 'tr -d "\\r"', 'tr -s " "',
    // grep family and rg
    'grep foo a.txt', 'grep -rn foo src', 'grep -c foo a.txt', 'grep -E "a|b" a.txt', 'grep -A 3 foo a.txt', 'grep -i -v foo a.txt', 'grep -rl --include "*.js" foo .', 'egrep -n foo a.txt', 'fgrep -x foo a.txt',
    'rg foo', 'rg -n -i foo src', 'rg -l --type js foo', 'rg --files -g "*.md"', 'rg -C 3 foo', "rg foo -g '!node_modules'", 'rg -e foo -e bar', 'rg --hidden --no-ignore foo', 'rg --count-matches foo', 'rg -w -F "a.b"', 'rg --json foo',
    // tree, file, stat, du, df, ls, dir
    'tree', 'tree -L 2', 'tree -a -I node_modules', 'tree -d src', 'tree -L 2 -d',
    'file a.bin', 'file -b --mime-type a.bin', 'file -i a.bin', 'file -L a.bin',
    'stat a.txt', 'stat -c %s a.txt', 'du -sh .', 'du -h --max-depth=1', 'du -ah -d 1', 'df -h', 'df -hT', 'df .',
    'ls', 'ls -la', 'ls -la src', 'ls -lh', 'ls -R', 'ls -1 --color=never', 'ls -lt | head -5', 'dir', 'dir /s',
    // small ones
    'date', 'date -u', 'date +%s', 'date -u +%Y-%m-%dT%H:%M:%SZ', 'date -I', 'date -Iseconds', 'date -d @0', 'date --utc', 'date "+%Y-%m-%d"', "date '+%Y-%m-%d %H:%M'",
    'uname', 'uname -a', 'uname -m', 'hostname', 'hostname -s', 'whoami', 'whoami /groups', 'id', 'id -u', 'id -un', 'pwd', 'which node', 'which -a node',
    'basename a/b.txt', 'basename a/b.txt .txt', 'dirname a/b', 'realpath .', 'readlink -f a', 'echo hello', 'echo -n hello', 'cat a.txt', 'cat -n a.txt',
    // git
    'git status', 'git status -s', 'git status --porcelain', 'git status -sb', 'git status --short --branch', 'git log', 'git log --oneline', 'git log --oneline -5', 'git log --oneline -n 5',
    'git log --stat -n 3', 'git log -p -1', 'git log --graph --oneline --decorate --all', 'git log --format=%H -1', 'git log --author=alice --since=2.weeks', 'git log -- a.txt', 'git log --follow -p a.txt',
    'git log main..HEAD', 'git log --oneline --no-merges origin/main..HEAD', 'git diff', 'git diff --stat', 'git diff --cached', 'git diff --name-only', 'git diff --name-status HEAD~1', 'git diff HEAD~1 -- a.txt',
    'git diff -U5', 'git diff --no-ext-diff', 'git diff main...HEAD', 'git diff -w', 'git show', 'git show HEAD', 'git show HEAD --stat', 'git show HEAD:a.txt', 'git show -s --format=%H HEAD',
    'git branch', 'git branch -a', 'git branch -v', 'git branch -vv', 'git branch -r', 'git branch --show-current', 'git branch --list', 'git branch --list "feat*"', 'git branch --contains HEAD', 'git branch --merged main',
    'git tag', 'git tag -l', 'git tag --list', 'git tag -l "v*"', 'git tag --sort=-creatordate', 'git tag --contains HEAD', 'git tag -n',
    'git remote', 'git remote -v', 'git config --get user.name', 'git config --list', 'git config --global --get user.email', 'git config -l', 'git config --get-all remote.origin.fetch', 'git config --show-origin --list',
    'git rev-parse HEAD', 'git rev-parse --abbrev-ref HEAD', 'git rev-parse --show-toplevel', 'git rev-parse --short HEAD', 'git rev-parse --is-inside-work-tree', 'git rev-parse --git-dir',
    'git ls-files', 'git ls-files --others --exclude-standard', 'git ls-files -m', 'git ls-files -s', 'git blame a.txt', 'git blame -L 1,10 a.txt', 'git describe --tags', 'git describe --always --dirty',
    'git shortlog -sn', 'git shortlog -s -n --all', 'git rev-list --count HEAD', 'git rev-list HEAD~5..HEAD', 'git cat-file -p HEAD', 'git cat-file -t HEAD', 'git ls-tree HEAD', 'git ls-tree -r --name-only HEAD',
    'git grep foo', 'git grep -n foo', 'git grep -n -i foo -- src', 'git grep -l foo', 'git grep -e foo --and -e bar', 'git grep -c foo', 'git grep -w -F "a.b"', 'git grep --no-index foo',
    // PowerShell
    'Get-ChildItem', 'Get-ChildItem -Recurse src | Select-Object -First 5', 'Get-ChildItem -Path . -Recurse -Filter *.mjs | Select-Object -First 5 FullName', 'Get-ChildItem -Force -ErrorAction SilentlyContinue',
    'ls -Recurse', 'ls -Force', 'gci -Recurse -File', 'dir -Name', 'Get-ChildItem -Directory -Depth 2', 'Get-Content a.txt', 'Get-Content a.txt -Raw', 'Get-Content a.txt -Tail 5', 'Get-Content a.txt -TotalCount 5',
    'Get-Content -Path a.txt', 'gc a.txt | Select-Object -First 3', 'cat a.txt | Select-String needle', 'type a.txt', 'Select-String -Path *.txt -Pattern foo -CaseSensitive', 'sls foo a.txt', 'Select-String -Pattern foo -Path a.txt -Context 2',
    'Get-Content a.txt | Measure-Object -Line', 'Measure-Object -Property Length -Sum', 'Get-ChildItem | Where-Object Length -gt 100', 'Get-ChildItem | Sort-Object Length -Descending', 'Get-ChildItem | Group-Object Extension',
    'Get-ChildItem | Select-Object Name, Length | Format-Table -AutoSize', 'Get-ChildItem | Format-List', 'Get-ChildItem | Out-String', 'Test-Path a.txt', 'Test-Path a.txt -PathType Leaf', 'Resolve-Path .',
    'Get-FileHash a.txt', 'Get-FileHash a.txt -Algorithm SHA256', 'Get-Date', 'Get-Date -Format yyyy-MM-dd', 'Get-Command node', 'Get-Command -Name node -All', 'Get-Process', 'Get-Process -Name node', 'Get-Item a.txt', 'gi a.txt',
    'Get-ItemProperty a.txt', 'Get-Location', 'pwd', 'Get-Help Get-ChildItem', 'Get-Help Get-ChildItem -Examples', 'Write-Output hello', 'hostname', 'whoami', 'Get-Host', 'Get-PSDrive', 'Get-PSDrive -Name C',
    'Get-Content a.json | ConvertFrom-Json', 'Get-Content a.json | ConvertFrom-Json | ConvertTo-Json -Depth 3', 'Compare-Object a b', 'Get-Content a | Get-Unique',
    // versions
    'node --version', 'node -v', 'python --version', 'git --version', 'dotnet --version', 'java -version', 'npm -v',
  ]);
});

test('#3975: maskQuotes still returns the quoted text to the caller, without changing what it returned before', () => {
  const quoted = [];
  assert.equal(maskQuotes('date "+%Y" \'a b\'', quoted), 'date "_" "_"');
  assert.deepEqual(quoted, ['+%Y', 'a b']);
  assert.equal(maskQuotes('Get-ChildItem "C:\\Q&A" | Select-Object Name'), 'Get-ChildItem "_" | Select-Object Name');
  assert.equal(maskQuotes('echo "$HOME"'), null);
});
