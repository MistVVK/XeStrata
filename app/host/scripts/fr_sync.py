"""French (fr) catalog tooling for the desktop app (#3970).

The French texts live in src/locales/fr.json, fix-fr.json, tune-fr.json and tools-fr.json, next to the other languages.
They are edited by hand (they are the translations, not generated), so a string added to an English catalog later has no
French yet. This script lists and merges those. From apps/host:

    python scripts/fr_sync.py missing [out.tsv]        write every English string that has no French yet
    python scripts/fr_sync.py apply done.tsv           merge translations, rewrite the four fr files in English key order
    python scripts/fr_sync.py check                    exit 1 if a French string is missing, empty or breaks a placeholder / tag

The TSV has one string per line:  <id><TAB><text>.  In the text, a newline is written \\n and a backslash \\\\.
In the "done" file a lone  =  means "keep the English text as it is" (class names, product names, file names).
The id prefix picks the file:  s0001 -> fr.json, fx01 -> fix-fr.json, tune01 -> tune-fr.json, tl01 -> tools-fr.json.
Use "vous", plain short words, a next step, and a no-break space (U+00A0) before : ; ? ! and inside guillemets.
"""
import json
import re
import sys
from pathlib import Path

LOCALES = Path(__file__).resolve().parent.parent / "src" / "locales"
PAIRS = [("en.json", "fr.json"), ("fix-en.json", "fix-fr.json"), ("tune-en.json", "tune-fr.json"), ("tools-en.json", "tools-fr.json")]
SLOT = re.compile(r"\{\d+\}")
TAG = re.compile(r"</?[A-Za-z][^>]*>")


def load(name):
    path = LOCALES / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def file_of(identifier):
    for en_name, fr_name in PAIRS:
        if identifier in load(en_name):
            return fr_name
    raise SystemExit(f"unknown id {identifier}: it is in no English catalog")


def escape(text):
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def unescape(text):
    out, i = [], 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            out.append({"n": "\n", "\\": "\\"}.get(text[i + 1], text[i + 1]))
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


NBSP = " "


def typography(text):
    """French spacing and apostrophes, applied to every translated (not kept) string.
    l'image -> l'image with a typographic apostrophe (also before a {0} slot); a no-break space before : ; ? ! % and inside the guillemets."""
    text = re.sub(r"(?<=[A-Za-zÀ-ſ])'(?=[A-Za-zÀ-ſ{])", "’", text)
    text = re.sub(r"(?<=\S) ([:;?!])", NBSP + r"\1", text)
    text = re.sub(r"(?<=[\d}]) %", NBSP + "%", text)
    text = re.sub("« ?", "«" + NBSP, text)
    text = re.sub(" ?»", NBSP + "»", text)
    return text


def problems(identifier, source, text):
    found = []
    if not text.strip():
        found.append("empty")
    expected = [slot for slot in SLOT.findall(source) if not (identifier == "s2463" and slot == "{1}")]  # the tests allow this one to drop {1} (the plural suffix)
    if sorted(expected) != sorted(SLOT.findall(text)):
        found.append("placeholders differ")
    if sorted(TAG.findall(source)) != sorted(TAG.findall(text)):
        found.append("markup differs")
    return found


def missing():
    todo = {}
    for en_name, fr_name in PAIRS:
        en, fr = load(en_name), load(fr_name)
        for identifier, text in en.items():
            if identifier not in fr:
                todo[identifier] = text
    return todo


def main(argv):
    command = argv[1] if len(argv) > 1 else ""
    if command == "missing":
        todo = missing()
        lines = [f"{identifier}\t{escape(text)}" for identifier, text in todo.items()]
        if len(argv) > 2:
            Path(argv[2]).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            print(f"{len(lines)} strings without French written to {argv[2]}")
        else:
            print("\n".join(lines))
            print(f"{len(lines)} strings without French", file=sys.stderr)
        return 0
    if command == "apply" and len(argv) > 2:
        incoming = {}
        for number, line in enumerate(Path(argv[2]).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            if "\t" not in line:
                raise SystemExit(f"{argv[2]}:{number}: no tab")
            identifier, text = line.split("\t", 1)
            incoming[identifier.strip()] = text
        bad = 0
        for en_name, fr_name in PAIRS:
            en, fr = load(en_name), load(fr_name)
            changed = False
            for identifier, source in en.items():
                if identifier in incoming:
                    text = source if incoming[identifier] == "=" else typography(unescape(incoming[identifier]))
                    for why in problems(identifier, source, text):
                        print(f"{identifier}: {why}: {source!r} -> {text!r}")
                        bad += 1
                    fr[identifier] = text
                    changed = True
            if changed:
                ordered = {identifier: fr[identifier] for identifier in en if identifier in fr}
                (LOCALES / fr_name).write_text(json.dumps(ordered, ensure_ascii=False, indent=1 if en_name != "en.json" else 2) + "\n", encoding="utf-8")
        unknown = [identifier for identifier in incoming if not any(identifier in load(en_name) for en_name, _ in PAIRS)]
        for identifier in unknown:
            print(f"{identifier}: not in any English catalog")
            bad += 1
        print(f"applied {len(incoming)} strings, {bad} problems")
        return 1 if bad else 0
    if command == "check":
        bad = 0
        for en_name, fr_name in PAIRS:
            en, fr = load(en_name), load(fr_name)
            for identifier, source in en.items():
                if identifier not in fr:
                    print(f"{fr_name}: {identifier} has no French yet: {source!r}")
                    bad += 1
                    continue
                for why in problems(identifier, source, fr[identifier]):
                    print(f"{fr_name}: {identifier}: {why}")
                    bad += 1
            for identifier in fr:
                if identifier not in en:
                    print(f"{fr_name}: {identifier} is not in {en_name}")
                    bad += 1
        print("French catalogs complete" if not bad else f"{bad} problems")
        return 1 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
