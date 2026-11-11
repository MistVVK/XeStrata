"""Writes src/locales/net-<lang>.json: the line shown while a download server has not answered yet (lane NET, owner wording 11-10, W14).
Run from apps/host:  python scripts/make-net-locales.py
The English text is strata_control/netpolicy.py WORDS["waiting"] with {host} -> {0} and {seconds} -> {1}; setup prints it and
strata_control/progress_words.py hands it to the app as the install step's label. Every entry names all nine languages; keys are net01.. in
the order of S. Placeholders ({0}, {1}) must be kept in every language. The eight translations are first drafts for the owner's later look."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr"]
S = []


def add(en, zh, ja, de, es, ru, pt, ko, fr):
    S.append(dict(en=en, zh=zh, ja=ja, de=de, es=es, ru=ru, pt=pt, ko=ko, fr=fr))


add("Waiting for {0} to answer ({1} s so far)",
    "正在等待 {0} 响应（已等待 {1} 秒）",
    "{0} の応答を待っています（{1} 秒経過）",
    "Warten auf eine Antwort von {0} (bisher {1} s)",
    "Esperando la respuesta de {0} ({1} s hasta ahora)",
    "Ожидание ответа от {0} (прошло {1} с)",
    "A aguardar a resposta de {0} ({1} s até agora)",
    "{0}의 응답을 기다리는 중 ({1}초 경과)",
    "En attente de la réponse de {0} ({1} s jusqu'ici)")

here = Path(__file__).resolve().parent.parent / "src" / "locales"
NBSP = " "


def french(text):
    text = re.sub(r"(?<=[A-Za-zÀ-ſ])'(?=[A-Za-zÀ-ſ{])", "’", text)
    text = re.sub(r"(?<=\S) ([:;?!])", NBSP + r"\1", text)
    return text


for lang in LANGS:
    data = {}
    for i, entry in enumerate(S):
        text = entry[lang]
        assert sorted(re.findall(r"\{\d+\}", entry["en"])) == sorted(re.findall(r"\{\d+\}", text)), (lang, entry["en"], text)
        data[f"net{i + 1:02d}"] = french(text) if lang == "fr" else text
    with (here / f"net-{lang}.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
print("written", len(S), "strings in", len(LANGS), "languages")
