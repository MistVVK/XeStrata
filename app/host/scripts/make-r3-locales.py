"""Writes src/locales/r3-<lang>.json: the strings added by the owner's GUI feedback round 3 (branch f-0291-ui-r3).
Run from apps/host:  python scripts/make-r3-locales.py
Every entry names all nine languages; the keys are r3_01.. in the order of S. r3-en.json is the English source. Placeholders ({0}, {1}) must be kept in every
language. A separate file keeps this work from colliding with the other catalogs (the same reason pick-, upd-, df2- and the others exist)."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr"]
S = []


def add(en, zh, ja, de, es, ru, pt, ko, fr):
    S.append(dict(en=en, zh=zh, ja=ja, de=de, es=es, ru=ru, pt=pt, ko=ko, fr=fr))


# 2: the Read pictures row while the installed engine has no picture reader (a tooltip, not a paragraph)
add("Needs an engine update",
    "需要更新引擎", "エンジンの更新が必要です", "Braucht ein Engine-Update", "Necesita actualizar el motor", "Нужно обновить движок", "Precisa de uma atualização do motor", "엔진 업데이트가 필요합니다", "Nécessite une mise à jour du moteur")

# 5: the one word the version list writes for a version that does not fully fit in memory
add("SSD streaming · will run slower",
    "SSD 流式读取 · 运行会更慢", "SSD ストリーミング · 動作は遅くなります", "SSD-Streaming · läuft langsamer", "Streaming desde el SSD · irá más lento", "Потоковая подгрузка с SSD · будет работать медленнее",
    "Streaming do SSD · vai rodar mais devagar", "SSD 스트리밍 · 더 느리게 실행됩니다", "Streaming depuis le SSD · plus lent")

# 6: the warning under the conversation context when a long context leaves little graphics memory for the model (nothing is refused or lowered)
add("This much context takes about {0} GB of graphics memory and leaves little room for the model on this card. Strata will try it; if it does not start, choose a shorter context.",
    "这么长的上下文大约占用 {0} GB 显存，留给模型的空间很少。Strata 会尝试运行；如果无法启动，请选择更短的上下文。",
    "このコンテキストはグラフィックスメモリーを約 {0} GB 使い、このカードではモデルに残る余裕がわずかです。Strata は試しますが、起動しない場合はより短いコンテキストを選んでください。",
    "So viel Kontext belegt etwa {0} GB Grafikspeicher und lässt auf dieser Karte wenig Platz für das Modell. Strata versucht es; startet es nicht, wähle einen kürzeren Kontext.",
    "Este contexto ocupa unos {0} GB de memoria gráfica y deja poco espacio para el modelo en esta tarjeta. Strata lo intentará; si no arranca, elige un contexto más corto.",
    "Такой контекст занимает около {0} ГБ видеопамяти и оставляет модели мало места на этой карте. Strata попробует запустить; если не получится, выберите контекст короче.",
    "Esse contexto ocupa cerca de {0} GB de memória gráfica e deixa pouco espaço para o modelo nesta placa. O Strata vai tentar; se não iniciar, escolha um contexto menor.",
    "이 정도의 컨텍스트는 그래픽 메모리를 약 {0} GB 사용하며 이 카드에서는 모델이 쓸 공간이 거의 남지 않습니다. Strata가 시도하지만 시작되지 않으면 더 짧은 컨텍스트를 선택하세요.",
    "Ce contexte occupe environ {0} Go de mémoire graphique et laisse peu de place au modèle sur cette carte. Strata va essayer ; s'il ne démarre pas, choisissez un contexte plus court.")

# 9: a chat that is working while the person is elsewhere (the ring on its row, and the model chip's line)
add("Working · {0}",
    "进行中 · {0}", "作業中 · {0}", "In Arbeit · {0}", "Trabajando · {0}", "В работе · {0}", "Trabalhando · {0}", "작업 중 · {0}", "En cours de travail · {0}")
add("{0} chats working",
    "{0} 个对话进行中", "{0} 件のチャットが作業中", "{0} Chats in Arbeit", "{0} chats trabajando", "В работе чатов: {0}", "{0} chats trabalhando", "대화 {0}개 작업 중", "{0} conversations en cours")

# 10: the size line of a file being written
add("{0} lines",
    "{0} 行", "{0} 行", "{0} Zeilen", "{0} líneas", "{0} строк", "{0} linhas", "{0}줄", "{0} lignes")

#@@MORE@@

here = Path(__file__).resolve().parent.parent / "src" / "locales"
NBSP = " "


def french(text):
    text = re.sub(r"(?<=[A-Za-zÀ-ſ])'(?=[A-Za-zÀ-ſ{])", "’", text)
    text = re.sub(r"(?<=\S) ([:;?!])", NBSP + r"\1", text)
    text = re.sub(r"(?<=[\d}]) %", NBSP + "%", text)
    return text


for lang in LANGS:
    data = {}
    for i, entry in enumerate(S):
        text = entry[lang]
        assert sorted(re.findall(r"\{\d+\}", entry["en"])) == sorted(re.findall(r"\{\d+\}", text)), (lang, entry["en"], text)
        data[f"r3_{i + 1:02d}"] = french(text) if lang == "fr" else text
    (here / f"r3-{lang}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("written", len(S), "strings in", len(LANGS), "languages")
