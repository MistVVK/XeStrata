"""Writes src/locales/sav-<lang>.json: the notice for chats the host could not write (owner wording 11-04, #5192).
Run from apps/host:  python scripts/make-chatsave-locales.py
Every entry names all nine languages; the keys are sav01.. in the order of S. sav-en.json is the English source. Placeholders ({0}) must be kept in every
language. A separate file keeps this work from colliding with the other catalogs (the same reason upd-, df2-, pick- and the others exist)."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr", "he"]
S = []


def add(en, zh, ja, de, es, ru, pt, ko, fr, he):
    S.append(dict(en=en, zh=zh, ja=ja, de=de, es=es, ru=ru, pt=pt, ko=ko, fr=fr, he=he))


add("Some chats could not be saved",
    "部分聊天无法保存",
    "一部のチャットを保存できませんでした",
    "Einige Chats konnten nicht gespeichert werden",
    "No se pudieron guardar algunos chats",
    "Не удалось сохранить некоторые чаты",
    "Alguns chats não puderam ser guardados",
    "일부 채팅을 저장하지 못했습니다",
    "Certaines conversations n'ont pas pu être enregistrées",
    he="חלק מהצ'אטים לא נשמרו")
add("Strata's chat storage is full, so the newest changes to {0} chat(s) are only in this window. Delete old chats to make room; they then save by themselves.",
    "Strata 的聊天存储空间已满，因此 {0} 个聊天的最新更改目前只保存在此窗口中。请删除旧聊天以腾出空间，之后它们会自动保存。",
    "Strata のチャット保存領域がいっぱいのため、{0} 件のチャットの最新の変更はこのウィンドウにしかありません。古いチャットを削除して空きを作ると、自動的に保存されます。",
    "Der Chat-Speicher von Strata ist voll, daher gibt es die neuesten Änderungen an {0} Chat(s) nur in diesem Fenster. Löschen Sie alte Chats, um Platz zu schaffen; dann werden sie automatisch gespeichert.",
    "El almacenamiento de chats de Strata está lleno, así que los cambios más recientes de {0} chat(s) solo están en esta ventana. Elimina chats antiguos para liberar espacio; después se guardarán solos.",
    "Хранилище чатов Strata заполнено, поэтому последние изменения в чатах ({0}) есть только в этом окне. Удалите старые чаты, чтобы освободить место; после этого они сохранятся сами.",
    "O armazenamento de chats do Strata está cheio, por isso as alterações mais recentes de {0} chat(s) só existem nesta janela. Apague chats antigos para libertar espaço; depois eles guardam-se sozinhos.",
    "Strata의 채팅 저장 공간이 가득 차서 {0}개 채팅의 최신 변경 내용이 이 창에만 있습니다. 오래된 채팅을 삭제해 공간을 확보하세요. 그러면 자동으로 저장됩니다.",
    "L'espace de stockage des conversations de Strata est plein : les dernières modifications de {0} conversation(s) ne sont que dans cette fenêtre. Supprimez d'anciennes conversations pour faire de la place ; elles s'enregistreront alors d'elles-mêmes.",
    he="אחסון הצ'אטים של Strata מלא, ולכן השינויים האחרונים ב-{0} צ'אטים קיימים רק בחלון הזה. יש למחוק צ'אטים ישנים כדי לפנות מקום; אחר כך הם יישמרו מעצמם.")
add("Copy unsaved chats",
    "复制未保存的聊天",
    "未保存のチャットをコピー",
    "Nicht gespeicherte Chats kopieren",
    "Copiar chats sin guardar",
    "Скопировать несохранённые чаты",
    "Copiar chats não guardados",
    "저장되지 않은 채팅 복사",
    "Copier les conversations non enregistrées",
    he="העתקת הצ'אטים שלא נשמרו")

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
        data[f"sav{i + 1:02d}"] = french(text) if lang == "fr" else text
    with (here / f"sav-{lang}.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
print("written", len(S), "strings in", len(LANGS), "languages")
