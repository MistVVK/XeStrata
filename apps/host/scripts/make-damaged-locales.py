"""Writes src/locales/dmg-<lang>.json: the surface for chat files the host cannot read (#5421).
Run from apps/host:  python scripts/make-damaged-locales.py
Every entry names all nine languages; the keys are dmg01.. in the order of S. dmg-en.json is the English source. Placeholders ({0}) must be kept in every
language. A separate file keeps this work from colliding with the other catalogs (the same reason sav-, upd-, df2-, pick- and the others exist)."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr"]
S = []


def add(en, zh, ja, de, es, ru, pt, ko, fr):
    S.append(dict(en=en, zh=zh, ja=ja, de=de, es=es, ru=ru, pt=pt, ko=ko, fr=fr))


add("Unreadable chats",
    "无法读取的聊天",
    "読み込めないチャット",
    "Unlesbare Chats",
    "Chats ilegibles",
    "Нечитаемые чаты",
    "Chats ilegíveis",
    "읽을 수 없는 채팅",
    "Conversations illisibles")
# dmg02 and dmg09: English as the coordinator approved it (11-16); the eight translations are GUI-1116 drafts for the release review.
add("Strata could not read these chat files. You can delete them here.",
    "Strata 无法读取这些聊天文件。你可以在这里删除它们。",
    "Strata はこれらのチャットファイルを読み込めませんでした。ここで削除できます。",
    "Strata konnte diese Chat-Dateien nicht lesen. Sie können sie hier löschen.",
    "Strata no pudo leer estos archivos de chat. Puedes eliminarlos aquí.",
    "Strata не смог прочитать эти файлы чатов. Их можно удалить здесь.",
    "O Strata não conseguiu ler estes arquivos de chat. Você pode excluí-los aqui.",
    "Strata가 이 채팅 파일들을 읽을 수 없습니다. 여기에서 삭제할 수 있습니다.",
    "Strata n'a pas pu lire ces fichiers de conversation. Vous pouvez les supprimer ici.")
add("Strata could not read this chat file, so its messages are not shown.",
    "Strata 无法读取此聊天文件，因此不显示其消息。",
    "Strata はこのチャットファイルを読み込めなかったため、メッセージは表示されません。",
    "Strata konnte diese Chat-Datei nicht lesen, daher werden ihre Nachrichten nicht angezeigt.",
    "Strata no pudo leer este archivo de chat, así que no se muestran sus mensajes.",
    "Strata не смог прочитать этот файл чата, поэтому его сообщения не показаны.",
    "O Strata não conseguiu ler este arquivo de chat, então as mensagens não são exibidas.",
    "Strata가 이 채팅 파일을 읽을 수 없어서 메시지가 표시되지 않습니다.",
    "Strata n'a pas pu lire ce fichier de conversation ; ses messages ne sont donc pas affichés.")
add("Delete chat file",
    "删除聊天文件",
    "チャットファイルを削除",
    "Chat-Datei löschen",
    "Eliminar archivo de chat",
    "Удалить файл чата",
    "Excluir arquivo de chat",
    "채팅 파일 삭제",
    "Supprimer le fichier de conversation")
add("Delete {0}.json",
    "删除 {0}.json",
    "{0}.json を削除",
    "{0}.json löschen",
    "Eliminar {0}.json",
    "Удалить {0}.json",
    "Excluir {0}.json",
    "{0}.json 삭제",
    "Supprimer {0}.json")
add("Delete this chat file?",
    "删除此聊天文件？",
    "このチャットファイルを削除しますか？",
    "Diese Chat-Datei löschen?",
    "¿Eliminar este archivo de chat?",
    "Удалить этот файл чата?",
    "Excluir este arquivo de chat?",
    "이 채팅 파일을 삭제할까요?",
    "Supprimer ce fichier de conversation ?")
add("Keep file",
    "保留文件",
    "ファイルを保持",
    "Datei behalten",
    "Conservar archivo",
    "Оставить файл",
    "Manter arquivo",
    "파일 유지",
    "Garder le fichier")
add("Delete file",
    "删除文件",
    "ファイルを削除",
    "Datei löschen",
    "Eliminar archivo",
    "Удалить файл",
    "Excluir arquivo",
    "파일 삭제",
    "Supprimer le fichier")
add("{0} was deleted. Your other chats are not affected.",
    "{0} 已删除。你的其他聊天不受影响。",
    "{0} を削除しました。ほかのチャットには影響しません。",
    "{0} wurde gelöscht. Ihre anderen Chats sind nicht betroffen.",
    "{0} se eliminó. Tus otros chats no se ven afectados.",
    "{0} удалён. Другие ваши чаты не затронуты.",
    "{0} foi excluído. Seus outros chats não são afetados.",
    "{0}을(를) 삭제했습니다. 다른 채팅에는 영향이 없습니다.",
    "{0} a été supprimé. Vos autres conversations ne sont pas concernées.")
add("The chat file could not be deleted",
    "无法删除聊天文件",
    "チャットファイルを削除できませんでした",
    "Die Chat-Datei konnte nicht gelöscht werden",
    "No se pudo eliminar el archivo de chat",
    "Не удалось удалить файл чата",
    "O arquivo de chat não pôde ser excluído",
    "채팅 파일을 삭제할 수 없습니다",
    "Le fichier de conversation n'a pas pu être supprimé")
# dmg11 (coordinator 11-16): a chat file of a newer Strata is not damaged and is never offered for deletion.
add("This chat was saved by a newer version of Strata. Update Strata to open it.",
    "此聊天由较新版本的 Strata 保存。请先更新 Strata 再打开。",
    "このチャットは新しいバージョンの Strata で保存されています。開くには Strata を更新してください。",
    "Dieser Chat wurde mit einer neueren Version von Strata gespeichert. Aktualisieren Sie Strata, um ihn zu öffnen.",
    "Este chat se guardó con una versión más reciente de Strata. Actualiza Strata para abrirlo.",
    "Этот чат сохранён более новой версией Strata. Обновите Strata, чтобы открыть его.",
    "Este chat foi salvo por uma versão mais recente do Strata. Atualize o Strata para abri-lo.",
    "이 채팅은 더 새로운 버전의 Strata에서 저장되었습니다. 열려면 Strata를 업데이트하세요.",
    "Cette conversation a été enregistrée par une version plus récente de Strata. Mettez Strata à jour pour l'ouvrir.")

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
        data[f"dmg{i + 1:02d}"] = french(text) if lang == "fr" else text
    with (here / f"dmg-{lang}.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
print("written", len(S), "strings in", len(LANGS), "languages")
