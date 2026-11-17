"""Writes src/locales/upd-<lang>.json: the strings of the engine update beside a loaded model (the dialog of Settings > About, branch f-0291-ui-final).
Run from apps/host:  python scripts/make-update-locales.py
Every entry names all nine languages; the keys are upd01.. in the order of S. upd-en.json is the English source. Placeholders ({0}) must be kept in every
language. A separate file keeps this work from colliding with the other catalogs (the same reason pick-, tune-, tools-, fix-, gh-, ui-, engine- and df2- exist)."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr", "he"]
S = []


def add(en, zh, ja, de, es, ru, pt, ko, fr, he):
    S.append(dict(en=en, zh=zh, ja=ja, de=de, es=es, ru=ru, pt=pt, ko=ko, fr=fr, he=he))


add("To update the engine, Strata has to stop the model that is loaded: {0}.",
    "要更新引擎，Strata 必须先停止当前已加载的模型：{0}。",
    "エンジンを更新するには、読み込み中のモデルを Strata が停止する必要があります：{0}。",
    "Um die Engine zu aktualisieren, muss Strata das geladene Modell anhalten: {0}.",
    "Para actualizar el motor, Strata tiene que detener el modelo cargado: {0}.",
    "Чтобы обновить движок, Strata должна остановить загруженную модель: {0}.",
    "Para atualizar o motor, o Strata precisa parar o modelo carregado: {0}.",
    "엔진을 업데이트하려면 Strata가 불러온 모델을 중지해야 합니다: {0}.",
    "Pour mettre à jour le moteur, Strata doit arrêter le modèle chargé : {0}.",
    he="כדי לעדכן את המנוע, Strata צריכה לעצור את המודל הטעון: {0}.")
add("The new engine downloads first while the model keeps answering. Then the model stops (answers in progress finish first), the engine is updated, and you can start the model again with the same settings. Your models and settings are kept.",
    "新引擎会先下载，此时模型仍可继续回答。然后模型停止（正在进行的回答会先完成），引擎随即更新，之后你可以用相同的设置再次启动该模型。你的模型和设置都会保留。",
    "まず新しいエンジンをダウンロードします。その間、モデルは応答を続けます。その後モデルが停止し（進行中の回答は先に完了します）、エンジンが更新され、同じ設定でモデルをもう一度起動できます。モデルと設定はそのまま残ります。",
    "Zuerst wird die neue Engine heruntergeladen, während das Modell weiter antwortet. Danach hält das Modell an (laufende Antworten werden zuerst beendet), die Engine wird aktualisiert, und Sie können das Modell mit denselben Einstellungen wieder starten. Ihre Modelle und Einstellungen bleiben erhalten.",
    "Primero se descarga el motor nuevo mientras el modelo sigue respondiendo. Después el modelo se detiene (las respuestas en curso terminan antes), se actualiza el motor y puedes volver a iniciar el modelo con los mismos ajustes. Tus modelos y ajustes se conservan.",
    "Сначала новый движок скачивается, пока модель продолжает отвечать. Затем модель останавливается (ответы в процессе сначала завершаются), движок обновляется, и вы можете снова запустить модель с теми же настройками. Ваши модели и настройки сохраняются.",
    "Primeiro o novo motor é baixado enquanto o modelo continua a responder. Depois o modelo para (as respostas em curso terminam antes), o motor é atualizado e pode iniciar o modelo de novo com as mesmas definições. Os seus modelos e definições são mantidos.",
    "먼저 모델이 계속 응답하는 동안 새 엔진을 다운로드합니다. 그다음 모델이 중지되고(진행 중인 답변은 먼저 끝납니다) 엔진이 업데이트되며, 같은 설정으로 모델을 다시 시작할 수 있습니다. 모델과 설정은 그대로 유지됩니다.",
    "Le nouveau moteur est d'abord téléchargé pendant que le modèle continue de répondre. Ensuite le modèle s'arrête (les réponses en cours se terminent d'abord), le moteur est mis à jour, et vous pouvez redémarrer le modèle avec les mêmes réglages. Vos modèles et réglages sont conservés.",
    he="המנוע החדש יורד קודם, בזמן שהמודל ממשיך לענות. אחר כך המודל נעצר (תשובות בתהליך מסתיימות קודם), המנוע מתעדכן, ואפשר להפעיל את המודל שוב עם אותן הגדרות. המודלים וההגדרות שלכם נשמרים.")
add("Stop and update", "停止并更新", "停止して更新", "Anhalten und aktualisieren", "Detener y actualizar", "Остановить и обновить", "Parar e atualizar", "중지하고 업데이트", "Arrêter et mettre à jour",
    he="עצירה ועדכון")
add("Getting the new engine ready. {0} keeps working until it is ready.",
    "正在准备新引擎。在准备好之前，{0} 会继续工作。",
    "新しいエンジンを準備しています。準備ができるまで {0} は動き続けます。",
    "Die neue Engine wird vorbereitet. {0} arbeitet weiter, bis sie bereit ist.",
    "Preparando el motor nuevo. {0} sigue funcionando hasta que esté listo.",
    "Готовим новый движок. {0} продолжает работать, пока он не будет готов.",
    "A preparar o novo motor. {0} continua a funcionar até estar pronto.",
    "새 엔진을 준비하는 중입니다. 준비될 때까지 {0}은(는) 계속 작동합니다.",
    "Préparation du nouveau moteur. {0} continue de fonctionner jusqu'à ce qu'il soit prêt.",
    he="מכין את המנוע החדש. {0} ממשיך לעבוד עד שהוא יהיה מוכן.")
add("Stopping {0}. Answers in progress finish first.",
    "正在停止 {0}。正在进行的回答会先完成。",
    "{0} を停止しています。進行中の回答は先に完了します。",
    "{0} wird angehalten. Laufende Antworten werden zuerst beendet.",
    "Deteniendo {0}. Las respuestas en curso terminan antes.",
    "Останавливаем {0}. Ответы в процессе сначала завершаются.",
    "A parar {0}. As respostas em curso terminam antes.",
    "{0}을(를) 중지하는 중입니다. 진행 중인 답변은 먼저 끝납니다.",
    "Arrêt de {0}. Les réponses en cours se terminent d'abord.",
    he="עוצר את {0}. תשובות בתהליך יסתיימו קודם.")
add("{0} is stopped. Updating the engine…",
    "{0} 已停止。正在更新引擎…",
    "{0} を停止しました。エンジンを更新しています…",
    "{0} ist angehalten. Die Engine wird aktualisiert…",
    "{0} está detenido. Actualizando el motor…",
    "{0} остановлена. Обновляем движок…",
    "{0} está parado. A atualizar o motor…",
    "{0}이(가) 중지되었습니다. 엔진을 업데이트하는 중입니다…",
    "{0} est arrêté. Mise à jour du moteur…",
    he="{0} נעצר. מעדכן את המנוע…")
add("Updating the engine", "正在更新引擎", "エンジンを更新しています", "Engine wird aktualisiert", "Actualizando el motor", "Обновление движка", "A atualizar o motor", "엔진 업데이트 중", "Mise à jour du moteur",
    he="עדכון המנוע")
add("The engine is updated", "引擎已更新", "エンジンを更新しました", "Die Engine ist aktualisiert", "El motor está actualizado", "Движок обновлён", "O motor está atualizado", "엔진이 업데이트되었습니다", "Le moteur est à jour",
    he="המנוע עודכן")
add("The engine was not updated", "引擎未更新", "エンジンは更新されませんでした", "Die Engine wurde nicht aktualisiert", "El motor no se actualizó", "Движок не обновлён", "O motor não foi atualizado", "엔진이 업데이트되지 않았습니다", "Le moteur n'a pas été mis à jour",
    he="המנוע לא עודכן")
add("The engine is updated. Start {0} again with the same settings?",
    "引擎已更新。要用相同的设置再次启动 {0} 吗？",
    "エンジンを更新しました。同じ設定で {0} をもう一度起動しますか？",
    "Die Engine ist aktualisiert. {0} mit denselben Einstellungen wieder starten?",
    "El motor está actualizado. ¿Iniciar {0} de nuevo con los mismos ajustes?",
    "Движок обновлён. Запустить {0} снова с теми же настройками?",
    "O motor está atualizado. Iniciar {0} de novo com as mesmas definições?",
    "엔진이 업데이트되었습니다. 같은 설정으로 {0}을(를) 다시 시작할까요?",
    "Le moteur est à jour. Redémarrer {0} avec les mêmes réglages ?",
    he="המנוע עודכן. להפעיל שוב את {0} עם אותן הגדרות?")
add("The update didn't finish. Your old engine is unchanged and still works. Start {0} again on it?",
    "更新没有完成。你原来的引擎没有改动，仍然可用。要在它上面再次启动 {0} 吗？",
    "更新は完了しませんでした。以前のエンジンは変更されておらず、そのまま使えます。そのエンジンで {0} をもう一度起動しますか？",
    "Die Aktualisierung wurde nicht abgeschlossen. Ihre alte Engine ist unverändert und funktioniert weiter. {0} damit wieder starten?",
    "La actualización no terminó. Tu motor anterior no cambió y sigue funcionando. ¿Iniciar {0} de nuevo con él?",
    "Обновление не завершилось. Прежний движок не изменён и работает. Запустить {0} снова на нём?",
    "A atualização não terminou. O seu motor anterior não foi alterado e continua a funcionar. Iniciar {0} de novo com ele?",
    "업데이트가 끝나지 않았습니다. 기존 엔진은 그대로이며 계속 작동합니다. 기존 엔진으로 {0}을(를) 다시 시작할까요?",
    "La mise à jour ne s'est pas terminée. Votre ancien moteur est inchangé et fonctionne toujours. Redémarrer {0} avec lui ?",
    he="העדכון לא הסתיים. המנוע הישן שלכם לא השתנה ועדיין עובד. להפעיל עליו שוב את {0}?")
add("Not now", "暂不", "今はしない", "Jetzt nicht", "Ahora no", "Не сейчас", "Agora não", "나중에", "Pas maintenant",
    he="לא עכשיו")
add("Starting {0}. Loading can take a few minutes.",
    "正在启动 {0}。加载可能需要几分钟。",
    "{0} を起動しています。読み込みには数分かかることがあります。",
    "{0} wird gestartet. Das Laden kann einige Minuten dauern.",
    "Iniciando {0}. La carga puede tardar unos minutos.",
    "Запускаем {0}. Загрузка может занять несколько минут.",
    "A iniciar {0}. O carregamento pode demorar alguns minutos.",
    "{0}을(를) 시작하는 중입니다. 불러오는 데 몇 분이 걸릴 수 있습니다.",
    "Démarrage de {0}. Le chargement peut prendre quelques minutes.",
    he="מפעיל את {0}. הטעינה יכולה להימשך כמה דקות.")
add("Starting {0} again", "正在再次启动 {0}", "{0} をもう一度起動しています", "{0} wird wieder gestartet", "Iniciando {0} de nuevo", "Снова запускаем {0}", "A iniciar {0} de novo", "{0}을(를) 다시 시작하는 중", "Redémarrage de {0}",
    he="מפעיל שוב את {0}")
add("The model did not start again", "模型没有重新启动", "モデルを再起動できませんでした", "Das Modell wurde nicht wieder gestartet", "El modelo no se inició de nuevo", "Модель не запустилась снова", "O modelo não voltou a iniciar", "모델이 다시 시작되지 않았습니다", "Le modèle n'a pas redémarré",
    he="המודל לא הופעל שוב")
add("You can load it again from Models.", "你可以在“模型”中再次加载它。", "「モデル」からもう一度読み込めます。", "Sie können es unter „Modelle“ erneut laden.", "Puedes cargarlo de nuevo desde Modelos.", "Вы можете загрузить её снова в разделе «Модели».", "Pode carregá-lo de novo em Modelos.", "모델 화면에서 다시 불러올 수 있습니다.", "Vous pouvez le recharger depuis Modèles.",
    he="אפשר לטעון אותו שוב ממודלים.")
add("A model that was not started from this app is running, so nothing was changed. Stop it, then update the engine again.",
    "有一个不是从此应用启动的模型正在运行，因此没有做任何更改。请先停止它，然后再次更新引擎。",
    "このアプリから起動していないモデルが動作しているため、何も変更していません。それを停止してから、もう一度エンジンを更新してください。",
    "Ein Modell, das nicht aus dieser App gestartet wurde, läuft noch, daher wurde nichts geändert. Halten Sie es an und aktualisieren Sie die Engine dann erneut.",
    "Hay un modelo en ejecución que no se inició desde esta aplicación, así que no se cambió nada. Deténlo y vuelve a actualizar el motor.",
    "Работает модель, запущенная не из этого приложения, поэтому ничего не изменено. Остановите её и обновите движок снова.",
    "Está em execução um modelo que não foi iniciado a partir desta aplicação, por isso nada foi alterado. Pare-o e atualize o motor de novo.",
    "이 앱에서 시작하지 않은 모델이 실행 중이어서 아무것도 변경하지 않았습니다. 그 모델을 중지한 다음 엔진을 다시 업데이트하세요.",
    "Un modèle qui n'a pas été démarré depuis cette application est en cours d'exécution, rien n'a donc été modifié. Arrêtez-le, puis mettez à nouveau le moteur à jour.",
    he="פועל מודל שלא הופעל מהאפליקציה הזו, ולכן לא שונה דבר. יש לעצור אותו ולעדכן שוב את המנוע.")

here = Path(__file__).resolve().parent.parent / "src" / "locales"
NBSP = "\u00a0"


def french(text):
    text = re.sub(r"(?<=[A-Za-zÀ-ſ])'(?=[A-Za-zÀ-ſ{])", "’", text)
    text = re.sub(r"(?<=\S) ([:;?!])", NBSP + r"\1", text)
    return text


for lang in LANGS:
    data = {}
    for i, entry in enumerate(S):
        text = entry[lang]
        assert sorted(re.findall(r"\{\d+\}", entry["en"])) == sorted(re.findall(r"\{\d+\}", text)), (lang, entry["en"], text)
        data[f"upd{i + 1:02d}"] = french(text) if lang == "fr" else text
    with (here / f"upd-{lang}.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
print("written", len(S), "strings in", len(LANGS), "languages")
