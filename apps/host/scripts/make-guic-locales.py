"""Writes src/locales/guic-<lang>.json: the strings of the 0.2.16 new-UI features of lane GUI-C (#6051 move a chat to another workspace, ...).
Run from apps/host:  python scripts/make-guic-locales.py
The languages are the app's own list (LANGUAGES in src/lib/i18n.ts), not a list kept here: a language added there makes this script stop until
every entry below names it, so a new language is one step (add its text to each entry, run the script). Each entry names its text per language
code; the keys are guic01.. in the order of S, so a new string is always appended at the end. guic-en.json is the English source. Placeholders
({0}) must be kept in every language. German uses "Sie". A separate file keeps this work from colliding with the other catalogs (the same reason
dmg-, sav-, upd-, df2-, pick- and the others exist)."""
import json
import re
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
LANGS = re.findall(r"\{value:'([a-z]{2,3}(?:-[A-Za-z]+)?)'", re.search(r"export const LANGUAGES=\[(.*?)\] as const", (APP / "src" / "lib" / "i18n.ts").read_text(encoding="utf-8")).group(1))
assert LANGS and LANGS[0] == "en", LANGS
S = []


def add(**texts):
    missing = [lang for lang in LANGS if not texts.get(lang)]
    assert not missing, (texts.get("en"), "needs a text for", missing)
    S.append(texts)


# #6051: move a chat to another workspace (the chat row's Actions menu and its dialog). English approved by the coordinator 11-16.
add(en="Move to workspace…",
    zh="移动到工作区…",
    ja="ワークスペースへ移動…",
    de="In Arbeitsbereich verschieben…",
    es="Mover a otro espacio de trabajo…",
    ru="Переместить в рабочее пространство…",
    pt="Mover para espaço de trabalho…",
    ko="작업 공간으로 이동…",
    fr="Déplacer vers un espace de travail…",
    he="העברה לסביבת עבודה…")
add(en="Move chat to another workspace",
    zh="将聊天移动到其他工作区",
    ja="チャットを別のワークスペースへ移動",
    de="Chat in einen anderen Arbeitsbereich verschieben",
    es="Mover el chat a otro espacio de trabajo",
    ru="Переместить чат в другое рабочее пространство",
    pt="Mover o chat para outro espaço de trabalho",
    ko="채팅을 다른 작업 공간으로 이동",
    fr="Déplacer la conversation vers un autre espace de travail",
    he="העברת הצ'אט לסביבת עבודה אחרת")
add(en="Workspace",
    zh="工作区",
    ja="ワークスペース",
    de="Arbeitsbereich",
    es="Espacio de trabajo",
    ru="Рабочее пространство",
    pt="Espaço de trabalho",
    ko="작업 공간",
    fr="Espace de travail",
    he="סביבת עבודה")
add(en="The chat is listed under this workspace. Its next answer works in that workspace’s folder, unless you chose a folder for this chat.",
    zh="聊天会列在这个工作区下。它的下一个回答会在该工作区的文件夹中进行，除非你为这个聊天选择了文件夹。",
    ja="チャットはこのワークスペースの下に表示されます。このチャット用にフォルダーを選んでいない限り、次の回答はそのワークスペースのフォルダーで作業します。",
    de="Der Chat wird unter diesem Arbeitsbereich aufgeführt. Seine nächste Antwort arbeitet im Ordner dieses Arbeitsbereichs, es sei denn, Sie haben für diesen Chat einen Ordner gewählt.",
    es="El chat aparece en este espacio de trabajo. Su próxima respuesta trabaja en la carpeta de ese espacio de trabajo, salvo que hayas elegido una carpeta para este chat.",
    ru="Чат будет показан в этом рабочем пространстве. Следующий ответ будет работать в папке этого пространства, если вы не выбрали папку для этого чата.",
    pt="O chat passa a aparecer neste espaço de trabalho. A próxima resposta trabalha na pasta desse espaço de trabalho, a menos que você tenha escolhido uma pasta para este chat.",
    ko="채팅이 이 작업 공간 아래에 표시됩니다. 이 채팅에 폴더를 직접 고르지 않았다면 다음 답변은 그 작업 공간의 폴더에서 작업합니다.",
    fr="La conversation est rangée dans cet espace de travail. Sa prochaine réponse travaille dans le dossier de cet espace, sauf si vous avez choisi un dossier pour cette conversation.",
    he="הצ'אט יופיע תחת סביבת העבודה הזו. התשובה הבאה שלו תעבוד בתיקייה של סביבת העבודה הזו, אלא אם בחרתם תיקייה לצ'אט הזה.")
add(en="Move chat",
    zh="移动聊天",
    ja="チャットを移動",
    de="Chat verschieben",
    es="Mover chat",
    ru="Переместить чат",
    pt="Mover chat",
    ko="채팅 이동",
    fr="Déplacer la conversation",
    he="העברת הצ'אט")
add(en="This chat is still working. Stop it or wait for it to finish, then move it.",
    zh="这个聊天仍在工作。请先停止它或等它完成，然后再移动。",
    ja="このチャットはまだ作業中です。停止するか終わるまで待ってから移動してください。",
    de="Dieser Chat arbeitet noch. Stoppen Sie ihn oder warten Sie, bis er fertig ist, und verschieben Sie ihn dann.",
    es="Este chat todavía está trabajando. Detenlo o espera a que termine y luego muévelo.",
    ru="Этот чат ещё работает. Остановите его или дождитесь окончания, затем переместите.",
    pt="Este chat ainda está trabalhando. Pare-o ou espere ele terminar e depois mova-o.",
    ko="이 채팅은 아직 작업 중입니다. 중지하거나 끝날 때까지 기다린 다음 이동하세요.",
    fr="Cette conversation travaille encore. Arrêtez-la ou attendez qu’elle se termine, puis déplacez-la.",
    he="הצ'אט הזה עדיין עובד. עצרו אותו או המתינו שיסיים, ואז העבירו אותו.")
# #5084: the person's own rules for every chat (Settings > General > Your rules). English approved by the coordinator 11-16 (hint in the coordinator's words).
add(en="Your rules",
    zh="你的规则",
    ja="あなたのルール",
    de="Ihre Regeln",
    es="Tus reglas",
    ru="Ваши правила",
    pt="Suas regras",
    ko="내 규칙",
    fr="Vos règles",
    he="הכללים שלכם")
add(en='Rules the assistant follows in every chat, for example "Answer in German" or "Keep answers short". New answers use them right away.',
    zh="助手在每个聊天中遵循的规则，例如“用德语回答”或“回答要简短”。新的回答会立即使用它们。",
    ja="アシスタントがすべてのチャットで従うルールです。たとえば「ドイツ語で答える」や「答えは短く」など。新しい回答からすぐに使われます。",
    de="Regeln, denen der Assistent in jedem Chat folgt, zum Beispiel „Antworte auf Deutsch“ oder „Halte die Antworten kurz“. Neue Antworten verwenden sie sofort.",
    es="Reglas que el asistente sigue en todos los chats, por ejemplo «Responde en alemán» o «Da respuestas cortas». Las nuevas respuestas las usan de inmediato.",
    ru="Правила, которым ассистент следует в каждом чате, например «Отвечай по-немецки» или «Отвечай кратко». Новые ответы сразу их учитывают.",
    pt="Regras que o assistente segue em todos os chats, por exemplo “Responda em alemão” ou “Dê respostas curtas”. As novas respostas já as usam.",
    ko="어시스턴트가 모든 채팅에서 따르는 규칙입니다. 예: “독일어로 답하기”, “짧게 답하기”. 새 답변부터 바로 적용됩니다.",
    fr="Des règles que l'assistant suit dans chaque conversation, par exemple « Réponds en allemand » ou « Fais des réponses courtes ». Les nouvelles réponses les utilisent tout de suite.",
    he="כללים שהעוזר ממלא בכל צ'אט, למשל “ענה בגרמנית” או “כתוב תשובות קצרות”. תשובות חדשות משתמשות בהם מיד.")
add(en="Edit rules",
    zh="编辑规则",
    ja="ルールを編集",
    de="Regeln bearbeiten",
    es="Editar reglas",
    ru="Изменить правила",
    pt="Editar regras",
    ko="규칙 편집",
    fr="Modifier les règles",
    he="עריכת הכללים")
add(en="Write one rule per line.",
    zh="每行写一条规则。",
    ja="1行に1つのルールを書いてください。",
    de="Schreiben Sie eine Regel pro Zeile.",
    es="Escribe una regla por línea.",
    ru="Пишите по одному правилу в строке.",
    pt="Escreva uma regra por linha.",
    ko="한 줄에 규칙 하나씩 쓰세요.",
    fr="Écrivez une règle par ligne.",
    he="כתבו כלל אחד בכל שורה.")
add(en="{0} of 64 KB used",
    zh="已用 {0}，共 64 KB",
    ja="{0} / 64 KB 使用中",
    de="{0} von 64 KB belegt",
    es="{0} de 64 KB usados",
    ru="Занято {0} из 64 КБ",
    pt="{0} de 64 KB usados",
    ko="64KB 중 {0} 사용",
    fr="{0} sur 64 Ko utilisés",
    he="נעשה שימוש ב-{0} מתוך 64 KB")
add(en="Save rules",
    zh="保存规则",
    ja="ルールを保存",
    de="Regeln speichern",
    es="Guardar reglas",
    ru="Сохранить правила",
    pt="Salvar regras",
    ko="규칙 저장",
    fr="Enregistrer les règles",
    he="שמירת הכללים")
add(en="These rules are too long. Make them shorter than 64 KB, then save again.",
    zh="这些规则太长了。请缩短到 64 KB 以内，然后再保存。",
    ja="ルールが長すぎます。64 KB 未満に短くしてから、もう一度保存してください。",
    de="Diese Regeln sind zu lang. Kürzen Sie sie auf unter 64 KB und speichern Sie dann erneut.",
    es="Estas reglas son demasiado largas. Acórtalas a menos de 64 KB y vuelve a guardar.",
    ru="Эти правила слишком длинные. Сократите их до менее чем 64 КБ и сохраните снова.",
    pt="Estas regras são longas demais. Deixe-as com menos de 64 KB e salve de novo.",
    ko="규칙이 너무 깁니다. 64KB보다 짧게 줄인 다음 다시 저장하세요.",
    fr="Ces règles sont trop longues. Raccourcissez-les à moins de 64 Ko, puis enregistrez de nouveau.",
    he="הכללים האלה ארוכים מדי. קצרו אותם לפחות מ-64 KB ושמרו שוב.")
add(en="The rules were changed in another window. Close this and open them again to see the newest version.",
    zh="规则已在另一个窗口中更改。请关闭此窗口并重新打开，查看最新版本。",
    ja="ルールは別のウィンドウで変更されました。これを閉じてもう一度開くと、最新の内容が表示されます。",
    de="Die Regeln wurden in einem anderen Fenster geändert. Schließen Sie dies und öffnen Sie sie erneut, um die neueste Fassung zu sehen.",
    es="Las reglas se cambiaron en otra ventana. Cierra esto y ábrelas de nuevo para ver la versión más reciente.",
    ru="Правила изменили в другом окне. Закройте это окно и откройте их снова, чтобы увидеть последнюю версию.",
    pt="As regras foram alteradas em outra janela. Feche isto e abra-as de novo para ver a versão mais recente.",
    ko="다른 창에서 규칙이 바뀌었습니다. 이 창을 닫고 다시 열어 최신 내용을 확인하세요.",
    fr="Les règles ont été modifiées dans une autre fenêtre. Fermez ceci et rouvrez-les pour voir la version la plus récente.",
    he="הכללים שונו בחלון אחר. סגרו את החלון הזה ופתחו אותם שוב כדי לראות את הגרסה העדכנית.")
# Owner 0.2.16: STRATA_ROUTE_TAIL_SKIP as a choice in Expert settings (strata_control/speed_switches.py route_tail_skip). English approved by the coordinator 11-16.
add(en="Skip rarely used parts of the model",
    zh="跳过模型中很少用到的部分",
    ja="モデルのあまり使われない部分を省く",
    de="Selten genutzte Teile des Modells überspringen",
    es="Omitir partes del modelo que casi no se usan",
    ru="Пропускать редко используемые части модели",
    pt="Pular partes do modelo pouco usadas",
    ko="모델에서 거의 쓰이지 않는 부분 건너뛰기",
    fr="Ignorer les parties du modèle peu utilisées",
    he="דילוג על חלקים במודל שכמעט לא בשימוש")
add(en="Faster answers: parts of the model that matter least for a word are skipped when they are not on the graphics card yet. Answers can change slightly. On by default where it helps.",
    zh="更快的回答：对某个词影响最小的模型部分，如果还不在显卡上，就会被跳过。回答可能略有变化。在有帮助的地方默认开启。",
    ja="回答が速くなります：ある語にとって最も重要度の低いモデルの部分が、まだグラフィックカードにない場合は省かれます。回答がわずかに変わることがあります。効果がある場合は既定でオンです。",
    de="Schnellere Antworten: Teile des Modells, die für ein Wort am wenigsten zählen, werden übersprungen, wenn sie noch nicht auf der Grafikkarte sind. Antworten können sich leicht ändern. Standardmäßig an, wo es hilft.",
    es="Respuestas más rápidas: las partes del modelo que menos importan para una palabra se omiten cuando aún no están en la tarjeta gráfica. Las respuestas pueden cambiar un poco. Activado por defecto donde ayuda.",
    ru="Более быстрые ответы: части модели, которые для слова важны меньше всего, пропускаются, если их ещё нет на видеокарте. Ответы могут немного измениться. По умолчанию включено там, где это помогает.",
    pt="Respostas mais rápidas: as partes do modelo que menos importam para uma palavra são puladas quando ainda não estão na placa de vídeo. As respostas podem mudar um pouco. Ligado por padrão onde ajuda.",
    ko="더 빠른 답변: 한 단어에 가장 덜 중요한 모델 부분이 아직 그래픽 카드에 없으면 건너뜁니다. 답변이 조금 달라질 수 있습니다. 도움이 되는 경우 기본으로 켜져 있습니다.",
    fr="Réponses plus rapides : les parties du modèle qui comptent le moins pour un mot sont ignorées quand elles ne sont pas encore sur la carte graphique. Les réponses peuvent légèrement changer. Activé par défaut là où cela aide.",
    he="תשובות מהירות יותר: חלקים במודל שחשובים הכי פחות למילה מסוימת מדולגים כשהם עדיין לא נמצאים בכרטיס המסך. התשובות עשויות להשתנות מעט. מופעל כברירת מחדל במקום שזה עוזר.")
add(en="Automatic (recommended)",
    zh="自动（推荐）",
    ja="自動（推奨）",
    de="Automatisch (empfohlen)",
    es="Automático (recomendado)",
    ru="Автоматически (рекомендуется)",
    pt="Automático (recomendado)",
    ko="자동(권장)",
    fr="Automatique (recommandé)",
    he="אוטומטי (מומלץ)")
add(en="Lower numbers skip more parts and are faster; answers can change more. Only 7 has been measured.",
    zh="数字越小，跳过的部分越多，速度越快；回答的变化也可能更大。只有 7 经过了测量。",
    ja="数字が小さいほど多くの部分を省き、速くなります。回答の変化も大きくなることがあります。測定済みなのは 7 だけです。",
    de="Kleinere Zahlen überspringen mehr Teile und sind schneller; Antworten können sich stärker ändern. Nur 7 wurde gemessen.",
    es="Los números más bajos omiten más partes y son más rápidos; las respuestas pueden cambiar más. Solo se ha medido el 7.",
    ru="Чем меньше число, тем больше частей пропускается и тем быстрее; ответы могут меняться сильнее. Измерено только значение 7.",
    pt="Números menores pulam mais partes e são mais rápidos; as respostas podem mudar mais. Só o 7 foi medido.",
    ko="숫자가 작을수록 더 많은 부분을 건너뛰어 빨라지지만 답변이 더 많이 달라질 수 있습니다. 측정된 값은 7뿐입니다.",
    fr="Les petits nombres ignorent plus de parties et sont plus rapides ; les réponses peuvent changer davantage. Seul 7 a été mesuré.",
    he="מספרים נמוכים יותר מדלגים על יותר חלקים והם מהירים יותר; התשובות עשויות להשתנות יותר. רק 7 נמדד.")

# Owner 0.2.16: the chat status line says what the model does now (src/lib/live-status.ts statusWords). English approved by the coordinator 11-16.
add(en="Waiting for the model…",
    zh="正在等待模型…",
    ja="モデルを待っています…",
    de="Warten auf das Modell…",
    es="Esperando al modelo…",
    ru="Ожидание модели…",
    pt="Aguardando o modelo…",
    ko="모델을 기다리는 중…",
    fr="En attente du modèle…",
    he="ממתין למודל…")
add(en="Calling {0}…",
    zh="正在调用 {0}…",
    ja="{0} を呼び出し中…",
    de="{0} wird aufgerufen…",
    es="Llamando a {0}…",
    ru="Вызов {0}…",
    pt="Chamando {0}…",
    ko="{0} 호출 중…",
    fr="Appel de {0}…",
    he="קורא ל-{0}…")
add(en="Writing file {0}…",
    zh="正在写入文件 {0}…",
    ja="ファイル {0} を書き込み中…",
    de="Datei {0} wird geschrieben…",
    es="Escribiendo el archivo {0}…",
    ru="Запись файла {0}…",
    pt="Escrevendo o arquivo {0}…",
    ko="파일 {0} 쓰는 중…",
    fr="Écriture du fichier {0}…",
    he="כותב את הקובץ {0}…")
add(en="Running {0}…",
    zh="正在运行 {0}…",
    ja="{0} を実行中…",
    de="{0} läuft…",
    es="Ejecutando {0}…",
    ru="Выполняется {0}…",
    pt="Executando {0}…",
    ko="{0} 실행 중…",
    fr="Exécution de {0}…",
    he="מריץ את {0}…")
# #5529: set up without a local engine (a link on the first-run engine card and the muted line under it). English approved by the coordinator.
add(en="Only connect to Strata on another computer",
    zh="只连接到另一台电脑上的 Strata",
    ja="別のコンピューターの Strata にだけ接続する",
    de="Nur mit Strata auf einem anderen Computer verbinden",
    es="Conectarse solo a Strata en otro equipo",
    ru="Только подключиться к Strata на другом компьютере",
    pt="Apenas conectar ao Strata em outro computador",
    ko="다른 컴퓨터의 Strata에만 연결",
    fr="Se connecter uniquement à Strata sur un autre ordinateur",
    he="להתחבר רק ל-Strata במחשב אחר")
add(en="Nothing is downloaded. You can add the engine later from the Models page.",
    zh="不会下载任何内容。你之后可以在模型页面添加引擎。",
    ja="何もダウンロードされません。エンジンは後で「モデル」ページから追加できます。",
    de="Es wird nichts heruntergeladen. Sie können die Engine später auf der Seite „Modelle“ hinzufügen.",
    es="No se descarga nada. Puedes añadir el motor más tarde desde la página Modelos.",
    ru="Ничего не загружается. Движок можно добавить позже на странице «Модели».",
    pt="Nada é baixado. Você pode adicionar o mecanismo depois, na página Modelos.",
    ko="아무것도 다운로드되지 않습니다. 엔진은 나중에 모델 페이지에서 추가할 수 있습니다.",
    fr="Rien n’est téléchargé. Vous pourrez ajouter le moteur plus tard depuis la page Modèles.",
    he="שום דבר לא יורד. אפשר להוסיף את המנוע מאוחר יותר מעמוד המודלים.")
here = APP / "src" / "locales"
NBSP = " "


def french(text):
    text = re.sub(r"(?<=[A-Za-zÀ-ſ])'(?=[A-Za-zÀ-ſ{])", "’", text)
    text = re.sub(r"(?<=\S) ([:;?!])", NBSP + r"\1", text)
    text = re.sub(r"« ", "«" + NBSP, text)
    text = re.sub(r" »", NBSP + "»", text)
    return text


for lang in LANGS:
    data = {}
    for i, entry in enumerate(S):
        text = entry[lang]
        assert sorted(re.findall(r"\{\d+\}", entry["en"])) == sorted(re.findall(r"\{\d+\}", text)), (lang, entry["en"], text)
        data[f"guic{i + 1:02d}"] = french(text) if lang == "fr" else text
    with (here / f"guic-{lang}.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
print("written", len(S), "strings in", len(LANGS), "languages")
