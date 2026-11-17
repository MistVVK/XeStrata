"""Writes src/locales/ui-<lang>.json: the interface strings of the 0.2.10 interface changes (branch f-0291-ui).
Run from apps/host:  python scripts/make-ui-locales.py
Every entry names all nine languages; the keys are ui01.. in the order of S. ui-en.json is the English source. Placeholders ({0}, {1}) must be
kept in every language. A separate file keeps this work from colliding with the other catalogs (the same reason tune-, tools-, fix- and gh- exist)."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr"]
S = []


def add(**kw):
    assert sorted(kw) == sorted(LANGS), sorted(set(LANGS) ^ set(kw))
    S.append(kw)


# #4725 the approval stays in view
add(en="Waiting for your answer",
    zh="等待你的回答", ja="あなたの返答を待っています", de="Wartet auf deine Antwort", es="Esperando tu respuesta",
    ru="Ждёт вашего ответа", pt="Aguardando a sua resposta", ko="답변을 기다리는 중", fr="En attente de votre réponse")
add(en="Waiting for your answer below",
    zh="等待你在下方回答", ja="下で返答をお待ちしています", de="Wartet unten auf deine Antwort", es="Esperando tu respuesta abajo",
    ru="Ждёт вашего ответа ниже", pt="Aguardando a sua resposta abaixo", ko="아래에서 답변을 기다리는 중", fr="En attente de votre réponse ci-dessous")

# #4804 why Start tuning is greyed out
add(en="Strata is still checking this model for saved tunings. Start tuning works as soon as that is done.",
    zh="Strata 仍在检查此模型是否有已保存的调优。检查完成后即可开始调优。",
    ja="Strata がこのモデルの保存済みチューニングを確認しています。確認が終わりしだい、チューニングを開始できます。",
    de="Strata prüft dieses Modell noch auf gespeicherte Abstimmungen. Sobald das fertig ist, kannst du die Abstimmung starten.",
    es="Strata todavía está comprobando si este modelo tiene ajustes guardados. Podrás empezar el ajuste en cuanto termine.",
    ru="Strata ещё проверяет, есть ли у этой модели сохранённые настройки. Как только проверка закончится, можно будет запустить настройку.",
    pt="O Strata ainda está verificando se este modelo tem ajustes salvos. Assim que terminar, você poderá iniciar o ajuste.",
    ko="Strata가 이 모델의 저장된 최적화를 확인하는 중입니다. 확인이 끝나면 바로 최적화를 시작할 수 있습니다.",
    fr="Strata vérifie encore si ce modèle a des optimisations enregistrées. Dès que c’est fini, vous pourrez lancer l’optimisation.")
add(en="Strata could not read the tuning details for this model, so tuning cannot start. Close this window and open it again.",
    zh="Strata 无法读取此模型的调优信息，因此无法开始调优。请关闭此窗口后重新打开。",
    ja="Strata がこのモデルのチューニング情報を読み取れなかったため、チューニングを開始できません。このウィンドウを閉じて、もう一度開いてください。",
    de="Strata konnte die Abstimmungsdaten dieses Modells nicht lesen, deshalb lässt sich die Abstimmung nicht starten. Schließe dieses Fenster und öffne es erneut.",
    es="Strata no pudo leer los datos de ajuste de este modelo, así que el ajuste no puede empezar. Cierra esta ventana y ábrela de nuevo.",
    ru="Strata не смогла прочитать данные настройки этой модели, поэтому настройку нельзя запустить. Закройте это окно и откройте его снова.",
    pt="O Strata não conseguiu ler os dados de ajuste deste modelo, então o ajuste não pode começar. Feche esta janela e abra-a novamente.",
    ko="Strata가 이 모델의 최적화 정보를 읽지 못해 최적화를 시작할 수 없습니다. 이 창을 닫았다가 다시 여세요.",
    fr="Strata n’a pas pu lire les données d’optimisation de ce modèle, l’optimisation ne peut donc pas démarrer. Fermez cette fenêtre et rouvrez-la.")
add(en="Checking for saved tunings\u2026",
    zh="正在检查已保存的调优\u2026", ja="保存済みチューニングを確認しています\u2026", de="Gespeicherte Abstimmungen werden geprüft\u2026", es="Comprobando los ajustes guardados\u2026",
    ru="Проверяем сохранённые настройки\u2026", pt="Verificando os ajustes salvos\u2026", ko="저장된 최적화를 확인하는 중\u2026", fr="Vérification des optimisations enregistrées\u2026")

# #4802 the progress card
add(en="Still working for {0}. You don't need to do anything.",
    zh="仍在处理，已用时 {0}。你无需做任何操作。",
    ja="処理中です（{0}経過）。操作は必要ありません。",
    de="Läuft noch seit {0}. Du musst nichts tun.",
    es="Sigue trabajando desde hace {0}. No tienes que hacer nada.",
    ru="Всё ещё работает: {0}. От вас ничего не требуется.",
    pt="Ainda trabalhando há {0}. Você não precisa fazer nada.",
    ko="{0}째 작업 중입니다. 따로 할 일은 없습니다.",
    fr="Toujours en cours depuis {0}. Vous n'avez rien à faire.")
add(en="The buttons below wait until this finishes.",
    zh="下方的按钮会在此操作完成后恢复可用。",
    ja="この処理が終わるまで、下のボタンは使えません。",
    de="Die Schaltflächen unten warten, bis das fertig ist.",
    es="Los botones de abajo esperan a que esto termine.",
    ru="Кнопки ниже подождут, пока это не закончится.",
    pt="Os botões abaixo esperam até isto terminar.",
    ko="아래 버튼은 이 작업이 끝날 때까지 기다립니다.",
    fr="Les boutons ci-dessous attendent la fin de cette opération.")

# #4798 two addresses
add(en="API base URL on this computer",
    zh="此电脑上的 API 基础地址", ja="このパソコンでの API ベース URL", de="API-Basis-URL auf diesem Computer", es="URL base de la API en este equipo",
    ru="Базовый URL API на этом компьютере", pt="URL base da API neste computador", ko="이 컴퓨터의 API 기본 URL", fr="URL de base de l'API sur cet ordinateur")
add(en="Address for other devices",
    zh="其他设备使用的地址", ja="他のデバイス用のアドレス", de="Adresse für andere Geräte", es="Dirección para otros dispositivos",
    ru="Адрес для других устройств", pt="Endereço para outros dispositivos", ko="다른 기기용 주소", fr="Adresse pour les autres appareils")
add(en="Other devices use this address. It has its own port, separate from the one Strata's window uses, because only the model's API is shared on the network, not the app.",
    zh="其他设备使用此地址。它有自己的端口，与 Strata 窗口使用的端口分开，因为网络上只共享模型的 API，而不是应用本身。",
    ja="他のデバイスはこのアドレスを使います。ネットワークで共有されるのはモデルの API だけでアプリ本体ではないため、Strata のウィンドウが使うポートとは別のポートを使います。",
    de="Andere Geräte verwenden diese Adresse. Sie hat einen eigenen Port, getrennt von dem des Strata-Fensters, weil im Netzwerk nur die API des Modells freigegeben wird, nicht die App.",
    es="Los demás dispositivos usan esta dirección. Tiene su propio puerto, distinto del que usa la ventana de Strata, porque en la red solo se comparte la API del modelo, no la aplicación.",
    ru="Другие устройства используют этот адрес. У него свой порт, отдельный от порта окна Strata, потому что в сети доступен только API модели, а не само приложение.",
    pt="Os outros dispositivos usam este endereço. Ele tem a própria porta, separada da que a janela do Strata usa, porque só a API do modelo é compartilhada na rede, não o aplicativo.",
    ko="다른 기기는 이 주소를 사용합니다. 네트워크에는 앱이 아니라 모델의 API만 공유되므로, Strata 창이 쓰는 포트와는 별도의 포트를 씁니다.",
    fr="Les autres appareils utilisent cette adresse. Elle a son propre port, distinct de celui de la fenêtre de Strata, car seule l'API du modèle est partagée sur le réseau, pas l'application.")

# #3723 Context memory and the out-of-graphics-memory reason
add(en="INT8 is the standard: good quality and a medium amount of memory.",
    zh="INT8 是标准选项：质量好，内存占用适中。",
    ja="INT8 は標準の設定です。品質が良く、メモリ使用量は中程度です。",
    de="INT8 ist der Standard: gute Qualität und mittlerer Speicherbedarf.",
    es="INT8 es el estándar: buena calidad y un uso de memoria medio.",
    ru="INT8 — стандартный вариант: хорошее качество и средний расход памяти.",
    pt="INT8 é o padrão: boa qualidade e uso médio de memória.",
    ko="INT8은 표준 설정입니다. 품질이 좋고 메모리 사용량은 보통입니다.",
    fr="INT8 est le réglage standard : bonne qualité et une consommation de mémoire moyenne.")
add(en="Q4 uses about half the memory of INT8, but is less precise on very long documents.",
    zh="Q4 占用的内存约为 INT8 的一半，但在很长的文档上精度较低。",
    ja="Q4 は INT8 の約半分のメモリで済みますが、非常に長い文書では精度が少し下がります。",
    de="Q4 braucht etwa halb so viel Speicher wie INT8, ist aber bei sehr langen Dokumenten weniger genau.",
    es="Q4 usa aproximadamente la mitad de memoria que INT8, pero es menos preciso con documentos muy largos.",
    ru="Q4 занимает примерно вдвое меньше памяти, чем INT8, но менее точен на очень длинных документах.",
    pt="Q4 usa cerca de metade da memória do INT8, mas é menos preciso em documentos muito longos.",
    ko="Q4는 INT8의 약 절반 메모리를 쓰지만, 아주 긴 문서에서는 정밀도가 떨어집니다.",
    fr="Q4 utilise environ moitié moins de mémoire que INT8, mais il est moins précis sur de très longs documents.")
add(en="K8 / V4 uses about a quarter less memory than INT8 and is more precise than Q4. Prompts are 2 to 5% slower.",
    zh="K8 / V4 比 INT8 约少占用四分之一的内存，精度高于 Q4。处理提示词会慢 2% 到 5%。",
    ja="K8 / V4 は INT8 より約4分の1少ないメモリで済み、Q4 より精度が高い設定です。プロンプトの処理は 2〜5% 遅くなります。",
    de="K8 / V4 braucht etwa ein Viertel weniger Speicher als INT8 und ist genauer als Q4. Prompts sind 2 bis 5 % langsamer.",
    es="K8 / V4 usa aproximadamente una cuarta parte menos de memoria que INT8 y es más preciso que Q4. Los prompts son entre un 2 y un 5 % más lentos.",
    ru="K8 / V4 занимает примерно на четверть меньше памяти, чем INT8, и точнее, чем Q4. Запросы обрабатываются на 2–5 % медленнее.",
    pt="K8 / V4 usa cerca de um quarto a menos de memória que o INT8 e é mais preciso que o Q4. Os prompts ficam de 2 a 5 % mais lentos.",
    ko="K8 / V4는 INT8보다 메모리를 약 4분의 1 적게 쓰고 Q4보다 정밀합니다. 프롬프트 처리는 2~5% 느려집니다.",
    fr="K8 / V4 utilise environ un quart de mémoire en moins que INT8 et est plus précis que Q4. Les prompts sont 2 à 5 % plus lents.")
add(en="The model did not fit in your graphics card’s memory.",
    zh="模型放不进你的显卡显存。",
    ja="モデルがグラフィックスカードのメモリに収まりませんでした。",
    de="Das Modell hat nicht in den Speicher deiner Grafikkarte gepasst.",
    es="El modelo no cabía en la memoria de tu tarjeta gráfica.",
    ru="Модель не поместилась в память вашей видеокарты.",
    pt="O modelo não coube na memória da sua placa de vídeo.",
    ko="모델이 그래픽 카드 메모리에 들어가지 않았습니다.",
    fr="Le modèle ne tenait pas dans la mémoire de votre carte graphique.")
add(en="Lower the conversation context, or set Context memory to Q4 in Expert settings, then try again.",
    zh="请调低对话上下文，或在专家设置中把“上下文内存”设为 Q4，然后重试。",
    ja="会話のコンテキストを小さくするか、上級設定で「コンテキストメモリー」を Q4 にして、もう一度お試しください。",
    de="Verkleinere den Gesprächskontext oder stelle in den Experteneinstellungen „Kontextspeicher“ auf Q4 und versuche es erneut.",
    es="Reduce el contexto de la conversación o pon «Memoria de contexto» en Q4 en Ajustes expertos, y vuelve a intentarlo.",
    ru="Уменьшите контекст разговора или установите «Память контекста» в Q4 в разделе «Экспертные настройки» и повторите попытку.",
    pt="Reduza o contexto da conversa ou defina «Memória de contexto» como Q4 em Configurações avançadas e tente de novo.",
    ko="대화 컨텍스트를 줄이거나 전문가 설정에서 ‘컨텍스트 메모리’를 Q4로 바꾼 뒤 다시 시도하세요.",
    fr="Réduisez le contexte de la conversation, ou réglez « Mémoire de contexte » sur Q4 dans Réglages experts, puis réessayez.")

# #4749 a message for a helper
add(en="Message this helper",
    zh="给这个助手发消息", ja="このヘルパーにメッセージを送る", de="Diesem Helfer schreiben", es="Escribir a este ayudante",
    ru="Написать этому помощнику", pt="Enviar mensagem a este ajudante", ko="이 도우미에게 메시지 보내기", fr="Écrire à cet assistant")
add(en="Message to {0}",
    zh="发给 {0} 的消息", ja="{0} へのメッセージ", de="Nachricht an {0}", es="Mensaje para {0}",
    ru="Сообщение для {0}", pt="Mensagem para {0}", ko="{0}에게 보낼 메시지", fr="Message pour {0}")
add(en="Write to {0}\u2026",
    zh="写给 {0}\u2026", ja="{0} に書く\u2026", de="An {0} schreiben\u2026", es="Escribe a {0}\u2026",
    ru="Напишите {0}\u2026", pt="Escreva para {0}\u2026", ko="{0}에게 쓰기\u2026", fr="Écrire à {0}\u2026")
add(en="Sent to {0}.",
    zh="已发给 {0}。", ja="{0} に送りました。", de="An {0} gesendet.", es="Enviado a {0}.",
    ru="Отправлено: {0}.", pt="Enviado para {0}.", ko="{0}에게 보냈습니다.", fr="Envoyé à {0}.")

# #4783 a message while an answer is being written
add(en="While an answer is being written",
    zh="回答生成期间", ja="回答の作成中", de="Während eine Antwort geschrieben wird", es="Mientras se escribe una respuesta",
    ru="Пока пишется ответ", pt="Enquanto uma resposta é escrita", ko="답변을 작성하는 동안", fr="Pendant qu'une réponse s'écrit")
add(en="What Enter does when you send a message while the assistant is still answering.",
    zh="助手仍在回答时，你发送消息后按回车会发生什么。", ja="アシスタントがまだ回答している間にメッセージを送ったとき、Enter で何が起こるかを選びます。",
    de="Was die Eingabetaste bewirkt, wenn du eine Nachricht sendest, während der Assistent noch antwortet.",
    es="Qué hace Intro cuando envías un mensaje mientras el asistente todavía está respondiendo.",
    ru="Что делает Enter, когда вы отправляете сообщение, пока ассистент ещё отвечает.",
    pt="O que o Enter faz quando você envia uma mensagem enquanto o assistente ainda está respondendo.",
    ko="어시스턴트가 아직 답하는 중에 메시지를 보낼 때 Enter가 하는 일입니다.",
    fr="Ce que fait Entrée quand vous envoyez un message pendant que l'assistant répond encore.")
add(en="Wait for the answer",
    zh="等回答完成", ja="回答が終わるまで待つ", de="Auf die Antwort warten", es="Esperar la respuesta",
    ru="Дождаться ответа", pt="Esperar a resposta", ko="답변이 끝날 때까지 기다리기", fr="Attendre la réponse")
add(en="Send after the answer",
    zh="回答完成后再发送", ja="回答が終わってから送る", de="Nach der Antwort senden", es="Enviar después de la respuesta",
    ru="Отправить после ответа", pt="Enviar depois da resposta", ko="답변이 끝난 뒤 보내기", fr="Envoyer après la réponse")
add(en="Interrupt the answer",
    zh="打断回答", ja="回答を中断する", de="Die Antwort unterbrechen", es="Interrumpir la respuesta",
    ru="Прервать ответ", pt="Interromper a resposta", ko="답변 중단하기", fr="Interrompre la réponse")
add(en="Enter sends it when the answer is done.",
    zh="回答完成后，按回车即可发送。", ja="回答が終わったら Enter で送信されます。", de="Die Eingabetaste sendet sie, sobald die Antwort fertig ist.",
    es="Intro lo envía cuando termina la respuesta.", ru="Enter отправит сообщение, когда ответ закончится.",
    pt="O Enter envia quando a resposta terminar.", ko="답변이 끝나면 Enter로 보냅니다.", fr="Entrée l'envoie quand la réponse est terminée.")
add(en="Enter stops the answer here and sends it now.",
    zh="按回车会在此处停止回答并立即发送。", ja="Enter で回答をここで止めて、すぐに送信します。", de="Die Eingabetaste stoppt die Antwort hier und sendet die Nachricht sofort.",
    es="Intro detiene la respuesta aquí y lo envía ahora.", ru="Enter остановит ответ здесь и сразу отправит сообщение.",
    pt="O Enter para a resposta aqui e envia agora.", ko="Enter를 누르면 여기서 답변을 멈추고 바로 보냅니다.", fr="Entrée arrête la réponse ici et l'envoie tout de suite.")
add(en="Waiting to send",
    zh="等待发送", ja="送信待ち", de="Wartet auf das Senden", es="Esperando para enviar",
    ru="Ждёт отправки", pt="Aguardando envio", ko="보내기 대기 중", fr="En attente d'envoi")
add(en="Take back",
    zh="撤回", ja="取り消す", de="Zurücknehmen", es="Retirar",
    ru="Забрать обратно", pt="Desfazer", ko="되돌리기", fr="Reprendre")

# #4785 quoting
add(en="Quote",
    zh="引用", ja="引用", de="Zitieren", es="Citar",
    ru="Цитировать", pt="Citar", ko="인용", fr="Citer")
add(en="Quoting the assistant",
    zh="引用助手的话", ja="アシスタントの発言を引用", de="Zitat des Assistenten", es="Citando al asistente",
    ru="Цитата ассистента", pt="Citando o assistente", ko="어시스턴트의 말을 인용", fr="Citation de l'assistant")
add(en="Quoting you",
    zh="引用你的话", ja="あなたの発言を引用", de="Zitat von dir", es="Citándote a ti",
    ru="Ваша цитата", pt="Citando você", ko="내 말을 인용", fr="Citation de vous")
add(en="Remove this quote",
    zh="移除这条引用", ja="この引用を外す", de="Dieses Zitat entfernen", es="Quitar esta cita",
    ru="Убрать эту цитату", pt="Remover esta citação", ko="이 인용 제거", fr="Retirer cette citation")
add(en="Go to the quoted message",
    zh="跳到被引用的消息", ja="引用元のメッセージへ移動", de="Zur zitierten Nachricht springen", es="Ir al mensaje citado",
    ru="Перейти к цитируемому сообщению", pt="Ir para a mensagem citada", ko="인용한 메시지로 이동", fr="Aller au message cité")
add(en="Quoted",
    zh="引用", ja="引用", de="Zitiert", es="Citado",
    ru="Цитата", pt="Citado", ko="인용됨", fr="Cité")
add(en="Show or hide the rest of the quote",
    zh="显示或隐藏引用的其余部分", ja="引用の続きを表示または非表示", de="Rest des Zitats ein- oder ausblenden", es="Mostrar u ocultar el resto de la cita",
    ru="Показать или скрыть остальную часть цитаты", pt="Mostrar ou ocultar o resto da citação", ko="인용의 나머지 부분 보기 또는 숨기기", fr="Afficher ou masquer la suite de la citation")

# #4244 the agent's own questions
add(en="Skip",
    zh="跳过", ja="スキップ", de="Überspringen", es="Omitir",
    ru="Пропустить", pt="Pular", ko="건너뛰기", fr="Passer")
add(en="The assistant has a question for you",
    zh="助手有一个问题要问你", ja="アシスタントからの質問があります", de="Der Assistent hat eine Frage an dich", es="El asistente tiene una pregunta para ti",
    ru="У ассистента есть вопрос к вам", pt="O assistente tem uma pergunta para você", ko="어시스턴트가 질문이 있습니다", fr="L'assistant a une question pour vous")
add(en="Type your own answer",
    zh="输入你自己的回答", ja="自分の答えを入力", de="Eigene Antwort eingeben", es="Escribe tu propia respuesta",
    ru="Введите свой ответ", pt="Digite a sua própria resposta", ko="직접 답변 입력", fr="Saisissez votre propre réponse")
add(en="Send answer",
    zh="发送回答", ja="回答を送る", de="Antwort senden", es="Enviar respuesta",
    ru="Отправить ответ", pt="Enviar resposta", ko="답변 보내기", fr="Envoyer la réponse")
add(en="Questions for you",
    zh="向你提问", ja="あなたへの質問", de="Fragen an dich", es="Preguntas para ti",
    ru="Вопросы к вам", pt="Perguntas para você", ko="내게 하는 질문", fr="Questions pour vous")
add(en="Lets the assistant stop and ask you a question, with answers to pick from, when it needs your decision.",
    zh="当助手需要你做决定时，可以暂停并向你提问，并给出可选的答案。", ja="アシスタントがあなたの判断を必要とするとき、選択肢つきで質問できるようにします。",
    de="Der Assistent kann anhalten und dir eine Frage mit Antworten zur Auswahl stellen, wenn er deine Entscheidung braucht.",
    es="Permite que el asistente se detenga y te haga una pregunta, con respuestas para elegir, cuando necesita tu decisión.",
    ru="Позволяет ассистенту остановиться и задать вам вопрос с вариантами ответа, когда ему нужно ваше решение.",
    pt="Permite que o assistente pare e faça uma pergunta, com respostas para escolher, quando precisa da sua decisão.",
    ko="어시스턴트가 결정이 필요할 때 멈추고 고를 수 있는 답과 함께 질문할 수 있게 합니다.",
    fr="Permet à l'assistant de s'arrêter et de vous poser une question, avec des réponses à choisir, quand il a besoin de votre décision.")

here = Path(__file__).resolve().parent.parent / "src" / "locales"
NBSP = " "


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
        data[f"ui{i + 1:02d}"] = french(text) if lang == "fr" else text
    (here / f"ui-{lang}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("written", len(S), "strings in", len(LANGS), "languages")
