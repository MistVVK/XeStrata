"""Writes src/locales/df2-<lang>.json: the strings of the Drafting choice that came with the DFlash2 option (branch f-0291-df2-option): the plain-words
paragraph under the dropdown, the three lines about the person's graphics card, the "not on this card" and "no internet" lines, and the SSD line of the
This computer panel.
Run from apps/host:  python scripts/make-df2-locales.py
Every entry names all nine languages; the keys are df01.. in the order of S. df2-en.json is the English source. Placeholders ({0}) must be kept in every
language. A separate file keeps this work from colliding with the other catalogs (the same reason pick-, tune-, tools-, fix-, gh-, ui- and engine- exist).
The English lines of df03-df07 and df09 are also written in strata_control/drafting.py and strata_control/drafting_fetch.py; tests/drafting.test.mjs keeps the two the same."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr", "he"]
S = []


def add(en, zh, ja, de, es, ru, pt, ko, fr, he):
    S.append(dict(en=en, zh=zh, ja=ja, de=de, es=es, ru=ru, pt=pt, ko=ko, fr=fr, he=he))


add("MTP is the default. DFlash2 is another way to guess the next words ahead, for Qwen3.8-Flash-Next only. On an RTX PRO 5000 it was 7.8% faster than MTP "
    "on 19 short prompts and 1.1% faster on long thinking, and faster on 21 of 23 prompts. On the short prompts the answers were identical.",
    "默认使用 MTP。DFlash2 是另一种提前猜测后续词语的方式，仅适用于 Qwen3.8-Flash-Next。在 RTX PRO 5000 上，它在 19 个短提示上比 MTP 快 7.8%，在长篇思考上快 1.1%，"
    "并且在 23 个提示中有 21 个更快。在短提示上，回答完全相同。",
    "既定では MTP を使います。DFlash2 は次の語を先読みするもう一つの方法で、Qwen3.8-Flash-Next 専用です。RTX PRO 5000 では、短いプロンプト 19 件で MTP より 7.8% 速く、"
    "長い思考では 1.1% 速く、23 件中 21 件で速くなりました。短いプロンプトでは回答が同一でした。",
    "MTP ist die Voreinstellung. DFlash2 ist eine andere Art, die nächsten Wörter vorauszuraten, nur für Qwen3.8-Flash-Next. Auf einer RTX PRO 5000 war es bei 19 kurzen "
    "Prompts 7,8 % schneller als MTP, bei langem Nachdenken 1,1 % schneller und bei 21 von 23 Prompts schneller. Bei den kurzen Prompts waren die Antworten identisch.",
    "MTP es la opción predeterminada. DFlash2 es otra forma de adivinar las siguientes palabras por adelantado, solo para Qwen3.8-Flash-Next. En una RTX PRO 5000 fue un 7,8 % "
    "más rápido que MTP en 19 prompts cortos y un 1,1 % más rápido en razonamientos largos, y más rápido en 21 de 23 prompts. En los prompts cortos las respuestas fueron idénticas.",
    "По умолчанию используется MTP. DFlash2 — другой способ угадывать следующие слова наперёд, только для Qwen3.8-Flash-Next. На RTX PRO 5000 он был на 7,8 % быстрее MTP "
    "на 19 коротких запросах и на 1,1 % быстрее на длинных рассуждениях, а на 21 из 23 запросов — быстрее. На коротких запросах ответы были идентичны.",
    "MTP é o padrão. DFlash2 é outra forma de adivinhar as próximas palavras antecipadamente, só para o Qwen3.8-Flash-Next. Numa RTX PRO 5000 foi 7,8 % mais rápido que o MTP "
    "em 19 prompts curtos e 1,1 % mais rápido em raciocínios longos, e mais rápido em 21 de 23 prompts. Nos prompts curtos as respostas foram idênticas.",
    "기본값은 MTP입니다. DFlash2는 다음 단어를 미리 추측하는 또 다른 방식이며 Qwen3.8-Flash-Next 전용입니다. RTX PRO 5000에서 짧은 프롬프트 19개에서는 MTP보다 7.8% 빨랐고, "
    "긴 사고 과정에서는 1.1% 빨랐으며, 23개 중 21개 프롬프트에서 더 빨랐습니다. 짧은 프롬프트에서는 답변이 동일했습니다.",
    "MTP est le réglage par défaut. DFlash2 est une autre façon de deviner à l'avance les mots suivants, uniquement pour Qwen3.8-Flash-Next. Sur une RTX PRO 5000, il a été 7,8 % "
    "plus rapide que MTP sur 19 prompts courts et 1,1 % plus rapide en réflexion longue, et plus rapide sur 21 prompts sur 23. Sur les prompts courts, les réponses étaient identiques.",
    he="MTP הוא ברירת המחדל. DFlash2 היא דרך נוספת לנחש מראש את המילים הבאות, ל-Qwen3.8-Flash-Next בלבד. ב-RTX PRO 5000 הוא היה מהיר ב-7.8% מ-MTP ב-19 הנחיות קצרות ומהיר ב-1.1% בחשיבה ארוכה, ומהיר יותר ב-21 מתוך 23 הנחיות. בהנחיות הקצרות התשובות היו זהות.")
add("It downloads about 1.2 GB once, works with one graphics card and one request at a time, and uses MTP otherwise.",
    "它会一次性下载约 1.2 GB，仅在使用一张显卡且一次处理一个请求时有效，其他情况下使用 MTP。",
    "約 1.2 GB を一度だけダウンロードします。グラフィックスカード 1 枚で一度に 1 件のリクエストを処理するときに動作し、それ以外では MTP を使います。",
    "Es lädt einmalig etwa 1,2 GB herunter, funktioniert mit einer Grafikkarte und einer Anfrage zur Zeit und nutzt sonst MTP.",
    "Descarga unos 1,2 GB una sola vez, funciona con una tarjeta gráfica y una solicitud a la vez, y en los demás casos usa MTP.",
    "Один раз скачивает около 1,2 ГБ, работает с одной видеокартой и одним запросом за раз, в остальных случаях используется MTP.",
    "Baixa cerca de 1,2 GB uma única vez, funciona com uma placa de vídeo e um pedido de cada vez, e nos outros casos usa o MTP.",
    "약 1.2 GB를 한 번만 내려받으며, 그래픽 카드 한 장과 한 번에 하나의 요청에서 작동하고, 그 외에는 MTP를 사용합니다.",
    "Il télécharge environ 1,2 Go une seule fois, fonctionne avec une carte graphique et une requête à la fois, et utilise MTP sinon.",
    he="הוא מוריד כ-1.2 GB פעם אחת, עובד עם כרטיס מסך אחד ובקשה אחת בכל פעם, ובמקרים אחרים משתמש ב-MTP.")
add("On graphics cards like yours, DFlash2 was faster than MTP on some kinds of text and slower on others in our tests.",
    "在与您类似的显卡上，我们的测试中 DFlash2 在某些类型的文本上比 MTP 快，在另一些上更慢。",
    "お使いのようなグラフィックスカードでは、私たちのテストで DFlash2 は MTP より速い種類の文章と遅い種類の文章がありました。",
    "Auf Grafikkarten wie Ihrer war DFlash2 in unseren Tests bei manchen Textarten schneller als MTP und bei anderen langsamer.",
    "En tarjetas gráficas como la suya, en nuestras pruebas DFlash2 fue más rápido que MTP con algunos tipos de texto y más lento con otros.",
    "На видеокартах, подобных вашей, в наших тестах DFlash2 был быстрее MTP на одних видах текста и медленнее на других.",
    "Em placas de vídeo como a sua, nos nossos testes o DFlash2 foi mais rápido que o MTP em alguns tipos de texto e mais lento em outros.",
    "고객님과 비슷한 그래픽 카드에서 저희 테스트 결과 DFlash2는 일부 유형의 글에서는 MTP보다 빨랐고 다른 유형에서는 더 느렸습니다.",
    "Sur des cartes graphiques comme la vôtre, DFlash2 a été plus rapide que MTP sur certains types de texte et plus lent sur d'autres lors de nos tests.",
    he="בבדיקות שלנו, בכרטיסי מסך כמו שלכם, DFlash2 היה מהיר יותר מ-MTP בסוגי טקסט מסוימים ואיטי יותר באחרים.")
add("On graphics cards like yours, DFlash2 was slower than MTP in our tests.",
    "在与您类似的显卡上，我们的测试中 DFlash2 比 MTP 更慢。",
    "お使いのようなグラフィックスカードでは、私たちのテストで DFlash2 は MTP より遅くなりました。",
    "Auf Grafikkarten wie Ihrer war DFlash2 in unseren Tests langsamer als MTP.",
    "En tarjetas gráficas como la suya, en nuestras pruebas DFlash2 fue más lento que MTP.",
    "На видеокартах, подобных вашей, в наших тестах DFlash2 был медленнее MTP.",
    "Em placas de vídeo como a sua, nos nossos testes o DFlash2 foi mais lento que o MTP.",
    "고객님과 비슷한 그래픽 카드에서 저희 테스트 결과 DFlash2는 MTP보다 느렸습니다.",
    "Sur des cartes graphiques comme la vôtre, DFlash2 a été plus lent que MTP lors de nos tests.",
    he="בבדיקות שלנו, בכרטיסי מסך כמו שלכם, DFlash2 היה איטי יותר מ-MTP.")
add("DFlash2 has not been tested on your graphics card yet.",
    "DFlash2 尚未在您的显卡上测试过。",
    "DFlash2 はまだお使いのグラフィックスカードでテストされていません。",
    "DFlash2 wurde auf Ihrer Grafikkarte noch nicht getestet.",
    "DFlash2 todavía no se ha probado en su tarjeta gráfica.",
    "DFlash2 ещё не проверялся на вашей видеокарте.",
    "O DFlash2 ainda não foi testado na sua placa de vídeo.",
    "DFlash2는 아직 고객님의 그래픽 카드에서 테스트되지 않았습니다.",
    "DFlash2 n'a pas encore été testé sur votre carte graphique.",
    he="DFlash2 עדיין לא נבדק על כרטיס המסך שלכם.")
add("Drafting: MTP (DFlash2 isn't available on this graphics card)",
    "预测方式：MTP（DFlash2 不适用于这张显卡）",
    "先読み：MTP（DFlash2 はこのグラフィックスカードでは使えません）",
    "Vorausraten: MTP (DFlash2 ist auf dieser Grafikkarte nicht verfügbar)",
    "Predicción: MTP (DFlash2 no está disponible en esta tarjeta gráfica)",
    "Упреждающая генерация: MTP (DFlash2 недоступен на этой видеокарте)",
    "Previsão: MTP (o DFlash2 não está disponível nesta placa de vídeo)",
    "미리 추측: MTP (이 그래픽 카드에서는 DFlash2를 사용할 수 없습니다)",
    "Anticipation : MTP (DFlash2 n'est pas disponible sur cette carte graphique)",
    he="טיוטה: MTP (DFlash2 אינו זמין בכרטיס המסך הזה)")
add("There is no internet connection, so the DFlash2 drafter can't be downloaded now.",
    "没有网络连接，因此现在无法下载 DFlash2 草稿模型。",
    "インターネットに接続されていないため、今は DFlash2 のドラフターをダウンロードできません。",
    "Es besteht keine Internetverbindung, deshalb kann der DFlash2-Drafter jetzt nicht heruntergeladen werden.",
    "No hay conexión a internet, así que ahora no se puede descargar el borrador de DFlash2.",
    "Нет подключения к интернету, поэтому сейчас нельзя скачать черновую модель DFlash2.",
    "Não há ligação à internet, por isso não é possível transferir agora o rascunhador do DFlash2.",
    "인터넷에 연결되어 있지 않아 지금은 DFlash2 드래프터를 내려받을 수 없습니다.",
    "Il n'y a pas de connexion internet, le drafter DFlash2 ne peut donc pas être téléchargé pour le moment.",
    he="אין חיבור לאינטרנט, ולכן אי אפשר להוריד עכשיו את מנוע הטיוטה של DFlash2.")
add("SSD: not needed",
    "SSD：不需要", "SSD：不要", "SSD: nicht nötig", "SSD: no hace falta", "SSD: не нужен", "SSD: não é necessário", "SSD: 필요 없음", "SSD : pas nécessaire",
    he="SSD: לא נדרש")

add("DFlash2 answers one request at a time, so Strata set this model to one request at a time.",
    "DFlash2 一次只处理一个请求，因此 Strata 将此模型设置为一次一个请求。",
    "DFlash2 は一度に 1 件のリクエストしか処理しないため、Strata はこのモデルを一度に 1 件に設定しました。",
    "DFlash2 beantwortet eine Anfrage nach der anderen, deshalb hat Strata dieses Modell auf eine Anfrage zur Zeit gestellt.",
    "DFlash2 responde una solicitud a la vez, así que Strata configuró este modelo para una solicitud a la vez.",
    "DFlash2 отвечает на один запрос за раз, поэтому Strata настроила эту модель на один запрос за раз.",
    "O DFlash2 responde a um pedido de cada vez, por isso o Strata configurou este modelo para um pedido de cada vez.",
    "DFlash2는 한 번에 하나의 요청만 처리하므로 Strata가 이 모델을 한 번에 하나의 요청으로 설정했습니다.",
    "DFlash2 répond à une requête à la fois, Strata a donc réglé ce modèle sur une requête à la fois.",
    he="DFlash2 עונה לבקשה אחת בכל פעם, ולכן Strata הגדירה את המודל הזה לבקשה אחת בכל פעם.")

add("Drafting: off (no drafter is downloaded yet)",
    "预测草稿：关闭（尚未下载任何草稿模型）",
    "先読み：オフ（ドラフトモデルがまだダウンロードされていません）",
    "Vorausraten: aus (noch kein Entwurfsbaustein heruntergeladen)",
    "Predicción: desactivada (aún no hay ningún predictor descargado)",
    "Предсказание: выключено (черновая модель ещё не скачана)",
    "Previsão: desativada (nenhum preditor foi baixado ainda)",
    "초안 예측: 꺼짐(초안 모델이 아직 내려받아지지 않았습니다)",
    "Anticipation : désactivée (aucun module d'anticipation n'est encore téléchargé)",
    he="טיוטה: כבויה (עדיין לא הורד מנוע טיוטה)")
add("Drafting: DFlash2 (the MTP drafter isn't downloaded yet)",
    "预测草稿：DFlash2（MTP 草稿模型尚未下载）",
    "先読み：DFlash2（MTP のドラフトモデルはまだダウンロードされていません）",
    "Vorausraten: DFlash2 (der MTP-Entwurfsbaustein ist noch nicht heruntergeladen)",
    "Predicción: DFlash2 (el predictor de MTP aún no está descargado)",
    "Предсказание: DFlash2 (черновая модель MTP ещё не скачана)",
    "Previsão: DFlash2 (o preditor do MTP ainda não foi baixado)",
    "초안 예측: DFlash2(MTP 초안 모델이 아직 내려받아지지 않았습니다)",
    "Anticipation : DFlash2 (le module d'anticipation MTP n'est pas encore téléchargé)",
    he="טיוטה: DFlash2 (מנוע הטיוטה של MTP עדיין לא הורד)")
add("Choosing DFlash2 downloads its drafter: about {0} GB, once.",
    "选择 DFlash2 会下载它的草稿模型：约 {0} GB，只下载一次。",
    "DFlash2 を選ぶと、そのドラフトモデルを一度だけダウンロードします（約 {0} GB）。",
    "Wenn Sie DFlash2 wählen, wird sein Entwurfsbaustein heruntergeladen: etwa {0} GB, einmalig.",
    "Elegir DFlash2 descarga su predictor: unos {0} GB, una sola vez.",
    "Если выбрать DFlash2, будет скачана его черновая модель: около {0} ГБ, один раз.",
    "Escolher o DFlash2 baixa o preditor dele: cerca de {0} GB, uma vez.",
    "DFlash2를 고르면 그 초안 모델을 한 번 내려받습니다(약 {0}GB).",
    "Choisir DFlash2 télécharge son module d'anticipation : environ {0} Go, une seule fois.",
    he="בחירה ב-DFlash2 מורידה את מנוע הטיוטה שלו: כ-{0} GB, פעם אחת.")
add("Choosing MTP downloads its drafter: about {0} GB, once.",
    "选择 MTP 会下载它的草稿模型：约 {0} GB，只下载一次。",
    "MTP を選ぶと、そのドラフトモデルを一度だけダウンロードします（約 {0} GB）。",
    "Wenn Sie MTP wählen, wird sein Entwurfsbaustein heruntergeladen: etwa {0} GB, einmalig.",
    "Elegir MTP descarga su predictor: unos {0} GB, una sola vez.",
    "Если выбрать MTP, будет скачана его черновая модель: около {0} ГБ, один раз.",
    "Escolher o MTP baixa o preditor dele: cerca de {0} GB, uma vez.",
    "MTP를 고르면 그 초안 모델을 한 번 내려받습니다(약 {0}GB).",
    "Choisir MTP télécharge son module d'anticipation : environ {0} Go, une seule fois.",
    he="בחירה ב-MTP מורידה את מנוע הטיוטה שלו: כ-{0} GB, פעם אחת.")
add("The DFlash2 drafter could not be downloaded, so the model starts without a drafter. Loading the model…",
    "无法下载 DFlash2 草稿模型，因此模型将不使用草稿模型启动。正在加载模型…",
    "DFlash2 のドラフトモデルをダウンロードできなかったため、ドラフトモデルなしで起動します。モデルを読み込んでいます…",
    "Der DFlash2-Entwurfsbaustein konnte nicht heruntergeladen werden, deshalb startet das Modell ohne Entwurfsbaustein. Das Modell wird geladen …",
    "No se pudo descargar el predictor de DFlash2, así que el modelo arranca sin predictor. Cargando el modelo…",
    "Не удалось скачать черновую модель DFlash2, поэтому модель запускается без черновой модели. Загружается модель…",
    "Não foi possível baixar o preditor do DFlash2, então o modelo inicia sem preditor. Carregando o modelo…",
    "DFlash2 초안 모델을 내려받지 못해 초안 모델 없이 시작합니다. 모델을 불러오는 중…",
    "Le module d'anticipation DFlash2 n'a pas pu être téléchargé, le modèle démarre donc sans module d'anticipation. Chargement du modèle…",
    he="לא ניתן היה להוריד את מנוע הטיוטה של DFlash2, ולכן המודל מופעל בלי מנוע טיוטה. טוען את המודל…")
add("Downloading the MTP drafter…",
    "正在下载 MTP 草稿模型…",
    "MTP のドラフトモデルをダウンロードしています…",
    "MTP-Entwurfsbaustein wird heruntergeladen …",
    "Descargando el predictor de MTP…",
    "Скачивается черновая модель MTP…",
    "Baixando o preditor do MTP…",
    "MTP 초안 모델을 내려받는 중…",
    "Téléchargement du module d'anticipation MTP…",
    he="מוריד את מנוע הטיוטה של MTP…")

add("The MTP drafter could not be downloaded.",
    "无法下载 MTP 草稿模型。",
    "MTP のドラフトモデルをダウンロードできませんでした。",
    "Der MTP-Entwurfsbaustein konnte nicht heruntergeladen werden.",
    "No se pudo descargar el predictor de MTP.",
    "Не удалось скачать черновую модель MTP.",
    "Não foi possível baixar o preditor do MTP.",
    "MTP 초안 모델을 내려받지 못했습니다.",
    "Le module d'anticipation MTP n'a pas pu être téléchargé.",
    he="לא ניתן היה להוריד את מנוע הטיוטה של MTP.")
add("Drafting: MTP (DFlash2 didn't start)",
    "预测草稿：MTP（DFlash2 未能启动）",
    "先読み：MTP（DFlash2 は起動しませんでした）",
    "Vorausraten: MTP (DFlash2 ist nicht gestartet)",
    "Predicción: MTP (DFlash2 no arrancó)",
    "Предсказание: MTP (DFlash2 не запустился)",
    "Previsão: MTP (o DFlash2 não iniciou)",
    "초안 예측: MTP(DFlash2가 시작되지 않았습니다)",
    "Anticipation : MTP (DFlash2 n'a pas démarré)",
    he="טיוטה: MTP (DFlash2 לא הופעל)")
add("Auto uses the all-cards split engine with the MTP draft layer: about {0} GB to download, once.",
    "自动模式使用全部显卡的拆分引擎和 MTP 草稿层：约 {0} GB，只下载一次。",
    "自動では、全カードのスプリットエンジンと MTP ドラフト層を使います。約 {0} GB を一度だけダウンロードします。",
    "Automatisch nutzt die Alle-Karten-Split-Engine mit der MTP-Entwurfsschicht: etwa {0} GB, einmalig herunterzuladen.",
    "Automático usa el motor de reparto entre todas las tarjetas con la capa de borrador MTP: unos {0} GB, una sola vez.",
    "Авто использует движок с разделением по всем картам и черновой слой MTP: около {0} ГБ, скачивается один раз.",
    "O automático usa o motor de divisão entre todas as placas com a camada de rascunho MTP: cerca de {0} GB, uma vez.",
    "자동은 모든 카드를 쓰는 분할 엔진과 MTP 초안 층을 사용합니다(약 {0}GB, 한 번만 내려받음).",
    "Auto utilise le moteur réparti sur toutes les cartes avec la couche de brouillon MTP : environ {0} Go à télécharger, une seule fois.",
    he="במצב אוטומטי נעשה שימוש במנוע הפיצול על כל הכרטיסים, עם שכבת הטיוטה של MTP: כ-{0} GB להורדה, פעם אחת.")
add("The all-cards split engine could not start, so the normal layer split runs.",
    "全显卡拆分引擎未能启动，因此改用常规的分层拆分。",
    "全カードのスプリットエンジンを起動できなかったため、通常のレイヤー分割で動作しています。",
    "Die Alle-Karten-Split-Engine konnte nicht starten, deshalb läuft die normale Schichtaufteilung.",
    "El motor de reparto entre todas las tarjetas no pudo arrancar, así que se usa el reparto normal por capas.",
    "Движок с разделением по всем картам не запустился, поэтому работает обычное послойное разделение.",
    "O motor de divisão entre todas as placas não conseguiu iniciar, então a divisão normal por camadas está em uso.",
    "모든 카드를 쓰는 분할 엔진을 시작하지 못해 일반 레이어 분할로 실행합니다.",
    "Le moteur réparti sur toutes les cartes n'a pas pu démarrer, la répartition normale par couches est donc utilisée.",
    he="לא ניתן היה להפעיל את מנוע הפיצול על כל הכרטיסים, ולכן פועל פיצול השכבות הרגיל.")
add("Drafting: off (the cards were short of memory)",
    "预测草稿：关闭（显卡内存不足）",
    "先読み：オフ（グラフィックスカードのメモリが不足していました）",
    "Vorausraten: aus (den Karten fehlte Speicher)",
    "Predicción: desactivada (a las tarjetas les faltaba memoria)",
    "Предсказание: выключено (на картах не хватило памяти)",
    "Previsão: desligada (faltou memória nas placas)",
    "초안 예측: 꺼짐(카드 메모리가 부족했습니다)",
    "Anticipation : désactivée (les cartes manquaient de mémoire)",
    he="טיוטה: כבויה (לא היה מספיק זיכרון בכרטיסים)")
add("On your Mac, DFlash2 was faster than MTP on this model in our tests.",
    "在您的 Mac 上，我们的测试中 DFlash2 在此模型上比 MTP 更快。",
    "お使いの Mac では、私たちのテストでこのモデルにおいて DFlash2 は MTP より速くなりました。",
    "Auf Ihrem Mac war DFlash2 bei diesem Modell in unseren Tests schneller als MTP.",
    "En su Mac, en nuestras pruebas DFlash2 fue más rápido que MTP con este modelo.",
    "На вашем Mac в наших тестах DFlash2 с этой моделью был быстрее MTP.",
    "No seu Mac, nos nossos testes o DFlash2 foi mais rápido que o MTP com este modelo.",
    "고객님의 Mac에서 저희 테스트 결과 이 모델에서는 DFlash2가 MTP보다 빨랐습니다.",
    "Sur votre Mac, DFlash2 a été plus rapide que MTP avec ce modèle lors de nos tests.",
    he="בבדיקות שלנו, ב-Mac שלכם DFlash2 היה מהיר יותר מ-MTP במודל הזה.")
add("DFlash2 is off by default on this graphics card for now; choose it in Drafting to use it.",
    "目前在这张显卡上 DFlash2 默认关闭；如需使用，请在“预测草稿”中选择它。",
    "現在、このグラフィックスカードでは DFlash2 は既定でオフです。使うには「先読み」で選んでください。",
    "DFlash2 ist auf dieser Grafikkarte vorerst standardmäßig aus; wählen Sie es unter „Vorausraten“ aus, um es zu nutzen.",
    "DFlash2 está desactivado de forma predeterminada en esta tarjeta gráfica por ahora; elíjalo en Predicción para usarlo.",
    "На этой видеокарте DFlash2 пока выключен по умолчанию; чтобы использовать его, выберите его в разделе «Предсказание».",
    "O DFlash2 está desligado por predefinição nesta placa de vídeo por agora; escolha-o em Previsão para o usar.",
    "현재 이 그래픽 카드에서는 DFlash2가 기본적으로 꺼져 있습니다. 사용하려면 ‘초안 예측’에서 선택하세요.",
    "DFlash2 est désactivé par défaut sur cette carte graphique pour le moment ; choisissez-le dans Anticipation pour l'utiliser.",
    he="DFlash2 כבוי כברירת מחדל בכרטיס המסך הזה בינתיים; כדי להשתמש בו יש לבחור בו בהגדרת הטיוטה.")
# strata_control/drafting.py NONE_IN_SLOTS after "Drafting: ": request slots with no --batch-mtp draft nothing (the line is printed whole, not inside "MTP (...)")
add("Drafting: none in parallel slots (one token per request and step); a single request still drafts",
    "预测草稿：并行槽位中没有（每个请求每一步一个词元）；单个请求仍会预测草稿",
    "先読み：並列の枠ではなし（リクエストごと、ステップごとに 1 トークン）；リクエストが 1 件だけのときは先読みします",
    "Vorausraten: keines in parallelen Plätzen (ein Token pro Anfrage und Schritt); eine einzelne Anfrage rät weiterhin voraus",
    "Predicción: ninguna en cupos paralelos (un token por solicitud y paso); una sola solicitud sigue prediciendo",
    "Предсказание: нет в параллельных слотах (один токен на запрос за шаг); один запрос по-прежнему предсказывает",
    "Previsão: nenhuma em vagas paralelas (um token por pedido e por passo); um único pedido ainda faz previsão",
    "초안 예측: 병렬 슬롯에서는 없음(요청당 한 단계에 토큰 하나); 요청이 하나뿐이면 초안 예측을 합니다",
    "Anticipation : aucune dans les emplacements parallèles (un token par requête et par étape) ; une requête unique anticipe toujours",
    he="טיוטה: לא בחריצים מקבילים (טוקן אחד לכל בקשה ולכל צעד); בקשה יחידה עדיין משתמשת בטיוטה")
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
        data[f"df{i + 1:02d}"] = french(text) if lang == "fr" else text
    with (here / f"df2-{lang}.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
print("written", len(S), "strings in", len(LANGS), "languages")
