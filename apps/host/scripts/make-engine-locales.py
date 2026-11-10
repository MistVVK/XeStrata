"""Writes src/locales/engine-<lang>.json: the strings of Settings > Engine (the list of installed engines: switch, delete) and the words that now point to
About for updating (branch f-0291-engine-tab).  Run from apps/host:  python scripts/make-engine-locales.py
Every entry names all nine languages; the keys are et01.. in the order of S. engine-en.json is the English source. Placeholders ({0}, {1}) must be kept in
every language. A separate file keeps this work from colliding with the other catalogs (the same reason tune-, tools-, fix-, gh- and ui- exist)."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr"]
S = []


def add(en, zh, ja, de, es, ru, pt, ko, fr):
    S.append(dict(en=en, zh=zh, ja=ja, de=de, es=es, ru=ru, pt=pt, ko=ko, fr=fr))


# ---- the page
add("Installed engines",
    "已安装的引擎", "インストール済みのエンジン", "Installierte Engines", "Motores instalados", "Установленные движки", "Motores instalados", "설치된 엔진", "Moteurs installés")
add("Switch to an older engine to test something, then switch back to the newest. Updating to the latest is in About.",
    "切换到较旧的引擎来测试某些内容，然后再切换回最新的。更新到最新版本请前往“关于”。",
    "古いエンジンに切り替えて試し、終わったら最新のエンジンに戻せます。最新版への更新は「情報」で行います。",
    "Wechsle zu einer älteren Engine, um etwas zu testen, und danach zurück zur neuesten. Das Update auf die neueste Version findest du unter „Über“.",
    "Cambia a un motor anterior para probar algo y vuelve después al más reciente. La actualización a la última versión está en «Acerca de».",
    "Переключитесь на более старый движок, чтобы что-то проверить, а потом вернитесь на самый новый. Обновление до последней версии — в разделе «О программе».",
    "Troque para um motor mais antigo para testar algo e depois volte ao mais recente. A atualização para a versão mais recente fica em «Sobre».",
    "이전 엔진으로 바꿔 테스트한 뒤 다시 최신 엔진으로 돌아올 수 있습니다. 최신 버전으로의 업데이트는 '정보'에서 합니다.",
    "Passez à un moteur plus ancien pour tester quelque chose, puis revenez au plus récent. La mise à jour vers la dernière version se trouve dans « À propos ».")
add("Engines can be switched and deleted in the installed Strata app.",
    "可以在已安装的 Strata 应用中切换和删除引擎。", "エンジンの切り替えと削除は、インストール済みの Strata アプリで行えます。",
    "Engines lassen sich in der installierten Strata-App wechseln und löschen.", "Los motores se pueden cambiar y eliminar en la aplicación Strata instalada.",
    "Переключать и удалять движки можно в установленном приложении Strata.", "Os motores podem ser trocados e excluídos no aplicativo Strata instalado.",
    "엔진 전환과 삭제는 설치된 Strata 앱에서 할 수 있습니다.", "Les moteurs peuvent être changés et supprimés dans l'application Strata installée.")
add("The installed engines couldn't be read. Try again in a moment.",
    "无法读取已安装的引擎。请稍后重试。", "インストール済みのエンジンを読み取れませんでした。しばらくしてからもう一度お試しください。",
    "Die installierten Engines konnten nicht gelesen werden. Versuche es gleich noch einmal.", "No se pudieron leer los motores instalados. Inténtalo de nuevo en un momento.",
    "Не удалось прочитать список установленных движков. Попробуйте ещё раз через минуту.", "Não foi possível ler os motores instalados. Tente novamente em instantes.",
    "설치된 엔진을 읽지 못했습니다. 잠시 후 다시 시도하세요.", "Les moteurs installés n'ont pas pu être lus. Réessayez dans un instant.")
add("No engine is installed on this computer yet. Update in About downloads the right one.",
    "此电脑上还没有安装引擎。在“关于”中点击更新即可下载合适的引擎。", "このコンピューターにはまだエンジンがありません。「情報」で更新すると、合ったものがダウンロードされます。",
    "Auf diesem Computer ist noch keine Engine installiert. Das Update unter „Über“ lädt die passende herunter.",
    "Todavía no hay ningún motor instalado en este equipo. Actualizar en «Acerca de» descarga el adecuado.",
    "На этом компьютере ещё не установлен движок. Обновление в разделе «О программе» скачает подходящий.",
    "Ainda não há nenhum motor instalado neste computador. Atualizar em «Sobre» baixa o adequado.",
    "이 컴퓨터에는 아직 설치된 엔진이 없습니다. '정보'에서 업데이트하면 알맞은 엔진을 내려받습니다.",
    "Aucun moteur n'est encore installé sur cet ordinateur. La mise à jour dans « À propos » télécharge le bon.")
add("Engines take {0} on this computer.",
    "引擎在此电脑上共占用 {0}。", "エンジンはこのコンピューターで {0} を使っています。", "Die Engines belegen auf diesem Computer {0}.", "Los motores ocupan {0} en este equipo.",
    "Движки занимают на этом компьютере {0}.", "Os motores ocupam {0} neste computador.", "엔진이 이 컴퓨터에서 {0}을(를) 차지합니다.", "Les moteurs occupent {0} sur cet ordinateur.")
add("Newest", "最新", "最新", "Neueste", "Más reciente", "Новейший", "Mais recente", "최신", "Le plus récent")
add("Use this", "使用此引擎", "これを使う", "Diese verwenden", "Usar este", "Использовать", "Usar este", "이 엔진 사용", "Utiliser celui-ci")
add("Qwen and other models",
    "Qwen 及其他模型", "Qwen とその他のモデル", "Qwen und andere Modelle", "Qwen y otros modelos", "Qwen и другие модели", "Qwen e outros modelos", "Qwen 및 기타 모델", "Qwen et autres modèles")
add("A running program is using this engine. Unload your models first.",
    "有程序正在使用此引擎。请先卸载你的模型。", "実行中のプログラムがこのエンジンを使っています。先にモデルをアンロードしてください。",
    "Ein laufendes Programm benutzt diese Engine. Entlade zuerst deine Modelle.", "Un programa en ejecución está usando este motor. Descarga primero tus modelos.",
    "Этот движок используется работающей программой. Сначала выгрузите свои модели.", "Um programa em execução está usando este motor. Descarregue seus modelos primeiro.",
    "실행 중인 프로그램이 이 엔진을 사용하고 있습니다. 먼저 모델을 언로드하세요.", "Un programme en cours utilise ce moteur. Déchargez d'abord vos modèles.")
add("Engine {0} is now in use.",
    "现在使用引擎 {0}。", "エンジン {0} を使用中です。", "Engine {0} wird jetzt verwendet.", "Ahora se usa el motor {0}.", "Теперь используется движок {0}.", "O motor {0} está em uso agora.",
    "이제 엔진 {0}을(를) 사용합니다.", "Le moteur {0} est maintenant utilisé.")
add("Engine {0} was deleted.",
    "引擎 {0} 已删除。", "エンジン {0} を削除しました。", "Engine {0} wurde gelöscht.", "Se eliminó el motor {0}.", "Движок {0} удалён.", "O motor {0} foi excluído.",
    "엔진 {0}을(를) 삭제했습니다.", "Le moteur {0} a été supprimé.")
add("Use engine {0}?",
    "使用引擎 {0}？", "エンジン {0} を使いますか？", "Engine {0} verwenden?", "¿Usar el motor {0}?", "Использовать движок {0}?", "Usar o motor {0}?", "엔진 {0}을(를) 사용할까요?", "Utiliser le moteur {0} ?")
add("Use engine {0}",
    "使用引擎 {0}", "エンジン {0} を使う", "Engine {0} verwenden", "Usar el motor {0}", "Использовать движок {0}", "Usar o motor {0}", "엔진 {0} 사용", "Utiliser le moteur {0}")
add("Models you start from now on run on engine {0}. Your models and settings stay as they are. You can switch back at any time.",
    "从现在起启动的模型将使用引擎 {0}。你的模型和设置保持不变。你可以随时切换回来。",
    "これから起動するモデルはエンジン {0} で動きます。モデルと設定はそのままです。いつでも元に戻せます。",
    "Modelle, die du ab jetzt startest, laufen mit Engine {0}. Deine Modelle und Einstellungen bleiben unverändert. Du kannst jederzeit zurückwechseln.",
    "Los modelos que inicies a partir de ahora usarán el motor {0}. Tus modelos y ajustes no cambian. Puedes volver cuando quieras.",
    "Модели, которые вы запустите с этого момента, будут работать на движке {0}. Ваши модели и настройки останутся как есть. Вернуться можно в любой момент.",
    "Os modelos que você iniciar a partir de agora usarão o motor {0}. Seus modelos e ajustes continuam como estão. Você pode voltar quando quiser.",
    "지금부터 시작하는 모델은 엔진 {0}에서 실행됩니다. 모델과 설정은 그대로 유지됩니다. 언제든 다시 바꿀 수 있습니다.",
    "Les modèles que vous démarrez à partir de maintenant utilisent le moteur {0}. Vos modèles et réglages restent inchangés. Vous pouvez revenir en arrière à tout moment.")
add("These models need a newer engine and may not start on it: {0}.",
    "这些模型需要更新的引擎，在此引擎上可能无法启动：{0}。", "次のモデルはより新しいエンジンが必要で、このエンジンでは起動しない可能性があります：{0}。",
    "Diese Modelle brauchen eine neuere Engine und starten damit eventuell nicht: {0}.", "Estos modelos necesitan un motor más reciente y puede que no arranquen con este: {0}.",
    "Этим моделям нужен более новый движок, и на этом они могут не запуститься: {0}.", "Estes modelos precisam de um motor mais novo e podem não iniciar com este: {0}.",
    "다음 모델은 더 새로운 엔진이 필요해 이 엔진에서는 시작되지 않을 수 있습니다: {0}.", "Ces modèles demandent un moteur plus récent et pourraient ne pas démarrer avec celui-ci : {0}.")
add("A model is running. Unload it in Models first, then switch.",
    "有模型正在运行。请先在“模型”中卸载它，然后再切换。", "モデルが実行中です。先に「モデル」でアンロードしてから切り替えてください。",
    "Ein Modell läuft. Entlade es zuerst unter „Modelle“ und wechsle dann.", "Hay un modelo en ejecución. Descárgalo primero en «Modelos» y luego cambia.",
    "Работает модель. Сначала выгрузите её в разделе «Модели», затем переключитесь.", "Há um modelo em execução. Descarregue-o primeiro em «Modelos» e depois troque.",
    "모델이 실행 중입니다. 먼저 '모델'에서 언로드한 다음 전환하세요.", "Un modèle est en cours d'exécution. Déchargez-le d'abord dans « Modèles », puis changez de moteur.")
add("Delete engine {0}?",
    "删除引擎 {0}？", "エンジン {0} を削除しますか？", "Engine {0} löschen?", "¿Eliminar el motor {0}?", "Удалить движок {0}?", "Excluir o motor {0}?", "엔진 {0}을(를) 삭제할까요?", "Supprimer le moteur {0} ?")
add("Keep engine", "保留引擎", "エンジンを残す", "Engine behalten", "Conservar motor", "Оставить движок", "Manter motor", "엔진 유지", "Garder le moteur")
add("Delete engine", "删除引擎", "エンジンを削除", "Engine löschen", "Eliminar motor", "Удалить движок", "Excluir motor", "엔진 삭제", "Supprimer le moteur")
add("This frees {0}. Your models and settings are kept.",
    "这将释放 {0}。你的模型和设置会保留。", "{0} が空きます。モデルと設定はそのまま残ります。", "Das gibt {0} frei. Deine Modelle und Einstellungen bleiben erhalten.",
    "Esto libera {0}. Tus modelos y ajustes se conservan.", "Это освободит {0}. Ваши модели и настройки сохранятся.", "Isso libera {0}. Seus modelos e ajustes são mantidos.",
    "{0}의 공간이 확보됩니다. 모델과 설정은 그대로 유지됩니다.", "Cela libère {0}. Vos modèles et réglages sont conservés.")
add("The newest engine can be downloaded again with Update in About.",
    "最新的引擎可以在“关于”中通过更新重新下载。", "最新のエンジンは、「情報」の更新でもう一度ダウンロードできます。",
    "Die neueste Engine lässt sich über das Update unter „Über“ erneut herunterladen.", "El motor más reciente se puede volver a descargar con Actualizar en «Acerca de».",
    "Новейший движок можно скачать снова через обновление в разделе «О программе».", "O motor mais recente pode ser baixado de novo com Atualizar em «Sobre».",
    "최신 엔진은 '정보'의 업데이트로 다시 내려받을 수 있습니다.", "Le moteur le plus récent peut être retéléchargé avec la mise à jour dans « À propos ».")
add("Strata can only download the newest engine, so a deleted older engine can't be brought back.",
    "Strata 只能下载最新的引擎，所以删除后无法找回较旧的引擎。", "Strata がダウンロードできるのは最新のエンジンだけなので、削除した古いエンジンは元に戻せません。",
    "Strata kann nur die neueste Engine herunterladen. Eine gelöschte ältere Engine lässt sich deshalb nicht wiederherstellen.",
    "Strata solo puede descargar el motor más reciente, así que un motor anterior eliminado no se puede recuperar.",
    "Strata может скачать только новейший движок, поэтому удалённый старый движок вернуть нельзя.",
    "O Strata só pode baixar o motor mais recente, então um motor antigo excluído não pode ser recuperado.",
    "Strata는 최신 엔진만 내려받을 수 있으므로, 삭제한 이전 엔진은 되돌릴 수 없습니다.",
    "Strata ne peut télécharger que le moteur le plus récent : un ancien moteur supprimé ne peut donc pas être récupéré.")

# ---- why an engine cannot be used on this computer
add("This engine was made for a different operating system.",
    "此引擎是为其他操作系统制作的。", "このエンジンは別のオペレーティングシステム向けです。", "Diese Engine wurde für ein anderes Betriebssystem gebaut.",
    "Este motor se creó para otro sistema operativo.", "Этот движок создан для другой операционной системы.", "Este motor foi feito para outro sistema operacional.",
    "이 엔진은 다른 운영체제용입니다.", "Ce moteur a été conçu pour un autre système d'exploitation.")
add("This engine was made for a different kind of graphics card than this computer has.",
    "此引擎是为与这台电脑不同类型的显卡制作的。", "このエンジンは、このコンピューターとは別の種類のグラフィックスカード向けです。",
    "Diese Engine wurde für eine andere Art von Grafikkarte gebaut als die in diesem Computer.", "Este motor se creó para un tipo de tarjeta gráfica distinto del de este equipo.",
    "Этот движок создан для другого типа видеокарты, чем в этом компьютере.", "Este motor foi feito para um tipo de placa de vídeo diferente da deste computador.",
    "이 엔진은 이 컴퓨터와 다른 종류의 그래픽 카드용입니다.", "Ce moteur a été conçu pour un autre type de carte graphique que celle de cet ordinateur.")
add("This engine needs Strata {0} or newer.",
    "此引擎需要 Strata {0} 或更新版本。", "このエンジンには Strata {0} 以降が必要です。", "Diese Engine braucht Strata {0} oder neuer.", "Este motor necesita Strata {0} o posterior.",
    "Для этого движка нужна Strata {0} или новее.", "Este motor precisa do Strata {0} ou mais recente.", "이 엔진에는 Strata {0} 이상이 필요합니다.", "Ce moteur nécessite Strata {0} ou plus récent.")
add("This engine doesn't support this computer's graphics card.",
    "此引擎不支持这台电脑的显卡。", "このエンジンは、このコンピューターのグラフィックスカードに対応していません。", "Diese Engine unterstützt die Grafikkarte dieses Computers nicht.",
    "Este motor no admite la tarjeta gráfica de este equipo.", "Этот движок не поддерживает видеокарту этого компьютера.", "Este motor não oferece suporte à placa de vídeo deste computador.",
    "이 엔진은 이 컴퓨터의 그래픽 카드를 지원하지 않습니다.", "Ce moteur ne prend pas en charge la carte graphique de cet ordinateur.")
add("This engine needs NVIDIA driver {0} or newer. This computer has {1}.",
    "此引擎需要 NVIDIA 驱动 {0} 或更新版本。这台电脑的驱动是 {1}。", "このエンジンには NVIDIA ドライバー {0} 以降が必要です。このコンピューターは {1} です。",
    "Diese Engine braucht den NVIDIA-Treiber {0} oder neuer. Dieser Computer hat {1}.", "Este motor necesita el controlador NVIDIA {0} o posterior. Este equipo tiene {1}.",
    "Для этого движка нужен драйвер NVIDIA {0} или новее. На этом компьютере {1}.", "Este motor precisa do driver NVIDIA {0} ou mais recente. Este computador tem o {1}.",
    "이 엔진에는 NVIDIA 드라이버 {0} 이상이 필요합니다. 이 컴퓨터는 {1}입니다.", "Ce moteur nécessite le pilote NVIDIA {0} ou plus récent. Cet ordinateur a le {1}.")
add("Some of this engine's files are missing or changed, so it can't be started.",
    "此引擎的部分文件缺失或已被更改，因此无法启动。", "このエンジンのファイルの一部が見つからないか変更されているため、起動できません。",
    "Einige Dateien dieser Engine fehlen oder wurden verändert, deshalb lässt sie sich nicht starten.", "Faltan archivos de este motor o han cambiado, así que no se puede iniciar.",
    "Часть файлов этого движка отсутствует или изменена, поэтому его нельзя запустить.", "Alguns arquivos deste motor estão ausentes ou foram alterados, então ele não pode ser iniciado.",
    "이 엔진의 일부 파일이 없거나 변경되어 시작할 수 없습니다.", "Certains fichiers de ce moteur sont absents ou modifiés : il ne peut pas démarrer.")
add("This engine's description is missing or damaged, so it can't be started.",
    "此引擎的说明文件缺失或已损坏，因此无法启动。", "このエンジンの説明ファイルが見つからないか壊れているため、起動できません。",
    "Die Beschreibung dieser Engine fehlt oder ist beschädigt, deshalb lässt sie sich nicht starten.", "Falta la descripción de este motor o está dañada, así que no se puede iniciar.",
    "Описание этого движка отсутствует или повреждено, поэтому его нельзя запустить.", "A descrição deste motor está ausente ou danificada, então ele não pode ser iniciado.",
    "이 엔진의 설명 파일이 없거나 손상되어 시작할 수 없습니다.", "La description de ce moteur est absente ou endommagée : il ne peut pas démarrer.")

# ---- updating moved to About: the words that sent people to Settings > Engine
add("Strata didn’t recognise an engine for this computer. The model will download, but it may not run until the right engine is installed (Settings → About).",
    "Strata 未能识别这台电脑的引擎。模型会下载，但在安装合适的引擎之前可能无法运行（设置 → 关于）。",
    "Strata はこのコンピューター用のエンジンを認識できませんでした。モデルはダウンロードされますが、合ったエンジンをインストールするまで動かない可能性があります（設定 → 情報）。",
    "Strata hat für diesen Computer keine Engine erkannt. Das Modell wird heruntergeladen, läuft aber möglicherweise erst, wenn die passende Engine installiert ist (Einstellungen → Über).",
    "Strata no reconoció ningún motor para este equipo. El modelo se descargará, pero puede que no funcione hasta que se instale el motor adecuado (Ajustes → Acerca de).",
    "Strata не распознала движок для этого компьютера. Модель скачается, но может не запуститься, пока не установлен подходящий движок (Настройки → О программе).",
    "O Strata não reconheceu um motor para este computador. O modelo será baixado, mas pode não funcionar até que o motor certo seja instalado (Configurações → Sobre).",
    "Strata가 이 컴퓨터용 엔진을 인식하지 못했습니다. 모델은 내려받지만, 알맞은 엔진이 설치되기 전에는 실행되지 않을 수 있습니다(설정 → 정보).",
    "Strata n’a pas reconnu de moteur pour cet ordinateur. Le modèle sera téléchargé, mais il risque de ne pas fonctionner tant que le bon moteur n’est pas installé (Réglages → À propos).")
add("Strata didn’t recognise an engine for this computer. The model is downloaded, but it can’t run until the right engine is installed. Open Settings → About.",
    "Strata 未能识别这台电脑的引擎。模型已下载，但在安装合适的引擎之前无法运行。请打开 设置 → 关于。",
    "Strata はこのコンピューター用のエンジンを認識できませんでした。モデルはダウンロード済みですが、合ったエンジンをインストールするまで動きません。設定 → 情報 を開いてください。",
    "Strata hat für diesen Computer keine Engine erkannt. Das Modell ist heruntergeladen, läuft aber erst, wenn die passende Engine installiert ist. Öffne Einstellungen → Über.",
    "Strata no reconoció ningún motor para este equipo. El modelo está descargado, pero no puede funcionar hasta que se instale el motor adecuado. Abre Ajustes → Acerca de.",
    "Strata не распознала движок для этого компьютера. Модель скачана, но не запустится, пока не установлен подходящий движок. Откройте Настройки → О программе.",
    "O Strata não reconheceu um motor para este computador. O modelo foi baixado, mas não pode funcionar até que o motor certo seja instalado. Abra Configurações → Sobre.",
    "Strata가 이 컴퓨터용 엔진을 인식하지 못했습니다. 모델은 내려받았지만, 알맞은 엔진이 설치되기 전에는 실행할 수 없습니다. 설정 → 정보를 여세요.",
    "Strata n’a pas reconnu de moteur pour cet ordinateur. Le modèle est téléchargé, mais il ne peut pas fonctionner tant que le bon moteur n’est pas installé. Ouvrez Réglages → À propos.")
add("Update the engine (Settings, About), then try again.",
    "请更新引擎（设置，关于），然后重试。", "エンジンを更新して（設定、情報）、もう一度お試しください。", "Aktualisiere die Engine (Einstellungen, Über) und versuche es dann erneut.",
    "Actualiza el motor (Ajustes, Acerca de) y vuelve a intentarlo.", "Обновите движок (Настройки, О программе) и попробуйте снова.", "Atualize o motor (Configurações, Sobre) e tente novamente.",
    "엔진을 업데이트(설정, 정보)한 뒤 다시 시도하세요.", "Mettez le moteur à jour (Réglages, À propos), puis réessayez.")

# ---- answers of the host (shown as they are, so each one is a catalog entry)
add("A model is running, so nothing was changed. Unload the model in Models, then try again.",
    "有模型正在运行，所以没有做任何更改。请先在“模型”中卸载模型，然后重试。", "モデルが実行中のため、何も変更していません。「モデル」でモデルをアンロードしてから、もう一度お試しください。",
    "Ein Modell läuft, deshalb wurde nichts geändert. Entlade das Modell unter „Modelle“ und versuche es dann erneut.", "Hay un modelo en ejecución, así que no se cambió nada. Descarga el modelo en «Modelos» y vuelve a intentarlo.",
    "Работает модель, поэтому ничего не изменено. Выгрузите модель в разделе «Модели» и попробуйте снова.", "Há um modelo em execução, então nada foi alterado. Descarregue o modelo em «Modelos» e tente novamente.",
    "모델이 실행 중이라 아무것도 바뀌지 않았습니다. '모델'에서 모델을 언로드한 다음 다시 시도하세요.", "Un modèle est en cours d'exécution, rien n'a donc été modifié. Déchargez le modèle dans « Modèles », puis réessayez.")
add("This is the engine in use. Switch to another one first, then delete it.",
    "这是正在使用的引擎。请先切换到另一个引擎，再删除它。", "これは使用中のエンジンです。先に別のエンジンに切り替えてから削除してください。",
    "Das ist die verwendete Engine. Wechsle zuerst zu einer anderen und lösche sie dann.", "Este es el motor en uso. Cambia primero a otro y luego elimínalo.",
    "Это используемый движок. Сначала переключитесь на другой, затем удалите его.", "Este é o motor em uso. Troque primeiro para outro e depois exclua-o.",
    "사용 중인 엔진입니다. 먼저 다른 엔진으로 전환한 다음 삭제하세요.", "C'est le moteur utilisé. Passez d'abord à un autre, puis supprimez-le.")
add("A program is still using this engine, so it was not deleted. Unload your models and try again.",
    "仍有程序在使用此引擎，所以没有删除。请卸载你的模型后重试。", "プログラムがまだこのエンジンを使っているため、削除しませんでした。モデルをアンロードしてから、もう一度お試しください。",
    "Ein Programm benutzt diese Engine noch, deshalb wurde sie nicht gelöscht. Entlade deine Modelle und versuche es erneut.", "Un programa todavía usa este motor, así que no se eliminó. Descarga tus modelos y vuelve a intentarlo.",
    "Этот движок ещё используется программой, поэтому он не удалён. Выгрузите модели и попробуйте снова.", "Um programa ainda está usando este motor, então ele não foi excluído. Descarregue seus modelos e tente novamente.",
    "프로그램이 아직 이 엔진을 사용하고 있어 삭제하지 않았습니다. 모델을 언로드한 다음 다시 시도하세요.", "Un programme utilise encore ce moteur : il n'a pas été supprimé. Déchargez vos modèles, puis réessayez.")
add("That engine is already gone. The list has been refreshed.",
    "该引擎已经不存在。列表已刷新。", "そのエンジンはすでにありません。一覧を更新しました。", "Diese Engine ist schon nicht mehr da. Die Liste wurde aktualisiert.",
    "Ese motor ya no está. La lista se ha actualizado.", "Этого движка уже нет. Список обновлён.", "Esse motor já não existe. A lista foi atualizada.", "해당 엔진은 이미 없습니다. 목록을 새로 고쳤습니다.", "Ce moteur n'existe déjà plus. La liste a été actualisée.")
add("An update is running. Wait for it to finish, then try again.",
    "正在进行更新。请等它完成后再试。", "更新を実行中です。終わるまで待ってから、もう一度お試しください。", "Ein Update läuft. Warte, bis es fertig ist, und versuche es dann erneut.",
    "Hay una actualización en curso. Espera a que termine y vuelve a intentarlo.", "Идёт обновление. Дождитесь его окончания и попробуйте снова.", "Uma atualização está em andamento. Aguarde o fim e tente novamente.",
    "업데이트가 진행 중입니다. 끝날 때까지 기다린 다음 다시 시도하세요.", "Une mise à jour est en cours. Attendez qu'elle se termine, puis réessayez.")
add("This engine can't run on this computer, so nothing was changed.",
    "此引擎无法在这台电脑上运行，所以没有做任何更改。", "このエンジンはこのコンピューターでは動かないため、何も変更していません。", "Diese Engine läuft auf diesem Computer nicht, deshalb wurde nichts geändert.",
    "Este motor no puede funcionar en este equipo, así que no se cambió nada.", "Этот движок не может работать на этом компьютере, поэтому ничего не изменено.", "Este motor não pode funcionar neste computador, então nada foi alterado.",
    "이 엔진은 이 컴퓨터에서 실행할 수 없어 아무것도 바뀌지 않았습니다.", "Ce moteur ne peut pas fonctionner sur cet ordinateur : rien n'a été modifié.")
add("The engine wasn't deleted. Nothing else was changed.",
    "引擎没有被删除。其他内容均未更改。", "エンジンは削除されませんでした。ほかには何も変更していません。", "Die Engine wurde nicht gelöscht. Sonst wurde nichts geändert.",
    "El motor no se eliminó. No se cambió nada más.", "Движок не удалён. Больше ничего не изменено.", "O motor não foi excluído. Nada mais foi alterado.", "엔진이 삭제되지 않았습니다. 다른 것은 바뀌지 않았습니다.", "Le moteur n'a pas été supprimé. Rien d'autre n'a été modifié.")
add("The engine wasn't switched. Nothing was changed.",
    "引擎没有被切换。没有做任何更改。", "エンジンは切り替わりませんでした。何も変更していません。", "Die Engine wurde nicht gewechselt. Es wurde nichts geändert.",
    "No se cambió el motor. No se modificó nada.", "Движок не переключён. Ничего не изменено.", "O motor não foi trocado. Nada foi alterado.", "엔진이 전환되지 않았습니다. 아무것도 바뀌지 않았습니다.", "Le moteur n'a pas été changé. Rien n'a été modifié.")
add("Choose an engine from the list.",
    "请从列表中选择一个引擎。", "一覧からエンジンを選んでください。", "Wähle eine Engine aus der Liste.", "Elige un motor de la lista.", "Выберите движок из списка.", "Escolha um motor da lista.", "목록에서 엔진을 선택하세요.", "Choisissez un moteur dans la liste.")
add("A newer engine is already on this computer. Update switches to it; nothing is downloaded.",
    "这台电脑上已经有更新的引擎。点击更新会切换到它，不会下载任何内容。", "このコンピューターにはすでに新しいエンジンがあります。更新するとそれに切り替わり、ダウンロードはされません。",
    "Auf diesem Computer ist schon eine neuere Engine. Das Update wechselt zu ihr; es wird nichts heruntergeladen.", "Este equipo ya tiene un motor más reciente. Actualizar cambia a él; no se descarga nada.",
    "На этом компьютере уже есть более новый движок. Обновление переключит на него; ничего не скачивается.", "Este computador já tem um motor mais novo. Atualizar troca para ele; nada é baixado.",
    "이 컴퓨터에 이미 더 새로운 엔진이 있습니다. 업데이트하면 그 엔진으로 전환되며, 내려받는 것은 없습니다.", "Cet ordinateur a déjà un moteur plus récent. La mise à jour bascule dessus ; rien n'est téléchargé.")
# ---- the Use dialog of a Qwen4 model shape's engine (owner 11-09: {0} is the group's title, the model's own name; {1} the engine's version). Only the first
# sentence is new: the two after it ("Your models and settings stay as they are. You can switch back at any time.") are the existing translations, taken from the
# Strata engine's Use sentence below so they read exactly the same in every language.
_SHAPE_FIRST = dict(en="{0} runs on engine {1} from its next start.",
                    zh="{0} 将从下次启动起使用引擎 {1}。", ja="{0} は次回の起動からエンジン {1} で動作します。",
                    de="{0} läuft ab dem nächsten Start mit Engine {1}.", es="{0} funcionará con el motor {1} a partir de su próximo inicio.",
                    ru="{0} будет работать на движке {1} со следующего запуска.", pt="{0} roda no motor {1} a partir da próxima inicialização.",
                    ko="{0}은(는) 다음 시작부터 엔진 {1}에서 실행됩니다.", fr="{0} fonctionnera avec le moteur {1} dès son prochain démarrage.")
_USE = next(e for e in S if e["en"].startswith("Models you start from now on run on engine {0}."))


def _rest(lang):
    """The existing Use sentence without its first sentence: the reassurance, as it is translated today."""
    text = _USE[lang]
    cut = re.search(r"[。.]\s*", text)
    return text[cut.end():]


add(*[_SHAPE_FIRST[l] + ("" if l in ("zh", "ja") else " ") + _rest(l) for l in LANGS])

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
        data[f"et{i + 1:02d}"] = french(text) if lang == "fr" else text
    (here / f"engine-{lang}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("written", len(S), "strings in", len(LANGS), "languages")
