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
    fr="Déplacer vers un espace de travail…")
add(en="Move chat to another workspace",
    zh="将聊天移动到其他工作区",
    ja="チャットを別のワークスペースへ移動",
    de="Chat in einen anderen Arbeitsbereich verschieben",
    es="Mover el chat a otro espacio de trabajo",
    ru="Переместить чат в другое рабочее пространство",
    pt="Mover o chat para outro espaço de trabalho",
    ko="채팅을 다른 작업 공간으로 이동",
    fr="Déplacer la conversation vers un autre espace de travail")
add(en="Workspace",
    zh="工作区",
    ja="ワークスペース",
    de="Arbeitsbereich",
    es="Espacio de trabajo",
    ru="Рабочее пространство",
    pt="Espaço de trabalho",
    ko="작업 공간",
    fr="Espace de travail")
add(en="The chat is listed under this workspace. Its next answer works in that workspace’s folder, unless you chose a folder for this chat.",
    zh="聊天会列在这个工作区下。它的下一个回答会在该工作区的文件夹中进行，除非你为这个聊天选择了文件夹。",
    ja="チャットはこのワークスペースの下に表示されます。このチャット用にフォルダーを選んでいない限り、次の回答はそのワークスペースのフォルダーで作業します。",
    de="Der Chat wird unter diesem Arbeitsbereich aufgeführt. Seine nächste Antwort arbeitet im Ordner dieses Arbeitsbereichs, es sei denn, Sie haben für diesen Chat einen Ordner gewählt.",
    es="El chat aparece en este espacio de trabajo. Su próxima respuesta trabaja en la carpeta de ese espacio de trabajo, salvo que hayas elegido una carpeta para este chat.",
    ru="Чат будет показан в этом рабочем пространстве. Следующий ответ будет работать в папке этого пространства, если вы не выбрали папку для этого чата.",
    pt="O chat passa a aparecer neste espaço de trabalho. A próxima resposta trabalha na pasta desse espaço de trabalho, a menos que você tenha escolhido uma pasta para este chat.",
    ko="채팅이 이 작업 공간 아래에 표시됩니다. 이 채팅에 폴더를 직접 고르지 않았다면 다음 답변은 그 작업 공간의 폴더에서 작업합니다.",
    fr="La conversation est rangée dans cet espace de travail. Sa prochaine réponse travaille dans le dossier de cet espace, sauf si vous avez choisi un dossier pour cette conversation.")
add(en="Move chat",
    zh="移动聊天",
    ja="チャットを移動",
    de="Chat verschieben",
    es="Mover chat",
    ru="Переместить чат",
    pt="Mover chat",
    ko="채팅 이동",
    fr="Déplacer la conversation")
add(en="This chat is still working. Stop it or wait for it to finish, then move it.",
    zh="这个聊天仍在工作。请先停止它或等它完成，然后再移动。",
    ja="このチャットはまだ作業中です。停止するか終わるまで待ってから移動してください。",
    de="Dieser Chat arbeitet noch. Stoppen Sie ihn oder warten Sie, bis er fertig ist, und verschieben Sie ihn dann.",
    es="Este chat todavía está trabajando. Detenlo o espera a que termine y luego muévelo.",
    ru="Этот чат ещё работает. Остановите его или дождитесь окончания, затем переместите.",
    pt="Este chat ainda está trabalhando. Pare-o ou espere ele terminar e depois mova-o.",
    ko="이 채팅은 아직 작업 중입니다. 중지하거나 끝날 때까지 기다린 다음 이동하세요.",
    fr="Cette conversation travaille encore. Arrêtez-la ou attendez qu’elle se termine, puis déplacez-la.")

here = APP / "src" / "locales"
NBSP = " "


def french(text):
    text = re.sub(r"(?<=[A-Za-zÀ-ſ])'(?=[A-Za-zÀ-ſ{])", "’", text)
    text = re.sub(r"(?<=\S) ([:;?!])", NBSP + r"\1", text)
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
