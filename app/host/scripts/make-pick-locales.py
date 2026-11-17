"""Writes src/locales/pick-<lang>.json: the strings of the model version list (VersionPicker), the code-block preview button and the This computer panel (branch f-0291-ui-fixes2).
Run from apps/host:  python scripts/make-pick-locales.py
Every entry names all nine languages; the keys are vp01.. in the order of S. pick-en.json is the English source. Placeholders ({0}, {1}) must be kept
in every language. A separate file keeps this work from colliding with the other catalogs (the same reason tune-, tools-, fix-, gh-, ui- and engine- exist)."""
import json
import re
from pathlib import Path

LANGS = ["en", "zh", "ja", "de", "es", "ru", "pt", "ko", "fr"]
S = []


def add(en, zh, ja, de, es, ru, pt, ko, fr):
    S.append(dict(en=en, zh=zh, ja=ja, de=de, es=es, ru=ru, pt=pt, ko=ko, fr=fr))


add("{0} download",
    "下载 {0}", "ダウンロード {0}", "{0} Download", "Descarga de {0}", "Загрузка {0}", "Download de {0}", "다운로드 {0}", "Téléchargement de {0}")
add("Needs {0} of memory",
    "需要 {0} 内存", "{0} のメモリーが必要", "Braucht {0} Speicher", "Necesita {0} de memoria", "Нужно {0} памяти", "Precisa de {0} de memória", "메모리 {0} 필요", "Nécessite {0} de mémoire")
add("Needs {0} of memory and about {1} GB of disk",
    "需要 {0} 内存和约 {1} GB 磁盘空间", "{0} のメモリーと約 {1} GB のディスク容量が必要", "Braucht {0} Speicher und etwa {1} GB Festplattenplatz", "Necesita {0} de memoria y unos {1} GB de disco",
    "Нужно {0} памяти и около {1} ГБ на диске", "Precisa de {0} de memória e cerca de {1} GB de disco", "메모리 {0}와 디스크 약 {1} GB 필요", "Nécessite {0} de mémoire et environ {1} Go de disque")
add("Fits",
    "可以运行", "問題なく動きます", "Passt", "Cabe", "Помещается", "Cabe", "문제없이 실행됩니다", "Tient")
add("Fits with SSD streaming",
    "可以运行，部分内容从 SSD 读取", "SSD から読み込みながら動きます", "Passt mit SSD-Streaming", "Cabe con streaming desde el SSD", "Помещается с потоковой подгрузкой с SSD", "Cabe com streaming do SSD", "SSD 스트리밍으로 실행됩니다", "Tient avec streaming depuis le SSD")
add("Doesn't fit",
    "放不下", "収まりません", "Passt nicht", "No cabe", "Не помещается", "Não cabe", "맞지 않습니다", "Ne tient pas")
add("Show details",
    "显示详情", "詳細を表示", "Details anzeigen", "Mostrar detalles", "Показать подробности", "Mostrar detalhes", "자세히 보기", "Afficher les détails")
add("Hide details",
    "隐藏详情", "詳細を隠す", "Details ausblenden", "Ocultar detalles", "Скрыть подробности", "Ocultar detalhes", "자세히 숨기기", "Masquer les détails")
add("{0} RAM + VRAM",
    "{0} 内存 + 显存", "{0} の RAM + VRAM", "{0} RAM + VRAM", "{0} de RAM + VRAM", "{0} RAM + VRAM", "{0} de RAM + VRAM", "RAM + VRAM {0}", "{0} de RAM + VRAM")

add("Close preview",
    "关闭预览", "プレビューを閉じる", "Vorschau schließen", "Cerrar vista previa", "Закрыть предпросмотр", "Fechar pré-visualização", "미리보기 닫기", "Fermer l'aperçu")

add("Free right now",
    "当前可用", "現在の空き", "Gerade frei", "Libre ahora", "Свободно сейчас", "Livre agora", "지금 비어 있음", "Libre en ce moment")
add("{0} free",
    "可用 {0}", "空き {0}", "{0} frei", "{0} libres", "свободно {0}", "{0} livres", "{0} 사용 가능", "{0} libres")
add("{0}: {1} free of {2}",
    "{0}：{2} 中有 {1} 可用", "{0}：{2} のうち {1} が空き", "{0}: {1} frei von {2}", "{0}: {1} libres de {2}", "{0}: свободно {1} из {2}", "{0}: {1} livres de {2}", "{0}: {2} 중 {1} 사용 가능", "{0} : {1} libres sur {2}")
add("It takes about",
    "它大约占用", "使用量の目安", "Es belegt etwa", "Ocupa aproximadamente", "Занимает примерно", "Ocupa cerca de", "대략 사용하는 양", "Il occupe environ")
add("Nothing. It fits in memory.",
    "没有。它完全放得进内存。", "ありません。すべてメモリーに収まっています。", "Nichts. Es passt in den Speicher.", "Nada. Cabe en la memoria.", "Ничего. Всё помещается в память.", "Nada. Cabe na memória.", "없음. 메모리에 모두 들어갑니다.", "Rien. Tout tient en mémoire.")
add("Reading from your SSD now: {0}",
    "此刻正从 SSD 读取：{0}", "現在 SSD から読み込み中：{0}", "Gerade vom SSD gelesen: {0}", "Leyendo del SSD ahora: {0}", "Сейчас читается с SSD: {0}", "Lendo do SSD agora: {0}", "지금 SSD에서 읽는 중: {0}", "Lecture depuis le SSD en cours : {0}")
add("This version would take about {0} of graphics memory and {1} of RAM.",
    "这个版本大约会占用 {0} 显存和 {1} 内存。", "このバージョンはグラフィックスメモリーを約 {0}、RAM を約 {1} 使います。", "Diese Version würde etwa {0} Grafikspeicher und {1} RAM belegen.",
    "Esta versión ocuparía aproximadamente {0} de memoria gráfica y {1} de RAM.", "Эта версия займёт примерно {0} видеопамяти и {1} оперативной памяти.", "Esta versão ocuparia cerca de {0} de memória gráfica e {1} de RAM.",
    "이 버전은 그래픽 메모리 약 {0}와 RAM 약 {1}을 사용합니다.", "Cette version occuperait environ {0} de mémoire graphique et {1} de RAM.")

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
        data[f"vp{i + 1:02d}"] = french(text) if lang == "fr" else text
    (here / f"pick-{lang}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("written", len(S), "strings in", len(LANGS), "languages")
