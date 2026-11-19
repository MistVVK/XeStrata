// Lane GUI-TUI (0.2.17): add strings to the gtui-<lang>.json family (idempotent: a key already there is kept). node scripts/add-gtui-strings.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const dir = join(dirname(fileURLToPath(import.meta.url)), '..', 'src', 'locales');
const add = {
  en: {
    gtui03: 'Some earlier pictures are missing',
    gtui04: 'Pictures from earlier messages are no longer stored in this browser, so the model does not see them in this answer. Attach them again if they matter.',
  },
  de: {
    gtui03: 'Einige frühere Bilder fehlen',
    gtui04: 'Bilder aus früheren Nachrichten sind in diesem Browser nicht mehr gespeichert, deshalb sieht das Modell sie in dieser Antwort nicht. Hängen Sie sie erneut an, wenn sie wichtig sind.',
  },
  es: {
    gtui03: 'Faltan algunas imágenes anteriores',
    gtui04: 'Las imágenes de mensajes anteriores ya no están guardadas en este navegador, así que el modelo no las ve en esta respuesta. Adjúntelas de nuevo si son importantes.',
  },
  fr: {
    gtui03: 'Certaines images précédentes sont absentes',
    gtui04: 'Les images des messages précédents ne sont plus conservées dans ce navigateur ; le modèle ne les voit donc pas dans cette réponse. Joignez-les de nouveau si elles comptent.',
  },
  pt: {
    gtui03: 'Faltam algumas imagens anteriores',
    gtui04: 'As imagens de mensagens anteriores já não estão guardadas neste navegador, por isso o modelo não as vê nesta resposta. Anexe-as de novo se forem importantes.',
  },
  ru: {
    gtui03: 'Некоторые прежние изображения недоступны',
    gtui04: 'Изображения из прежних сообщений больше не хранятся в этом браузере, поэтому модель не видит их в этом ответе. Прикрепите их снова, если они важны.',
  },
  zh: {
    gtui03: '部分之前的图片已丢失',
    gtui04: '之前消息中的图片已不再保存在此浏览器中，因此模型在这次回答中看不到它们。如有需要，请重新附加。',
  },
  ja: {
    gtui03: '以前の画像の一部が見つかりません',
    gtui04: '以前のメッセージの画像はこのブラウザーに保存されていないため、今回の返答ではモデルに見えません。必要なら、もう一度添付してください。',
  },
  ko: {
    gtui03: '이전 사진 일부가 없습니다',
    gtui04: '이전 메시지의 사진이 이 브라우저에 더 이상 저장되어 있지 않아 모델이 이번 답변에서 볼 수 없습니다. 필요하면 다시 첨부하세요.',
  },
  he: {
    gtui03: 'חלק מהתמונות הקודמות חסרות',
    gtui04: 'התמונות מהודעות קודמות כבר אינן שמורות בדפדפן הזה, ולכן המודל לא רואה אותן בתשובה הזאת. צרפו אותן שוב אם הן חשובות.',
  },
};
for (const [lang, strings] of Object.entries(add)) {
  const file = join(dir, `gtui-${lang}.json`);
  const data = JSON.parse(readFileSync(file, 'utf8'));
  for (const [k, v] of Object.entries(strings)) if (!(k in data)) data[k] = v;
  writeFileSync(file, JSON.stringify(data, null, 1) + '\n');
}
console.log('ok');
