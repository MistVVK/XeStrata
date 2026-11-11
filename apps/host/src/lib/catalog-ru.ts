/* All the interface strings of one language (ru): the base file and the 13 add-on files, merged in the order the app has always used (a later file wins).
 Loaded the first time the language is used (see loadCatalog in i18n.ts); English stays in the main file because the English text is the key. */
import base from '../locales/ru.json';
import tune from '../locales/tune-ru.json';
import tools from '../locales/tools-ru.json';
import fix from '../locales/fix-ru.json';
import gh from '../locales/gh-ru.json';
import ui from '../locales/ui-ru.json';
import engine from '../locales/engine-ru.json';
import pick from '../locales/pick-ru.json';
import df2 from '../locales/df2-ru.json';
import upd from '../locales/upd-ru.json';
import r3 from '../locales/r3-ru.json';
import pic from '../locales/pic-ru.json';
import gui from '../locales/gui-ru.json';
import sav from '../locales/sav-ru.json';
import net from '../locales/net-ru.json';
export default {...base,...tune,...tools,...fix,...gh,...ui,...engine,...pick,...df2,...upd,...r3,...pic,...gui,...sav,...net} as Record<string,string>;
