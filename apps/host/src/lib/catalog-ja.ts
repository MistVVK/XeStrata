/* All the interface strings of one language (ja): the base file and the 13 add-on files, merged in the order the app has always used (a later file wins).
 Loaded the first time the language is used (see loadCatalog in i18n.ts); English stays in the main file because the English text is the key. */
import base from '../locales/ja.json';
import tune from '../locales/tune-ja.json';
import tools from '../locales/tools-ja.json';
import fix from '../locales/fix-ja.json';
import gh from '../locales/gh-ja.json';
import ui from '../locales/ui-ja.json';
import engine from '../locales/engine-ja.json';
import pick from '../locales/pick-ja.json';
import df2 from '../locales/df2-ja.json';
import upd from '../locales/upd-ja.json';
import r3 from '../locales/r3-ja.json';
import pic from '../locales/pic-ja.json';
import gui from '../locales/gui-ja.json';
import sav from '../locales/sav-ja.json';
import net from '../locales/net-ja.json';
export default {...base,...tune,...tools,...fix,...gh,...ui,...engine,...pick,...df2,...upd,...r3,...pic,...gui,...sav,...net} as Record<string,string>;
