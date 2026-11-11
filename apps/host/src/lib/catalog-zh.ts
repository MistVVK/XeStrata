/* All the interface strings of one language (zh): the base file and the 13 add-on files, merged in the order the app has always used (a later file wins).
 Loaded the first time the language is used (see loadCatalog in i18n.ts); English stays in the main file because the English text is the key. */
import base from '../locales/zh.json';
import tune from '../locales/tune-zh.json';
import tools from '../locales/tools-zh.json';
import fix from '../locales/fix-zh.json';
import gh from '../locales/gh-zh.json';
import ui from '../locales/ui-zh.json';
import engine from '../locales/engine-zh.json';
import pick from '../locales/pick-zh.json';
import df2 from '../locales/df2-zh.json';
import upd from '../locales/upd-zh.json';
import r3 from '../locales/r3-zh.json';
import pic from '../locales/pic-zh.json';
import gui from '../locales/gui-zh.json';
import sav from '../locales/sav-zh.json';
import net from '../locales/net-zh.json';
export default {...base,...tune,...tools,...fix,...gh,...ui,...engine,...pick,...df2,...upd,...r3,...pic,...gui,...sav,...net} as Record<string,string>;
