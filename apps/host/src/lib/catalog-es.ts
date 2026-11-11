/* All the interface strings of one language (es): the base file and the 13 add-on files, merged in the order the app has always used (a later file wins).
 Loaded the first time the language is used (see loadCatalog in i18n.ts); English stays in the main file because the English text is the key. */
import base from '../locales/es.json';
import tune from '../locales/tune-es.json';
import tools from '../locales/tools-es.json';
import fix from '../locales/fix-es.json';
import gh from '../locales/gh-es.json';
import ui from '../locales/ui-es.json';
import engine from '../locales/engine-es.json';
import pick from '../locales/pick-es.json';
import df2 from '../locales/df2-es.json';
import upd from '../locales/upd-es.json';
import r3 from '../locales/r3-es.json';
import pic from '../locales/pic-es.json';
import gui from '../locales/gui-es.json';
import sav from '../locales/sav-es.json';
import net from '../locales/net-es.json';
export default {...base,...tune,...tools,...fix,...gh,...ui,...engine,...pick,...df2,...upd,...r3,...pic,...gui,...sav,...net} as Record<string,string>;
