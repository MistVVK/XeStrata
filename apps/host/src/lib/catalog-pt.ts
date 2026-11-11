/* All the interface strings of one language (pt): the base file and the 13 add-on files, merged in the order the app has always used (a later file wins).
 Loaded the first time the language is used (see loadCatalog in i18n.ts); English stays in the main file because the English text is the key. */
import base from '../locales/pt.json';
import tune from '../locales/tune-pt.json';
import tools from '../locales/tools-pt.json';
import fix from '../locales/fix-pt.json';
import gh from '../locales/gh-pt.json';
import ui from '../locales/ui-pt.json';
import engine from '../locales/engine-pt.json';
import pick from '../locales/pick-pt.json';
import df2 from '../locales/df2-pt.json';
import upd from '../locales/upd-pt.json';
import r3 from '../locales/r3-pt.json';
import pic from '../locales/pic-pt.json';
import gui from '../locales/gui-pt.json';
import sav from '../locales/sav-pt.json';
import net from '../locales/net-pt.json';
export default {...base,...tune,...tools,...fix,...gh,...ui,...engine,...pick,...df2,...upd,...r3,...pic,...gui,...sav,...net} as Record<string,string>;
