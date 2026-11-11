/* All the interface strings of one language (fr): the base file and the 13 add-on files, merged in the order the app has always used (a later file wins).
 Loaded the first time the language is used (see loadCatalog in i18n.ts); English stays in the main file because the English text is the key. */
import base from '../locales/fr.json';
import tune from '../locales/tune-fr.json';
import tools from '../locales/tools-fr.json';
import fix from '../locales/fix-fr.json';
import gh from '../locales/gh-fr.json';
import ui from '../locales/ui-fr.json';
import engine from '../locales/engine-fr.json';
import pick from '../locales/pick-fr.json';
import df2 from '../locales/df2-fr.json';
import upd from '../locales/upd-fr.json';
import r3 from '../locales/r3-fr.json';
import pic from '../locales/pic-fr.json';
import gui from '../locales/gui-fr.json';
import sav from '../locales/sav-fr.json';
import net from '../locales/net-fr.json';
export default {...base,...tune,...tools,...fix,...gh,...ui,...engine,...pick,...df2,...upd,...r3,...pic,...gui,...sav,...net} as Record<string,string>;
