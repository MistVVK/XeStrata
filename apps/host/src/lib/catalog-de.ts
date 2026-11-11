/* All the interface strings of one language (de): the base file and the 13 add-on files, merged in the order the app has always used (a later file wins).
 Loaded the first time the language is used (see loadCatalog in i18n.ts); English stays in the main file because the English text is the key. */
import base from '../locales/de.json';
import tune from '../locales/tune-de.json';
import tools from '../locales/tools-de.json';
import fix from '../locales/fix-de.json';
import gh from '../locales/gh-de.json';
import ui from '../locales/ui-de.json';
import engine from '../locales/engine-de.json';
import pick from '../locales/pick-de.json';
import df2 from '../locales/df2-de.json';
import upd from '../locales/upd-de.json';
import r3 from '../locales/r3-de.json';
import pic from '../locales/pic-de.json';
import gui from '../locales/gui-de.json';
import sav from '../locales/sav-de.json';
import net from '../locales/net-de.json';
export default {...base,...tune,...tools,...fix,...gh,...ui,...engine,...pick,...df2,...upd,...r3,...pic,...gui,...sav,...net} as Record<string,string>;
