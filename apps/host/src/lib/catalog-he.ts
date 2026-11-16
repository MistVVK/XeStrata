/* All the interface strings of one language (he): the base file and the 15 add-on files, merged in the order the app has always used (a later file wins).
 Loaded the first time the language is used (see loadCatalog in i18n.ts); English stays in the main file because the English text is the key. */
import base from '../locales/he.json';
import tune from '../locales/tune-he.json';
import tools from '../locales/tools-he.json';
import fix from '../locales/fix-he.json';
import gh from '../locales/gh-he.json';
import ui from '../locales/ui-he.json';
import engine from '../locales/engine-he.json';
import pick from '../locales/pick-he.json';
import df2 from '../locales/df2-he.json';
import upd from '../locales/upd-he.json';
import r3 from '../locales/r3-he.json';
import pic from '../locales/pic-he.json';
import gui from '../locales/gui-he.json';
import sav from '../locales/sav-he.json';
import net from '../locales/net-he.json';
export default {...base,...tune,...tools,...fix,...gh,...ui,...engine,...pick,...df2,...upd,...r3,...pic,...gui,...sav,...net,...dmg} as Record<string,string>;
import dmg from '../locales/dmg-he.json';
