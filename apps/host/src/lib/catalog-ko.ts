/* All the interface strings of one language (ko): the base file and the 13 add-on files, merged in the order the app has always used (a later file wins).
 Loaded the first time the language is used (see loadCatalog in i18n.ts); English stays in the main file because the English text is the key. */
import base from '../locales/ko.json';
import tune from '../locales/tune-ko.json';
import tools from '../locales/tools-ko.json';
import fix from '../locales/fix-ko.json';
import gh from '../locales/gh-ko.json';
import ui from '../locales/ui-ko.json';
import engine from '../locales/engine-ko.json';
import pick from '../locales/pick-ko.json';
import df2 from '../locales/df2-ko.json';
import upd from '../locales/upd-ko.json';
import r3 from '../locales/r3-ko.json';
import pic from '../locales/pic-ko.json';
import gui from '../locales/gui-ko.json';
import sav from '../locales/sav-ko.json';
import net from '../locales/net-ko.json';
export default {...base,...tune,...tools,...fix,...gh,...ui,...engine,...pick,...df2,...upd,...r3,...pic,...gui,...sav,...net} as Record<string,string>;
