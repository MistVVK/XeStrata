/** A number typed by a person: "0.95", ".95", "0,95" (a comma is a decimal point in many languages). null while it is not a whole number yet ("", ".", "0.", "-"). */
export function parseDecimal(text: string): number | null {
  const t = String(text ?? '').trim().replace(',', '.');
  if (!/^-?(\d+\.?\d*|\.\d+)$/.test(t) || t.endsWith('.')) return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}
/** Whether what is typed could still become a number: the field keeps it while the person is in the middle of "0." */
export const partialDecimal = (text: string) => /^-?\d*[.,]?\d*$/.test(String(text ?? '').trim());
