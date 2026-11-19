/**
 * Tencent Hy4-Flash runs on its own engine program (strata-hy4, the component "strata-hy4"), which takes none of the Flash-Next options the app writes for the Strata engine
 * (--vram-reserve-mib, --pool-workers, --resident-budget-gib, --batch, --mtp, --spec ...) and refuses a flag it does not know. This is the one test the app's settings code asks;
 * the run config says it with its arch (terminal/serve/hy4_kind.py is the Python side of the same test).
 */
export const HY4_ARCH = 'hy4flash';
export const HY4_FAMILY = 'hy4-flash';
export const isHy4 = config => config?.arch === HY4_ARCH || /(^|[\\/:])strata-hy4(\.exe)?$/i.test(String(config?.exe ?? ''));
export const isHy4Family = family => family === HY4_FAMILY;
