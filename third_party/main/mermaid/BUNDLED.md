<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# What mermaid.min.js bundles

mermaid 12.3.0's `dist/mermaid.min.js` is one script that carries these packages besides mermaid's own code
(MIT), read from the package's source map (`mermaid.min.js.map`, its `sources`), each with the license its npm
package states. Where a package offers a choice, XeStrata takes it under the first named here: elkjs under
EPL-2.0, DOMPurify under Apache-2.0. The license texts are in the repository's `LICENSES/`.

| Package | Version | License |
| --- | --- | --- |
| `@braintree/sanitize-url` | 7.1.2 | MIT |
| `@chevrotain/cst-dts-gen` | 13.2.0 | Apache-2.0 |
| `@chevrotain/gast` | 13.2.0 | Apache-2.0 |
| `@chevrotain/regexp-to-ast` | 13.2.0 | Apache-2.0 |
| `@chevrotain/utils` | 13.2.0 | Apache-2.0 |
| `@iconify/utils` | 3.0.2 | MIT |
| `@mermaid-js/parser` | 2.0.1 | MIT |
| `@upsetjs/venn.js` | 2.0.0 | MIT |
| `chevrotain` | 13.2.0 | Apache-2.0 |
| `cose-base` | 1.0.3 | MIT |
| `cose-base` | 2.2.0 | MIT |
| `cytoscape` | 3.34.0 | MIT |
| `cytoscape-cose-bilkent` | 4.1.0 | MIT |
| `cytoscape-fcose` | 2.2.0 | MIT |
| `d3` | 7.9.0 | ISC |
| `d3-array` | 2.12.1 | BSD-3-Clause |
| `d3-array` | 3.2.4 | ISC |
| `d3-axis` | 3.0.0 | ISC |
| `d3-brush` | 3.0.0 | ISC |
| `d3-chord` | 3.0.1 | ISC |
| `d3-color` | 3.1.0 | ISC |
| `d3-contour` | 4.0.2 | ISC |
| `d3-delaunay` | 6.0.4 | ISC |
| `d3-dispatch` | 3.0.1 | ISC |
| `d3-drag` | 3.0.0 | ISC |
| `d3-dsv` | 3.0.1 | ISC |
| `d3-ease` | 3.0.1 | BSD-3-Clause |
| `d3-fetch` | 3.0.1 | ISC |
| `d3-force` | 3.0.0 | ISC |
| `d3-format` | 3.1.0 | ISC |
| `d3-geo` | 3.1.1 | ISC |
| `d3-hierarchy` | 3.1.2 | ISC |
| `d3-interpolate` | 3.0.1 | ISC |
| `d3-path` | 1.0.9 | BSD-3-Clause |
| `d3-path` | 3.1.0 | ISC |
| `d3-polygon` | 3.0.1 | ISC |
| `d3-quadtree` | 3.0.1 | ISC |
| `d3-random` | 3.0.1 | ISC |
| `d3-sankey` | 0.12.3 | BSD-3-Clause |
| `d3-scale` | 4.0.2 | ISC |
| `d3-scale-chromatic` | 3.1.0 | ISC |
| `d3-selection` | 3.0.0 | ISC |
| `d3-shape` | 1.3.7 | BSD-3-Clause |
| `d3-shape` | 3.2.0 | ISC |
| `d3-time` | 3.1.0 | ISC |
| `d3-time-format` | 4.1.0 | ISC |
| `d3-timer` | 3.0.1 | ISC |
| `d3-transition` | 3.0.1 | ISC |
| `d3-zoom` | 3.0.0 | ISC |
| `dagre-d3-es` | 7.0.14 | MIT |
| `dayjs` | 1.11.21 | MIT |
| `dompurify` | 3.4.17 | (MPL-2.0 OR Apache-2.0) |
| `elkjs` | 0.12.0 | EPL-2.0 OR GPL-3.0-or-later |
| `es-toolkit` | 1.45.1 | MIT |
| `fastdom` | 1.0.12 | MIT |
| `internmap` | 2.0.3 | ISC |
| `js-yaml` | 4.3.2 | MIT |
| `katex` | 0.18.11 | MIT |
| `khroma` | 2.1.0 | MIT |
| `layout-base` | 1.0.2 | MIT |
| `layout-base` | 2.0.1 | MIT |
| `lodash-es` | 4.18.1 | MIT |
| `marked` | 16.3.0 | MIT |
| `roughjs` | 4.6.6 | MIT |
| `stylis` | 4.3.6 | MIT |
| `ts-dedent` | 2.2.0 | MIT |

## elkjs's source

elkjs (Copyright (c) 2017 Kiel University and others, EPL-2.0) is in the script compiled and minified, not in the
form its authors change it in. Its source is elkjs 0.12.0's, https://github.com/kieler/elkjs/tree/0.12.0 (the tag's
commit ff5771d7165445c42c408bb8a090c8035272218c), with the Eclipse Layout Kernel it is built from,
https://github.com/eclipse-elk/elk (EPL-2.0, the same holders); npm's package is
https://registry.npmjs.org/elkjs/-/elkjs-0.12.0.tgz.
