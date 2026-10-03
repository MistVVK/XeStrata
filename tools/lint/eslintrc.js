// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// eslint for the web app's script (tools/lint/run.sh): the recommended rules, for a browser
module.exports = { root: true, env: { browser: true, es2020: true }, parserOptions: { ecmaVersion: 2022 },
                   extends: "eslint:recommended",
                   // flags every await that writes a field it read before; eslint dropped it from the recommended set
                   rules: { "require-atomic-updates": "off" } };
