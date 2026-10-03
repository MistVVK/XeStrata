# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# markdownlint (mdl) style for XeStrata's Markdown (tools/lint/run.sh): every rule but the line length (MD013);
# the documents wrap at about 120 characters, and tables and links run longer
# MD041 and MD002 (a first-level heading on the first line) do not hold for files that start with their SPDX comment
all
exclude_rule 'MD013'
exclude_rule 'MD041'
exclude_rule 'MD002'
