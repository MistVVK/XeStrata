// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// The launchers' translations, as the web page's (serve/web/i18n.js): the English text sits in the code next to its
/// key, `tr "home.install" "Install a model ..."` or `trf "run.step" "Step {n}: {step}" [ "n", box 2; ... ]`, and
/// every other language is a catalog, locales/<lang>.json beside this file (a flat {"key": "text"} object), built into
/// the assembly.  tools/launcher_i18n.py checks the catalogs against the code: the same keys, the same {placeholders}.
/// What setup.py says (its steps, questions and the models' descriptions) stays as setup writes it.
module Xestrata.Launcher.I18n

open System.Collections.Generic
open System.Text.Json
open System.Text.RegularExpressions

/// [code, the language's own name], the web page's languages.
let languages =
    [ "en", "English"; "de", "Deutsch"; "fr", "Français"; "es", "Español"; "ja", "日本語"; "ko", "한국어"
      "pt", "Português"; "ru", "Русский"; "zh", "中文" ]

/// The language a culture name or a locale asks for (ja-JP and ja_JP.UTF-8 are ja, zh-Hant-TW is zh), English where
/// the launchers have none.
let pick (culture: string) =
    let code = (if isNull culture then "" else culture).Split([| '-'; '_'; '.'; '@' |]).[0].ToLowerInvariant()
    if languages |> List.exists (fun (c, _) -> c = code) then code else "en"

let private load (code: string) =
    let table = Dictionary<string, string>()
    if code <> "en" then
        use s = System.Reflection.Assembly.GetExecutingAssembly().GetManifestResourceStream("locales." + code + ".json")
        if not (isNull s) then
            use doc = JsonDocument.Parse s
            for p in doc.RootElement.EnumerateObject() do
                if p.Value.ValueKind = JsonValueKind.String then table[p.Name] <- p.Value.GetString()
    table

let mutable private lang = pick (Platform.culture Platform.isWindows System.Environment.GetEnvironmentVariable)
let mutable private catalog = load lang

/// The language in use.
let current () = lang

/// Use `culture`'s language (--lang; by default Windows' display language, or the locale on Linux).
let choose (culture: string) =
    lang <- pick culture
    catalog <- load lang

/// The catalog's text for `key`, else the English text in the code.
let tr (key: string) (english: string) =
    match catalog.TryGetValue key with
    | true, t -> t
    | _ -> english

let private slot = Regex(@"\{(\w+)\}", RegexOptions.Compiled)

/// tr with {name} filled from `vars`.
let trf (key: string) (english: string) (vars: (string * obj) list) =
    let m = dict vars
    slot.Replace(tr key english, fun (x: Match) ->
        match m.TryGetValue x.Groups[1].Value with
        | true, v -> string v
        | _ -> x.Value)
