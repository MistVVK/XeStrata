// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// setup's options as fields, drawn from what --describe says of them (setup.py's OPTIONS): a field of each kind, when
/// it is shown, what it passes back to setup and why a value is refused.  Their text is setup's English, or the
/// catalog's translation where it has setup.NAME.label and the rest (tools/launcher_i18n.py reads OPTIONS for the keys).
module Xestrata.Launcher.Fields

open System
open System.Text.Json
open System.Text.RegularExpressions
open Avalonia.Controls
open Avalonia.LogicalTree
open Xestrata.Launcher.I18n
open Xestrata.Launcher.Protocol
open Xestrata.Launcher.Views

/// A choice's value in a catalog's key: "" is unset, the characters a key does not take are _.
let private valueKey (v: string) = if v = "" then "unset" else Regex.Replace(v, @"[^\w\-]", "_")

let private fill (o: OptionSpec) (text: string) =
    if isNull o.Vars then text
    else Regex.Replace(text, @"\{(\w+)\}", fun m -> match o.Vars.TryGetValue m.Groups[1].Value with | true, v -> v | _ -> m.Value)

/// One of an option's texts (label, hint, takes, about, needed) in the window's language.
let text' (o: OptionSpec) (part: string) (english: string) =
    if String.IsNullOrEmpty english then "" else fill o (tr ($"setup.{o.Name}.{part}") english)

let label (o: OptionSpec) = text' o "label" o.Label

let choiceLabel (o: OptionSpec) (value: string) (english: string) = fill o (tr ($"setup.{o.Name}.choice.{valueKey value}") english)

let private defaultString (o: OptionSpec) =
    match o.Default.ValueKind with
    | JsonValueKind.String -> o.Default.GetString()
    | JsonValueKind.Number -> o.Default.ToString()
    | _ -> ""

let private defaultChecked (o: OptionSpec) =
    match o.Default.ValueKind with
    | JsonValueKind.True -> true
    | JsonValueKind.String -> not (isNull o.Values) && o.Values.Length > 0 && o.Default.GetString() = o.Values[0]
    | _ -> false

/// What the page knows when it shows a field: whether a condition of describe's (vision, kv, ...) holds, the family
/// chosen (its own choices of an option), the GPUs, and how a path is browsed for.
type Page =
    { Holds: string -> bool
      Family: unit -> Family option
      Gpus: Gpu[]
      Browse: string -> string -> TextBox -> unit }

/// A field: its control, its value as setup takes it ("" passes nothing), and its check.
type Field(o: OptionSpec, page: Page) =
    let takes () =
        trf "opt.takes" "{name} takes {what}." [ "name", box ("--" + o.Name); "what", box (text' o "takes" o.Takes) ]
    let box' () = TextBox(PlaceholderText = text' o "hint" o.Hint, MinWidth = (if o.Kind = "path" || o.Kind = "password" || o.Kind = "text" then 420.0 else 260.0))
    let browseButton (target: TextBox) =
        button (tr "common.browse" "Browse ...") (fun () -> page.Browse (if isNull o.Pick then "folder" else o.Pick) (label o) target)
    let gpuChoices () =
        match o.GpuChoices with
        | "pci" -> [ for g in page.Gpus do if not (String.IsNullOrEmpty g.Pci) then yield g.Pci, trf "opt.gpu" "{gpu} ({vram})" [ "gpu", box g.Name; "vram", box (gb g.VramGb) ] ]
        | "index" -> [ for g in page.Gpus -> string g.Index, trf "opt.gpu" "{gpu} ({vram})" [ "gpu", box g.Name; "vram", box (gb g.VramGb) ] ]
        | _ -> []
    // ---- the controls of each kind
    let combo = ComboBox(MinWidth = 260.0)
    let mutable items: (string * string) list = []
    let pathOfChoice = box' ()
    let pathRow = row 8.0 [ pathOfChoice; browseButton pathOfChoice ]
    let textBox = box' ()
    let check = CheckBox(Content = Views.text (label o), IsChecked = Nullable(defaultChecked o))
    let gpuBoxes = [| for g in page.Gpus -> CheckBox(Content = Views.text (sprintf "%s (%s)" g.Name (gb g.VramGb))) |]
    let mutable family: string = null
    /// the choices of the family chosen, and the one offered first
    let setChoices () =
        let own =
            match page.Family () with
            | Some f when not (isNull f.Options) ->
                match f.Options.TryGetValue o.Name with
                | true, x -> Some x
                | _ -> None
            | _ -> None
        let all = [ for c in o.Choices -> c[0], choiceLabel o c[0] c[1] ] @ gpuChoices ()
        items <- (match own with Some x -> all |> List.filter (fun (v, _) -> Array.contains v x.Choices) | None -> all)
        combo.ItemsSource <- [ for _, l in items -> l ]
        let want = match own with Some x -> x.Default | None -> defaultString o
        combo.SelectedIndex <- max 0 (List.tryFindIndex (fun (v, _) -> v = want) items |> Option.defaultValue 0)
    do
        if o.Kind = "choice" then
            setChoices ()
            combo.SelectionChanged.Add(fun _ -> pathRow.IsVisible <- not (isNull o.PathChoice) && combo.SelectedIndex >= 0 && fst items[combo.SelectedIndex] = o.PathChoice)
            pathRow.IsVisible <- false
        if o.Kind = "password" && o.Set then textBox.PlaceholderText <- tr "opt.keySet" "set (type to replace)"
        if o.Kind = "password" then textBox.PasswordChar <- '*'
        if o.Kind = "path" || o.Kind = "text" || o.Kind = "int" || o.Kind = "number" then
            let d = defaultString o
            if d <> "" then textBox.Text <- d
    let about = if String.IsNullOrEmpty o.About then None else Some(muted (text' o "about" o.About))
    let control: Control =
        let withAbout (c: Control) = match about with Some a -> stack 4.0 [ c; a :> Control ] :> Control | None -> c
        match o.Kind with
        | "choice" -> withAbout (field (label o) (stack 4.0 [ combo :> Control; pathRow :> Control ]))
        | "flag" -> withAbout check
        | "gpus" -> field (label o) (stack 4.0 [ for b in gpuBoxes -> b :> Control ])
        | "path" -> withAbout (field (label o) (row 8.0 [ textBox; browseButton textBox ]))
        | _ -> withAbout (field (label o) textBox)

    member _.Spec = o
    member _.Control = control

    /// The family chosen changed: its own choices.
    member _.FamilyChanged() =
        match page.Family () with
        | Some f when o.Kind = "choice" && f.Id <> family ->
            family <- f.Id
            setChoices ()
        | _ -> ()

    /// Its value as setup takes it: "" passes nothing.
    member _.Value =
        let trimmed (b: TextBox) = if isNull b.Text then "" else b.Text.Trim()
        match o.Kind with
        | "choice" ->
            if combo.SelectedIndex < 0 then ""
            else
                let v = fst items[combo.SelectedIndex]
                if not (isNull o.PathChoice) && v = o.PathChoice then trimmed pathOfChoice else v
        | "flag" ->
            let on = check.IsChecked = Nullable true
            if isNull o.Values || o.Values.Length < 2 then (if on then "true" else "") else o.Values[if on then 0 else 1]
        | "gpus" ->
            let picked = [ for i, b in Array.indexed gpuBoxes do if b.IsChecked = Nullable true then string page.Gpus[i].Index ]
            if picked.Length >= 2 then String.Join(",", picked) else ""
        | _ -> trimmed textBox

    /// Whether describe's conditions hold for it now; `valueOf` gives another field's value.
    member _.Shown(valueOf: string -> string) =
        isNull o.When
        || o.When |> Array.forall (fun w ->
            match w.IndexOf "!=", w.IndexOf "=" with
            | i, _ when i > 0 -> valueOf (w.Substring(0, i)) <> w.Substring(i + 2)
            | _, i when i > 0 -> valueOf (w.Substring(0, i)) = w.Substring(i + 1)
            | _ -> page.Holds w)

    /// The words it adds to setup's command line (`name`: the option's name there), or why its value is refused.
    member this.Args(name: string) : Result<string list, string> =
        let v = this.Value
        let number () = Double.TryParse(v, Globalization.NumberStyles.Float, Globalization.CultureInfo.InvariantCulture)
        let inRange (x: float) = (not o.Min.HasValue || x >= o.Min.Value) && (not o.Max.HasValue || x <= o.Max.Value)
        match o.Kind with
        | "password" -> Result.Ok []
        | _ when v = "" ->
            if o.Kind = "choice" && not (isNull o.PathChoice) && combo.SelectedIndex >= 0 && fst items[combo.SelectedIndex] = o.PathChoice then
                Error(takes ())
            else Result.Ok []
        | _ when o.Keep && v = defaultString o -> Result.Ok []
        | "flag" when isNull o.Values || o.Values.Length < 2 -> Result.Ok [ "--" + name ]
        | "int" ->
            match Int32.TryParse v with
            | true, n when inRange (float n) -> Result.Ok [ "--" + name; v ]
            | _ -> Error(takes ())
        | "number" ->
            match number () with
            | true, x when inRange x -> Result.Ok [ "--" + name; v ]
            | _ -> Error(takes ())
        | _ when not (String.IsNullOrEmpty o.Pattern) && not (Regex.IsMatch(v, "^(?:" + o.Pattern + ")$")) -> Error(takes ())
        | _ -> Result.Ok [ "--" + name; v ]

    /// A password: why it cannot be left blank, or None.
    member this.Missing =
        if o.Kind = "password" && this.Value = "" && not o.Set && not (String.IsNullOrEmpty o.Needed) then Some(text' o "needed" o.Needed)
        else None

/// The fields of `place`'s options, in setup's order.
let make (options: OptionSpec[]) (place: string) (page: Page) =
    [ for o in (if isNull options then [||] else options) do if o.Place = place then Field(o, page) ]

/// The fields' controls shown and hidden as their conditions hold now.
let refresh (fields: Field list) =
    let valueOf name = fields |> List.tryFind (fun f -> f.Spec.Name = name) |> Option.map (fun f -> f.Value) |> Option.defaultValue ""
    for f in fields do
        f.FamilyChanged()
    for f in fields do
        f.Control.IsVisible <- f.Shown valueOf

/// The shown fields' words for setup's command line in order (`rename`: an option's name there), or the first reason
/// one is refused.
let args (fields: Field list) (rename: string -> string) : Result<string list, string> =
    fields
    |> List.filter (fun f -> f.Control.IsVisible)
    |> List.fold (fun acc f ->
        match acc, f.Missing, f.Args(rename f.Spec.Name) with
        | Error e, _, _ -> Error e
        | _, Some why, _ -> Error why
        | Result.Ok a, None, Result.Ok b -> Result.Ok(a @ b)
        | _, None, Error e -> Error e) (Result.Ok [])

/// The shown passwords' environment variables and their values (null: unset), for setup's environment: a key never
/// goes on a command line, which process lists show.
let environment (fields: Field list) =
    [ for f in fields do
          if f.Spec.Kind = "password" && not (String.IsNullOrEmpty f.Spec.Env) then
              f.Spec.Env, (if f.Control.IsVisible && f.Value <> "" then f.Value else null) ]

/// Each field's change refreshes them all (a condition on another field's value).
let watch (fields: Field list) (changed: unit -> unit) =
    for f in fields do
        for c in Seq.append [ f.Control :> ILogical ] (f.Control.GetLogicalDescendants()) do
            match c with
            | :? ComboBox as b -> b.SelectionChanged.Add(fun _ -> changed ())
            | :? CheckBox as b -> b.IsCheckedChanged.Add(fun _ -> changed ())
            | _ -> ()
