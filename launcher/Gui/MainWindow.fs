// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// The launcher's window: this PC and the installed models (setup.py --describe), the choice of a model to install,
/// the install's progress and questions, and the running model.  setup.py does the work; the window shows it.
module Xestrata.Launcher.MainWindow

open System
open System.Diagnostics
open System.Text.RegularExpressions
open Avalonia
open Avalonia.Controls
open Avalonia.Layout
open Avalonia.Platform.Storage
open Avalonia.Threading
open Xestrata.Launcher.I18n
open Xestrata.Launcher.Protocol
open Xestrata.Launcher.Views

let private ui (f: unit -> unit) = Dispatcher.UIThread.Post(fun () -> f ())

let private openUrl (url: string) =
    Process.Start(ProcessStartInfo(url, UseShellExecute = true)) |> ignore

let private readyUrl = Regex(@"ready: (http://\S+?)(/v1)?\s", RegexOptions.Compiled)

let private trimmed (box: TextBox) = if isNull box.Text then "" else box.Text.Trim()

let private isInt (least: int) (t: string) =
    match Int32.TryParse t with
    | true, n -> n >= least
    | _ -> false

let private isNumber (least: float) (most: float) (t: string) =
    match Double.TryParse(t, Globalization.NumberStyles.Float, Globalization.CultureInfo.InvariantCulture) with
    | true, x -> x >= least && x <= most
    | _ -> false

/// A setup option from a text box: blank gives nothing, else `--name text` once `ok` takes the text (else why not).
let private optionOf (name: string) (box: TextBox) (ok: string -> bool) (takes: string) : Result<string list, string> =
    match trimmed box with
    | "" -> Result.Ok []
    | t when ok t -> Result.Ok [ name; t ]
    | _ -> Error(trf "opt.takes" "{name} takes {what}." [ "name", name :> obj; "what", takes :> obj ])

/// The options' words in order, or the first reason one is refused.
let private collectArgs (parts: Result<string list, string> list) =
    parts |> List.fold (fun acc p -> match acc, p with Result.Ok a, Result.Ok b -> Result.Ok(a @ b) | Error e, _ | _, Error e -> Error e) (Result.Ok [])

let private flag (name: string) (box: CheckBox) = if box.IsChecked = Nullable true then [ name ] else []

/// The log of a run, its last lines kept.
type private LogView() =
    let box =
        TextBox(
            IsReadOnly = true,
            AcceptsReturn = true,
            TextWrapping = Media.TextWrapping.NoWrap,
            FontFamily = Media.FontFamily "Cascadia Mono, Consolas, monospace",
            FontSize = 12.0,
            Height = 220.0
        )
    let lines = Collections.Generic.Queue<string>()

    member _.Control = box

    member _.Add(line: string) =
        lines.Enqueue line
        while lines.Count > 400 do
            lines.Dequeue() |> ignore
        box.Text <- String.Join("\n", lines)
        box.CaretIndex <- box.Text.Length

/// `setup`: how the window reaches setup (Setup.real; the launcher's tests give canned runs instead).
type MainWindow(setup: Setup.Access) as this =
    inherit Window(Title = "XeStrata", Width = 980.0, Height = 720.0, MinWidth = 720.0, MinHeight = 520.0)

    let host = ContentControl(Margin = Thickness(24.0, 16.0))
    let version = muted ""
    let updateBar = Border(IsVisible = false, Background = accentTint, CornerRadius = CornerRadius 6.0, Padding = Thickness 10.0)
    let mutable running: Setup.IRun option = None
    // setup's warnings while it checked this PC: why a GPU is missing ("the engine could not list the GPUs: ...", #41)
    let mutable checkWarnings: string list = []
    // a start that found its model already served (setup's attach_running): the address and whether the systemd user
    // service serves it; the window then neither stops nor waits for that server
    let mutable attached: (string * bool) option = None

    let show (content: Control) =
        host.Content <- ScrollViewer(Content = content, HorizontalScrollBarVisibility = Primitives.ScrollBarVisibility.Disabled)

    let failure (title: string) (detail: string) (back: unit -> unit) =
        show (
            stack 14.0 [
                heading title
                card (coloured failBrush detail)
                actions [ accentButton (tr "common.back" "Back") back ]
            ]
        )

    let pickFolder (title: string) (target: TextBox) =
        task {
            let top = TopLevel.GetTopLevel this
            let! picked = top.StorageProvider.OpenFolderPickerAsync(FolderPickerOpenOptions(Title = title, AllowMultiple = false))
            if picked.Count > 0 then
                match picked[0].TryGetLocalPath() with
                | null -> ()
                | path -> target.Text <- path
        }
        |> ignore

    let pickFile (title: string) (target: TextBox) =
        task {
            let top = TopLevel.GetTopLevel this
            let gguf = FilePickerFileType(tr "app.ggufFiles" "GGUF model files", Patterns = [| "*.gguf" |])
            let! picked =
                top.StorageProvider.OpenFilePickerAsync(
                    FilePickerOpenOptions(Title = title, AllowMultiple = false, FileTypeFilter = [| gguf |])
                )
            if picked.Count > 0 then
                match picked[0].TryGetLocalPath() with
                | null -> ()
                | path -> target.Text <- path
        }
        |> ignore

    // ---------------------------------------------------------------------------------------------- the update
    let checkUpdate () =
        task {
            let! messages, _ = setup.Collect [ "--check-update" ]
            for m in messages do
                match m with
                | Updated u when u.Newer ->
                    ui (fun () ->
                        updateBar.Child <-
                            stack 6.0 [
                                subheading (trf "update.out" "XeStrata {latest} is out (this is {current})" [ "latest", box u.Latest; "current", box u.Current ])
                                text (trf "update.how" "To update: {how}." [ "how", box u.How ])
                                actions [ button (tr "update.releasePage" "Release page") (fun () -> openUrl u.Url) ]
                            ]
                        updateBar.IsVisible <- true)
                | _ -> ()
        }
        |> ignore

    // ---------------------------------------------------------------------------------------------- a run of setup
    /// setup.py (or `script`, Setup.Run's) with `args`, its steps, progress, questions and log on a page of its own:
    /// `answer` answers a question the window has already asked (else the page asks it), `onReady` gets the server's
    /// address once it says so, `onDone` the exit code (after a failure: when the page is left).
    let runScriptPage (script: string) (title: string) (args: string list) (answer: string -> string option)
                      (onReady: string -> unit) (onDone: int -> unit) =
        let stepText = subheading (tr "setup.starting" "Starting ...")
        let status = text ""
        let bar = ProgressBar(IsIndeterminate = true, Minimum = 0.0, Maximum = 1.0, Height = 6.0)
        let warnings = stack 4.0 []
        let question = stack 8.0 []
        question.IsVisible <- false
        let log = LogView()
        let recent = Collections.Generic.List<string>()   // the says since the last question: a question's menu
        let details = Expander(Header = (tr "setup.details" "Details"), Content = log.Control, HorizontalAlignment = HorizontalAlignment.Stretch)
        let mutable run: Setup.IRun option = None
        let mutable failed = false
        let cancel = button (tr "setup.cancel" "Cancel") (fun () -> run |> Option.iter (fun r -> r.Stop()))
        let mutable readySeen = false
        let line (s: string) =
            log.Add s
            if not readySeen then
                let m = readyUrl.Match(s + " ")
                if m.Success then
                    readySeen <- true
                    onReady m.Groups[1].Value
        let onMessage (m: Message) =
            ui (fun () ->
                match m with
                | Step(n, t) ->
                    stepText.Text <- trf "setup.step" "Step {n}: {step}" [ "n", box n; "step", box t ]
                    bar.IsIndeterminate <- true
                    recent.Clear()
                    log.Add("=== " + t)
                | Say s ->
                    status.Text <- s.Trim()
                    if s.Trim() <> "" then recent.Add s
                    line s
                | Ok s ->
                    status.Text <- "✓ " + s
                    line ("[ok] " + s)
                | Warn s ->
                    warnings.Children.Add(coloured warnBrush ("⚠ " + s))
                    line ("[!] " + s)
                | Fail(s, hint) ->
                    failed <- true
                    failure (tr "setup.stopped" "Setup stopped") (if String.IsNullOrEmpty hint then s else s + "\n\n" + hint) (fun () -> onDone 1)
                | Progress(what, d, total) ->
                    if total > 0L then
                        bar.IsIndeterminate <- false
                        bar.Value <- float d / float total
                        status.Text <- trf "setup.progressOf" "{what}: {done} of {total}" [ "what", box what; "done", box (gb (float d / 1e9)); "total", box (gb (float total / 1e9)) ]
                    else
                        status.Text <- trf "setup.progress" "{what}: {done}" [ "what", box what; "done", box (gb (float d / 1e9)) ]
                | Log s
                | Text s -> line s
                | Ask(q, _, _) when (answer q).IsSome -> run |> Option.iter (fun r -> r.Answer (answer q).Value)
                | Ask(q, choices, def) ->
                    question.Children.Clear()
                    question.Children.Add(subheading q)
                    for r in recent do
                        question.Children.Add(text (r.TrimEnd()))
                    let buttons =
                        [ for c in choices ->
                              let answer () =
                                  question.IsVisible <- false
                                  recent.Clear()
                                  run |> Option.iter (fun r -> r.Answer c)
                              (if c = def then accentButton c answer else button c answer) :> Control ]
                    question.Children.Add(row 8.0 buttons)
                    question.IsVisible <- true
                | Attached(url, service) -> attached <- Some(url, service)
                | Described _
                | Updated _ -> ())
        show (
            stack 14.0 [
                heading title
                stepText
                bar
                status
                warnings
                card question
                details
                actions [ cancel ]
            ]
        )
        // A child that ends with a code but no Fail message (a server that would not start, an uncaught exception)
        // must not disappear into the home page: show the code and let the person go back when they have read it.
        let onExit code =
            ui (fun () ->
                if not failed && code <> 0 then
                    failed <- true
                    failure (tr "setup.stopped" "Setup stopped")
                        (trf "home.setupExited" "setup.py exited with {code}." [ "code", box code ])
                        (fun () -> onDone code)
                else
                    onDone code)
        let r = setup.Start script args onMessage onExit
        run <- Some r
        r

    let runPage = runScriptPage Places.setupPy

    /// Image picker. Setup supplies the family, version, origin and per-size estimates.
    let imagePicker (d: Describe) (current: string[]) (ramLeft: unit -> float) (changed: unit -> unit) =
        let mutable models = d.Images
        let mutable chosen = List.ofArray current
        let family = ComboBox(MinWidth = 320.0)
        let version = ComboBox(MinWidth = 320.0)
        let encoder = ComboBox(MinWidth = 320.0)
        encoder.ItemsSource <- [| tr "imgpick.removed" "Refusal direction removed (-uncensored)"; tr "imgpick.original" "Original" |]
        encoder.SelectedIndex <- 0
        let sizes = ListBox(MinWidth = 420.0)
        sizes.ItemTemplate <-
            itemTemplate (fun (label: string) ->
                let warning = tr "install.doesNotFitThe" "does not fit the RAM"
                let suffix = " · " + warning
                let doesNotFit = label.EndsWith suffix
                row 8.0 [
                    yield text (if doesNotFit then label.Substring(0, label.Length - suffix.Length) else label)
                    if label.StartsWith(tr "imgpick.auto" "Auto (recommended; largest that fits RAM)") then
                        yield badge okBrush (tr "install.recommended" "recommended")
                    if doesNotFit then yield badge failBrush warning
                ])
        let info = TextBlock(TextWrapping = Media.TextWrapping.Wrap)
        let origin = muted ""
        let license = muted ""
        let duplicate = muted ""
        let rows = StackPanel(Spacing = 6.0)
        let total = muted ""
        let filePath = TextBox(MinWidth = 320.0, PlaceholderText = tr "imgpick.filePath" "Anima .safetensors file")
        let fileStatus = coloured failBrush ""
        let ownBox = StackPanel(Spacing = 6.0, IsVisible = false)
        let versionField = field (tr "imgpick.version" "Version") version
        let encoderField = field (tr "imgpick.encoder" "Text encoder") encoder
        let mutable updating = false
        let selectedFamily () = match family.SelectedItem with :? string as x -> x | _ -> ""
        let familyModels () = models |> Array.filter (fun m -> m.Family = selectedFamily ())
        let selected () =
            let ms = familyModels ()
            if version.SelectedIndex >= 0 && version.SelectedIndex < ms.Length then Some ms[version.SelectedIndex] else None
        let choiceModel (choice: string) =
            models
            |> Array.sortByDescending (fun m -> m.Id.Length)
            |> Array.tryFind (fun m -> choice = m.Id || choice.StartsWith(m.Id + ":") || choice.StartsWith(m.Id + "-uncensored"))
        let choiceSize (m: ImageModel) (choice: string) =
            let suffix = choice.Substring(m.Id.Length).Replace("-uncensored", "")
            if suffix.StartsWith ":" then suffix.Substring 1 else ""
        let estimates (m: ImageModel) (unc: bool) = if m.Uncensored && unc then m.UncensoredSizes else m.Sizes
        let sizeFor (m: ImageModel) (choice: string) =
            let size = choiceSize m choice
            let ss = estimates m (choice.Substring(m.Id.Length).StartsWith "-uncensored")
            if size <> "" then ss |> Array.find (fun s -> s.Id = size)
            else ss |> Array.tryFind (fun s -> s.RamGb <= ramLeft ()) |> Option.defaultValue (Array.last ss)
        let choiceName () =
            selected () |> Option.map (fun m ->
                m.Id + (if m.Uncensored && encoder.SelectedIndex = 0 then "-uncensored" else "")
                + (if sizes.SelectedIndex > 0 then ":" + m.Sizes[sizes.SelectedIndex - 1].Id else ""))
        let key (v: string) =
            match choiceModel v with
            | Some m -> m.Id + (if v.Substring(m.Id.Length).StartsWith "-uncensored" then "-uncensored" else "")
            | None -> v
        let mutable refresh = ignore
        let add = accentButton (tr "imgpick.add" "Add to selection") (fun () ->
            match choiceName () with
            | Some v when not (chosen |> List.exists (fun x -> key x = key v)) ->
                chosen <- chosen @ [ v ]
                refresh ()
            | _ -> ())
        let columns (cells: Control list) =
            let grid = Grid(ColumnDefinitions = ColumnDefinitions("80,2*,*,2*,*,*,45"), ColumnSpacing = 6.0)
            for i, cell in List.indexed cells do
                Grid.SetColumn(cell, i)
                grid.Children.Add cell
            grid
        let drawRows () =
            rows.Children.Clear()
            rows.Children.Add(columns [ muted (tr "imgpick.default" "Default"); muted (tr "imgpick.modelVersion" "Model / version");
                                        muted (tr "imgpick.size" "Size"); muted (tr "imgpick.encoder" "Text encoder");
                                        muted (tr "imgpick.download" "Download"); muted "RAM"; muted "" ])
            let mutable download = 0.0
            for i, choice in List.indexed chosen do
                match choiceModel choice with
                | Some m ->
                    let s = sizeFor m choice
                    download <- download + s.DownloadGb
                    let def = RadioButton(GroupName = "image-default", IsChecked = Nullable<bool>((i = 0)))
                    def.IsCheckedChanged.Add(fun _ ->
                        if def.IsChecked = Nullable true && i <> 0 then
                            chosen <- choice :: (chosen |> List.filter ((<>) choice))
                            refresh ())
                    let remove = button "×" (fun () -> chosen <- chosen |> List.filter ((<>) choice); refresh ())
                    remove.MinWidth <- 0.0
                    let encoderName =
                        if not m.Uncensored then "—"
                        elif choice.Substring(m.Id.Length).StartsWith "-uncensored" then tr "imgpick.removed" "Refusal direction removed (-uncensored)"
                        else tr "imgpick.original" "Original"
                    rows.Children.Add(columns [ def; text m.Title;
                                                text (if choiceSize m choice = "" then tr "img.auto" "Auto" + " (" + s.Id + ")" else s.Id);
                                                muted encoderName; muted (gb s.DownloadGb); muted (gb s.RamGb); remove ])
                | None -> rows.Children.Add(muted choice)
            total.Text <- trf "imgpick.total" "Total download: {download}" [ "download", box (gb download) ]
        refresh <- fun () ->
            if not updating then
                updating <- true
                match selected () with
                | Some m ->
                    let ms = familyModels ()
                    versionField.IsVisible <- ms.Length > 1
                    encoderField.IsVisible <- m.Uncensored
                    info.Text <- m.Title + " — " + m.About + (if m.Edits then "" else " · " + tr "img.noEdits" "no edits")
                    origin.Text <- trf "imgpick.origin" "Model files: {origin}" [ "origin", box (tr ("imgpick.origin." + m.OriginKey) m.Origin) ]
                    license.Text <- m.License
                    let prev = sizes.SelectedIndex
                    let ss = estimates m (encoder.SelectedIndex = 0)
                    let auto = ss |> Array.tryFind (fun s -> s.RamGb <= ramLeft ()) |> Option.defaultValue (Array.last ss)
                    let labels =
                        [| yield tr "imgpick.auto" "Auto (recommended; largest that fits RAM)" + " — " + auto.Id + " · " + gb auto.DownloadGb + ", RAM " + gb auto.RamGb
                                 + (if auto.RamGb > ramLeft () then " · " + tr "install.doesNotFitThe" "does not fit the RAM" else "")
                           for s in ss do
                               yield s.Id + " — " + gb s.DownloadGb + ", RAM " + gb s.RamGb
                                     + (if s.RamGb > ramLeft () then " · " + tr "install.doesNotFitThe" "does not fit the RAM" else "") |]
                    sizes.ItemsSource <- labels
                    sizes.SelectedIndex <- if prev >= 0 && prev < labels.Length then prev else 0
                    let isDuplicate = choiceName () |> Option.exists (fun c -> chosen |> List.exists (fun x -> key x = key c))
                    add.IsEnabled <- not isDuplicate
                    duplicate.Text <- if isDuplicate then tr "imgpick.duplicate" "This model and text encoder are already selected." else ""
                | None -> ()
                drawRows ()
                updating <- false
                changed ()
        let setFamily () =
            if not updating then
                updating <- true
                version.ItemsSource <- familyModels () |> Array.map (fun m ->
                    if m.Own then trf "imgpick.ownVersion" "Your own: {name}" [ "name", box (m.Id.Substring(m.Id.IndexOf(':') + 1)) ]
                    else m.Version)
                version.SelectedIndex <- 0
                sizes.SelectedIndex <- 0
                encoder.SelectedIndex <- 0
                updating <- false
                refresh ()
        family.SelectionChanged.Add(fun _ -> setFamily ())
        version.SelectionChanged.Add(fun _ ->
            if not updating then
                updating <- true
                sizes.SelectedIndex <- 0
                encoder.SelectedIndex <- 0
                updating <- false
                refresh ())
        encoder.SelectionChanged.Add(fun _ -> refresh ())
        sizes.SelectionChanged.Add(fun _ -> refresh ())
        let browse = button (tr "imgpick.browse" "Browse …") (fun () ->
            task {
                let top = TopLevel.GetTopLevel this
                let kind = FilePickerFileType("safetensors", Patterns = [| "*.safetensors" |])
                let! files = top.StorageProvider.OpenFilePickerAsync(FilePickerOpenOptions(Title = tr "imgpick.file" "Add from a file on this PC …", AllowMultiple = false, FileTypeFilter = [| kind |]))
                if files.Count > 0 then
                    match files[0].TryGetLocalPath() with
                    | null -> ()
                    | path -> filePath.Text <- path
            } |> ignore)
        let import = accentButton (tr "imgpick.import" "Add as an Anima version") (fun () ->
            task {
                let path = trimmed filePath
                let! messages, code = setup.Collect [ "--import-anima"; path ]
                if code <> 0 then
                    ui (fun () -> fileStatus.Text <- messages |> List.choose (function Log x | Text x -> Some x | _ -> None) |> String.concat "\n")
                else
                    let! descriptions, _ = setup.Collect [ "--describe" ]
                    ui (fun () ->
                        match descriptions |> List.tryPick (function Described x -> Some x | _ -> None) with
                        | Some updated ->
                            models <- updated.Images
                            let id = "anima:" + IO.Path.GetFileNameWithoutExtension path
                            match models |> Array.tryFind (fun m -> m.Id = id) with
                            | Some m ->
                                family.ItemsSource <- models |> Array.map (fun x -> x.Family) |> Array.distinct
                                family.SelectedItem <- m.Family
                                setFamily ()
                                version.SelectedIndex <- familyModels () |> Array.findIndex (fun x -> x.Id = id)
                                ownBox.IsVisible <- false
                                fileStatus.Text <- ""
                                refresh ()
                            | None -> fileStatus.Text <- tr "imgpick.notAdded" "The imported file was not found."
                        | None -> fileStatus.Text <- tr "imgpick.notAdded" "The imported file was not found.")
            } |> ignore)
        ownBox.Children.AddRange [| muted (tr "imgpick.fileHelp" "Only a DiT made from Anima can be added here. Setup checks its header and copies it to the Anima folder."); row 8.0 [ filePath; browse; import ]; fileStatus |]
        let ownButton = button (tr "imgpick.file" "Add from a file on this PC …") (fun () -> ownBox.IsVisible <- not ownBox.IsVisible)
        family.ItemsSource <- models |> Array.map (fun m -> m.Family) |> Array.distinct
        family.SelectedIndex <- 0
        setFamily ()
        let value () = if List.isEmpty chosen then None else Some(String.concat "," chosen)
        let control = stack 8.0 [ field (tr "imgpick.model" "Model") (row 8.0 [ family; ownButton ]); ownBox; versionField; encoderField;
                                  card (stack 4.0 [ info; origin ]); field (tr "imgpick.size" "Size") sizes; license; duplicate; add;
                                  subheading (tr "imgpick.chosen" "Selected image models"); rows; total ]
        control, value, refresh

    // ---------------------------------------------------------------------------------------------- the pages
    let rec home () =
        show (stack 10.0 [ heading (tr "home.checkingThisPc" "Checking this PC ..."); ProgressBar(IsIndeterminate = true) ])
        task {
            let! messages, code = setup.Collect [ "--describe" ]
            let found = messages |> List.tryPick (function Described d -> Some d | _ -> None)
            checkWarnings <- messages |> List.choose (function Warn s -> Some s | _ -> None)
            ui (fun () ->
                match found with
                | Some d -> homePage d
                | None ->
                    let why =
                        messages
                        |> List.choose (function Fail(t, h) -> Some(t + " " + h) | Text t | Log t -> Some t | _ -> None)
                        |> String.concat "\n"
                    failure (tr "home.checkFailed" "XeStrata could not check this PC") (trf "home.setupExited" "setup.py exited with {code}." [ "code", box code ] + "\n" + why) home)
        }
        |> ignore

    and homePage (d: Describe) =
        version.Text <- "xe" + d.Version
        let gpu =
            if d.Gpu.HasValue then
                d.Gpus |> Array.tryFind (fun g -> g.Index = d.Gpu.Value)
            else
                None
        let pc =
            stack 6.0 [
                yield subheading (tr "home.thisPc" "This PC")
                match gpu with
                | Some g -> yield text (trf "home.gpu" "GPU: {name}, {vram} VRAM" [ "name", box g.Name; "vram", box (gb g.VramGb) ])
                | None -> yield coloured failBrush (tr "home.noGpuHereCan" "No GPU here can run XeStrata.")
                for g in d.Gpus do
                    if not (isNull g.Problem) then
                        yield coloured warnBrush (sprintf "%s: %s" g.Name g.Problem)
                for w in checkWarnings do
                    yield coloured warnBrush ("⚠ " + w)
                yield text (trf "home.ram" "RAM: {ram}" [ "ram", box (gb d.RamGb) ])
                yield
                    text (
                        trf "home.cpu" "CPU: {name}{isa}" [ "name", box d.Cpu.Name
                                                            "isa", box (if d.Cpu.Avx512 then " (AVX-512)" elif d.Cpu.Avx2 then " (AVX2)" else tr "home.noAvx2" " (no AVX2: not supported)") ]
                    )
            ]
        let installed =
            [ for i, m in Array.indexed d.Installed ->
                  card (
                      let info =
                          stack 2.0 [
                              subheading m.ModelName
                              muted (
                                  (if m.Context.HasValue then trf "home.context" "{tokens} of context, " [ "tokens", box (contextLabel m.Context.Value) ] else "")
                                  + sprintf "http://127.0.0.1:%d/" m.Port
                                  + (if m.Images.Length > 0 then trf "home.imagesList" ", images: {list}" [ "list", box (String.Join(", ", m.Images)) ] else "")
                              )
                          ]
                      let imagesOk = d.Families |> Array.exists (fun f -> f.Id = m.Family && f.Images)
                      let buttons =
                          actions [
                              if imagesOk && d.Images.Length > 0 then
                                  yield button (tr "home.images" "Images ...") (fun () -> imagesPage d m)
                              yield button (tr "home.measure" "Measure ...") (fun () -> benchPage d m)
                              yield button (tr "home.tune" "Tune") (fun () -> calibrateModel i m)
                              yield button (tr "home.startWithOptions" "Start with options ...") (fun () -> startOptionsPage d i m)
                              yield accentButton (tr "home.start" "Start") (fun () -> startModel i m [] false)
                          ]
                      stack 8.0 [ info; buttons ]
                  )
                  :> Control ]
        show (
            stack 14.0 [
                yield card pc :> Control
                if not installed.IsEmpty then
                    yield subheading (tr "home.installedModels" "Installed models")
                    yield! installed
                yield
                    actions [
                        button (tr "home.inspectAModelFile" "Inspect a model file ...") (fun () -> inspectPage d)
                        button (tr "home.removeXestrataSData" "Remove XeStrata's data ...") removeData
                        button (tr "home.installFromFilesOn" "Install from files on this PC ...") (fun () -> ownFilesPage d)
                        (if installed.IsEmpty then accentButton else button) (tr "home.install" "Install a model ...") (fun () -> choosePage d None)
                    ]
            ]
        )

    /// A model in GGUF files already on this PC (setup --gguf): setup reads their headers (--describe --gguf) and the
    /// install page then asks setup's questions for that model; the engine's verdict comes at the install.
    and ownFilesPage (d: Describe) =
        let file = TextBox(PlaceholderText = (tr "own.theModelSGguf" "the model's .gguf file (the first one of a split model)"), MinWidth = 420.0)
        let name = TextBox(PlaceholderText = (tr "own.optionalANameFor" "(optional) a name for it; default: the file's"), MinWidth = 260.0)
        let problem = coloured failBrush ""
        let go = accentButton (tr "own.next" "Next") ignore
        go.Click.Add(fun _ ->
            match trimmed file with
            | "" -> problem.Text <- tr "own.choose" "Choose the model's .gguf file."
            | path ->
                go.IsEnabled <- false
                problem.Text <- ""
                task {
                    let! messages, _ =
                        setup.Collect ([ "--describe"; "--gguf"; path ] @ (match trimmed name with "" -> [] | n -> [ "--name"; n ]))
                    let found = messages |> List.tryPick (function Described d when not (isNull (box d.Custom)) -> Some d | _ -> None)
                    ui (fun () ->
                        go.IsEnabled <- true
                        match found with
                        | Some own -> choosePage own (Some(path, trimmed name))
                        | None ->
                            problem.Text <-
                                messages
                                |> List.choose (function Fail(t, h) -> Some(if String.IsNullOrEmpty h then t else t + " " + h) | _ -> None)
                                |> String.concat "\n"
                            if problem.Text = "" then problem.Text <- tr "own.unreadable" "setup could not read this file.")
                }
                |> ignore)
        show (
            stack 14.0 [
                heading (tr "own.installFromFilesOn" "Install from files on this PC")
                muted (tr "own.aModelInGguf" "A model in GGUF files you already have, nothing downloaded: Qwen3.8-Flash-Next, or a model of the architectures XeStrata's other models have (Qwen3.5 and Qwen3.6 MoE, Qwen3.5 dense, Qwen3 MoE, GLM-5.3). Setup reads the file's headers, and the engine checks that it runs its types before anything is installed.")
                field (tr "own.theModelFile" "The model file") (row 8.0 [ file; button (tr "common.browse" "Browse ...") (fun () -> pickFile (tr "own.theModelSGguf2" "The model's GGUF file") file) ])
                field (tr "own.nameItIsInstalled" "Name (it is installed as custom-NAME)") name
                problem
                actions [ button (tr "common.back" "Back") (fun () -> homePage d); go ]
            ]
        )

    /// A page's fields of setup's options (describe's "options") at `places`, shown as `holds` says.
    and fieldsOf (d: Describe) (places: string list) (holds: string -> bool) (family: unit -> Family option) =
        let page: Fields.Page =
            { Holds = holds
              Family = family
              Gpus = d.Gpus |> Array.filter (fun g -> isNull g.Problem)
              Browse = fun kind title target -> if kind = "file" then pickFile title target else pickFolder title target }
        [ for p in places -> p, Fields.make d.Options p page ]

    /// The install page; `own`: the model in the user's own files (--gguf PATH, --name NAME) instead of a family's.
    and choosePage (d: Describe) (own: (string * string) option) =
        let pool =
            if own.IsSome then d.Families |> Array.filter (fun f -> f.Id = "custom")
            else d.Families |> Array.filter (fun f -> f.Id <> "custom")
        // the lists' levels (setup's TIERS): the model, its version, its quantization; from a setup without them, each
        // family a model of its own
        let baseOf (f: Family) = if String.IsNullOrEmpty f.Base then f.Id else f.Base
        let variantOf (f: Family) = if isNull f.Variant then "" else f.Variant
        let quantOf (f: Family) (m: Model) =
            if not (String.IsNullOrEmpty m.Quantizer) then m.Quantizer elif isNull f.Quantizer then "" else f.Quantizer
        let named (table: Collections.Generic.Dictionary<string, string>) (id: string) =
            if isNull table then id else match table.TryGetValue id with | true, t -> t | _ -> id
        let tiers = if isNull d.Bases then [||] else d.Bases
        let bases =
            pool |> Array.map baseOf |> Array.distinct
            |> Array.sortBy (fun b -> match Array.tryFindIndex (fun (t: Tier) -> t.Id = b) tiers with Some i -> i | None -> tiers.Length)
        let baseTitle b =
            match tiers |> Array.tryFind (fun t -> t.Id = b) with
            | Some t -> t.Title
            | None -> (pool |> Array.find (fun f -> baseOf f = b)).Title
        let recFamily = pool |> Array.tryFind (fun f -> f.Id = d.RecommendedFamily) |> Option.defaultValue pool[0]
        let recQuant (f: Family) = f.Models |> Array.tryFind (fun m -> m.Id = f.RecommendedSize) |> Option.map (quantOf f)
        let baseBox = ComboBox(ItemsSource = [ for b in bases -> baseTitle b ], MinWidth = 320.0)
        let variantBox = ComboBox(MinWidth = 320.0)
        let quantBox = ComboBox(MinWidth = 320.0)
        let variantField = field (tr "install.version" "Version: the original, or whose changed copy") variantBox
        let quantField = field (tr "install.quantization" "Quantization: whose model files") quantBox
        let familyInfo = stack 2.0 []
        let mutable variants: string[] = [||]
        let mutable quants: (string * Family)[] = [||]
        let mutable current: Family option = None
        let sizes = ListBox(MaxHeight = 320.0)
        sizes.ItemTemplate <-
            itemTemplate (fun (m: Model) ->
                stack 2.0 [
                    row 6.0 [
                        yield subheading m.Id
                        if m.Recommended then yield badge okBrush (tr "install.recommended" "recommended")
                        if m.Experimental then yield badge warnBrush (tr "common.experimental" "experimental")
                        match m.Fits with
                        | "low-ram" -> yield badge warnBrush (tr "install.lowRamMode" "low-RAM mode")
                        | "budget" -> yield badge warnBrush (tr "install.partFromTheSsd" "part from the SSD")
                        | "slow" | "no" -> yield badge failBrush (tr "install.doesNotFitThe" "does not fit the RAM")
                        | _ -> ()
                    ]
                    muted (trf "install.sizeAbout" "{about}; download {download}, needs ~{ram} of RAM" [ "about", box m.About; "download", box (gbTenth m.DownloadGb); "ram", box (gb m.RamGb) ])
                ])
        let contexts = ComboBox(MinWidth = 160.0)
        let license = muted ""
        let install = accentButton (tr "install.install" "Install") ignore
        let problem = coloured failBrush ""
        let selectedModel () =
            match sizes.SelectedItem with
            | :? Model as m -> Some m
            | _ -> None
        let selectedContext (f: Family) = f.Contexts[max 0 contexts.SelectedIndex]
        let mutable imagesChosen = false
        let mutable refreshFields = ignore
        let images, imagesValue, refreshImages = imagePicker d d.ImageDefault (fun () -> match selectedModel () with Some m -> max 0.0 (d.RamGb - m.RamGb) | None -> d.RamGb) (fun () -> refreshFields ())
        let imagesField = field (tr "install.imageGenerationModelsTo" "Image generation: models to make images with, besides the text model") images
        // what setup's options are shown for: the family, size and context chosen (step 2's questions where setup
        // would ask them)
        let holds (w: string) =
            match current, selectedModel () with
            | Some f, Some m ->
                let ctx = selectedContext f
                match w with
                | "vision" -> m.Vision
                | "kv" -> (f.Kv && ctx > 8192) || (not f.Kv && f.KvTakes.Length > 1)
                | "long_context" -> ctx > 262144
                | "speed_projection" -> f.SpeedProjection
                | "budget" -> m.Budget
                | "images" -> f.Images && d.Images.Length > 0 && imagesChosen
                | "multi_gpu" -> (d.Gpus |> Array.filter (fun g -> isNull g.Problem)).Length > 1
                | "download" -> own.IsNone
                | "sam3" -> not (isNull (box d.Sam3)) && not d.Sam3.Agreed
                | _ -> false
            | _ -> false
        let placed = fieldsOf d [ "install"; "advanced"; "images"; "start" ] holds (fun () -> current)
        let all = placed |> List.collect snd
        let at place = placed |> List.find (fst >> (=) place) |> snd
        let refresh () =
            match current, selectedModel () with
            | Some f, Some _ ->
                imagesField.IsVisible <- f.Images && d.Images.Length > 0
                imagesChosen <- imagesField.IsVisible && (imagesValue ()).IsSome
                Fields.refresh all
            | _ -> ()
        refreshFields <- refresh
        Fields.watch all refresh
        // the quantization chosen: its family, and that quantization's sizes with its recommended one chosen
        let familyChanged () =
            if quantBox.SelectedIndex >= 0 then
                let q, f = quants[quantBox.SelectedIndex]
                current <- Some f
                let models = f.Models |> Array.filter (fun m -> quantOf f m = q)
                familyInfo.Children.Clear()
                familyInfo.Children.AddRange [
                    row 6.0 [
                        subheading f.Title
                        if f.Experimental then badge warnBrush (tr "common.experimental" "experimental")
                    ] :> Control
                    muted (f.By + " - " + f.About) :> Control
                ]
                sizes.ItemsSource <- models
                sizes.SelectedItem <- (models |> Array.tryFind (fun m -> m.Recommended) |> Option.defaultValue models[0])
                license.Text <- (if String.IsNullOrEmpty f.License then "" else trf "install.license" "Its license: {license}" [ "license", box f.License ])
        let variantChanged () =
            if variantBox.SelectedIndex >= 0 then
                let b = bases[max 0 baseBox.SelectedIndex]
                let v = variants[variantBox.SelectedIndex]
                let fams = pool |> Array.filter (fun f -> baseOf f = b && variantOf f = v)
                quants <- [| for f in fams do for q in f.Models |> Array.map (quantOf f) |> Array.distinct -> q, f |] |> Array.distinctBy fst
                quantField.IsVisible <- quants.Length > 1
                quantBox.ItemsSource <- [ for q, _ in quants -> named d.Quantizers q ]
                let want = if Array.contains recFamily fams then recQuant recFamily else None
                quantBox.SelectedIndex <- max 0 (match want with Some q -> Array.FindIndex(quants, fun (x, _) -> x = q) | None -> 0)
                // a list of the same names keeps its selection without a change (Swift 1.5's versions: GSQ-RCO each)
                familyChanged ()
        let baseChanged () =
            if baseBox.SelectedIndex >= 0 then
                let b = bases[baseBox.SelectedIndex]
                variants <- pool |> Array.filter (fun f -> baseOf f = b) |> Array.map variantOf |> Array.distinct
                variantField.IsVisible <- variants.Length > 1
                variantBox.ItemsSource <- [ for v in variants -> named d.Variants v ]
                variantBox.SelectedIndex <- max 0 (if baseOf recFamily = b then Array.IndexOf(variants, variantOf recFamily) else 0)
                variantChanged ()
        let sizeChanged () =
            match current, selectedModel () with
            | Some f, Some m ->
                contexts.ItemsSource <- [ for c in f.Contexts -> contextLabel c ]
                contexts.SelectedIndex <- max 0 (Array.IndexOf(f.Contexts, m.Context))
                install.IsEnabled <- true
                refresh ()
                refreshImages ()
            | _ -> install.IsEnabled <- false
        baseBox.SelectionChanged.Add(fun _ -> baseChanged ())
        variantBox.SelectionChanged.Add(fun _ -> variantChanged ())
        quantBox.SelectionChanged.Add(fun _ -> familyChanged ())
        sizes.SelectionChanged.Add(fun _ -> sizeChanged ())
        contexts.SelectionChanged.Add(fun _ -> refresh ())
        install.Click.Add(fun _ ->
            match current, selectedModel () with
            | Some f, Some m ->
                refresh ()
                let ctx = selectedContext f
                let chosen =
                    collectArgs [
                        Result.Ok [ "--context"; string ctx ]
                        Fields.args (at "install") id
                        Result.Ok(match imagesValue () with Some v when f.Images -> [ "--images"; v; "--image-auto-largest" ] | _ -> [])
                        Fields.args (at "advanced") id
                        Fields.args (at "images") id
                        Fields.args (at "start") id
                    ]
                match chosen with
                | Error why -> problem.Text <- why
                | Result.Ok more ->
                    // the keys reach setup in its environment, never on its command line (process lists show that)
                    for name, value in Fields.environment all do
                        Environment.SetEnvironmentVariable(name, value)
                    let which =
                        match own with
                        | Some(path, name) -> [ "--gguf"; path ] @ (if name = "" then [] else [ "--name"; name ])
                        | None -> [ "--family"; f.Id; "--model"; m.Id ]
                    installPage (f.Title + " " + m.Id) ([ "--setup"; "--yes"; "--no-start" ] @ which @ more)
            | _ -> ())
        baseBox.SelectedIndex <- max 0 (Array.IndexOf(bases, baseOf recFamily))
        let controls place = [ for x in at place -> x.Control ]
        let advanced =
            Expander(
                Header = (tr "common.advanced" "Advanced"),
                HorizontalAlignment = HorizontalAlignment.Stretch,
                Content = stack 10.0 (controls "advanced" @ controls "images" @ controls "start")
            )
        show (
            stack 14.0 [
                yield heading (tr "install.installAModel" "Install a model") :> Control
                yield field (tr "install.model" "Model") baseBox :> Control
                yield variantField :> Control
                yield quantField :> Control
                yield card familyInfo :> Control
                yield field (tr "install.size" "Size") sizes :> Control
                yield license :> Control
                yield field (tr "install.contextHowMuchText" "Context: how much text the model sees at once (longer takes more VRAM)") contexts :> Control
                yield! controls "install"
                yield imagesField :> Control
                yield advanced :> Control
                yield problem :> Control
                yield actions [ button (tr "common.back" "Back") (fun () -> homePage d); install ] :> Control
            ]
        )
        refresh ()

    and installPage (what: string) (args: string list) =
        runPage (trf "install.installing" "Installing {what}" [ "what", box what ]) args (fun _ -> None) ignore (fun _ -> home ()) |> ignore

    /// An installed model's image generation: its image models chosen again (none: taken out), the files downloaded
    /// and its config rewritten by tools/image/install.py; the server takes it at its next start.
    and imagesPage (d: Describe) (m: Installed) =
        let textRam =
            d.Families |> Array.tryFind (fun f -> f.Id = m.Family)
            |> Option.bind (fun f -> f.Models |> Array.tryFind (fun x -> x.Id = m.Model))
            |> Option.map (fun x -> x.RamGb) |> Option.defaultValue 0.0
        let picker, value, _ = imagePicker d m.Images (fun () -> max 0.0 (d.RamGb - textRam)) ignore
        let holds w = match w with "images" -> true | "sam3" -> not (isNull (box d.Sam3)) && not d.Sam3.Agreed | _ -> false
        let options = fieldsOf d [ "images" ] holds (fun () -> None) |> List.collect snd
        Fields.watch options (fun () -> Fields.refresh options)
        Fields.refresh options
        let problem = coloured failBrush ""
        let apply () =
            let v = value () |> Option.defaultValue "no"
            // tools/image/install.py takes setup's --image-NAME as --NAME
            match (if v = "no" then Result.Ok [] else Fields.args options (fun n -> n.Substring "image-".Length)) with
            | Error why -> problem.Text <- why
            | Result.Ok more ->
                runScriptPage Places.imageInstallPy (trf "images.title" "Image generation for {model}" [ "model", box m.ModelName ])
                    ([ "--config"; m.Config; "--model"; v; "--yes"; "--auto-largest" ] @ more) (fun _ -> None) ignore (fun _ -> home ())
                |> ignore
        show (
            stack 14.0 [
                heading (trf "images.title" "Image generation for {model}" [ "model", box m.ModelName ])
                muted (tr "imgpick.instructions" "Choose image models and add them to the list. An empty list turns image generation off. The model takes the changes at its next start.")
                picker
                Expander(Header = (tr "common.advanced" "Advanced"), Content = stack 10.0 [ for f in options -> f.Control ], HorizontalAlignment = HorizontalAlignment.Stretch)
                problem
                actions [ button (tr "common.back" "Back") (fun () -> homePage d); accentButton (tr "images.apply" "Apply") apply ]
            ]
        )

    /// A start of an installed model with setup's options for a start (the GPUs, the port, who may connect, ...).
    and startOptionsPage (d: Describe) (index: int) (m: Installed) =
        let gpus = d.Gpus |> Array.filter (fun g -> isNull g.Problem)
        let holds w = match w with "multi_gpu" -> gpus.Length > 1 | _ -> false
        let options = fieldsOf d [ "start" ] holds (fun () -> None) |> List.collect snd
        Fields.watch options (fun () -> Fields.refresh options)
        Fields.refresh options
        let browser = CheckBox(IsChecked = Nullable true, Content = text (tr "start.openTheChatIn" "Open the chat in the browser when the model is ready"))
        let problem = coloured failBrush ""
        let start () =
            match Fields.args options id with
            | Error why -> problem.Text <- why
            | Result.Ok more -> startModel index m more (browser.IsChecked = Nullable true)
        show (
            stack 14.0 [
                yield heading (trf "start.title" "Start {model}" [ "model", box m.ModelName ]) :> Control
                yield! [ for f in options -> f.Control ]
                yield browser :> Control
                yield problem :> Control
                yield actions [ button (tr "common.back" "Back") (fun () -> homePage d); accentButton (tr "start.start" "Start") start ] :> Control
            ]
        )

    /// setup --calibrate for an installed model: setup asks whether to measure (it is rarely worth the time).
    and calibrateModel (index: int) (m: Installed) =
        let which = [ "--model"; m.Model; "--family"; m.Family ] |> fun a -> if String.IsNullOrEmpty m.Model then [] else a
        let answer (q: string) = if q.StartsWith "Tune which one" then Some(string (index + 1)) else None
        runPage (trf "tune.title" "Tuning {model}" [ "model", box m.ModelName ]) ([ "--calibrate"; "--no-start" ] @ which) answer ignore (fun _ -> home ()) |> ignore

    /// setup --bench (or upstream's benchmark mode) for an installed model: the record goes to the data folder's
    /// records/.
    and benchPage (d: Describe) (m: Installed) =
        let verbose = CheckBox(Content = text (tr "bench.whereTheTimeGoes" "Where the time goes in more detail (per-stage timings, the kernels, the GPU's load)"))
        let rounds = TextBox(Text = "2", MinWidth = 120.0)
        let temperature = TextBox(Text = "0.7", MinWidth = 120.0)
        let compat = CheckBox(Content = text (tr "bench.measureTheWayUpstream" "Measure the way upstream Strata's benchmark mode does, through the server's API (3-10 minutes)"))
        let long = CheckBox(Content = text (tr "bench.withUpstreamSLong" "With upstream's long prompts (--long)"))
        let clients = TextBox(PlaceholderText = "1", MinWidth = 120.0)
        let own = stack 10.0 [ verbose; field (tr "bench.roundsEachCaseMeasured" "Rounds (each case measured this many times)") rounds; field (tr "bench.temperatureOfTheSampled" "Temperature of the sampled runs (0 to 2, 0 = greedy only)") temperature ]
        let upstream = stack 10.0 [ long; field (tr "bench.clientsAtOnce" "Clients at once") clients ]
        let refresh () =
            own.IsVisible <- compat.IsChecked <> Nullable true
            upstream.IsVisible <- compat.IsChecked = Nullable true
        compat.IsCheckedChanged.Add(fun _ -> refresh ())
        refresh ()
        let problem = coloured failBrush ""
        let which = [ "--model"; m.Model; "--family"; m.Family ] |> fun a -> if String.IsNullOrEmpty m.Model then [] else a
        let start () =
            let args =
                if compat.IsChecked = Nullable true then
                    collectArgs [
                        Result.Ok(which @ [ "--bench-original-compat" ] @ flag "--long" long)
                        optionOf "--clients" clients (isInt 1) (tr "opt.clients" "a number of clients, 1 or more")
                    ]
                else
                    collectArgs [
                        Result.Ok((if verbose.IsChecked = Nullable true then [ "--bench-verbose" ] else [ "--bench" ]) @ which)
                        optionOf "--bench-rounds" rounds (isInt 1) (tr "opt.benchRounds" "a number of rounds, 1 or more")
                        optionOf "--bench-temperature" temperature (fun t -> isNumber 0.0 2.0 t && abs (float t * 10.0 - round (float t * 10.0)) < 1e-6) (tr "opt.benchTemperature" "0 to 2 in steps of 0.1")
                    ]
            match args with
            | Error why -> problem.Text <- why
            | Result.Ok a -> runPage (trf "bench.running" "Measuring {model}" [ "model", box m.ModelName ]) a (fun _ -> None) ignore (fun _ -> home ()) |> ignore
        show (
            stack 14.0 [
                heading (trf "bench.title" "Measure {model}" [ "model", box m.ModelName ])
                muted (tr "bench.measuresThisModelS" "Measures this model's speed on this PC (about 40 minutes; a running XeStrata is stopped first) and writes a record to the data folder's records/. Nothing is sent anywhere.")
                compat
                own
                upstream
                problem
                actions [ button (tr "common.back" "Back") (fun () -> homePage d); accentButton (tr "bench.measure" "Measure") start ]
            ]
        )

    /// setup --inspect: what a GGUF is and whether XeStrata runs it, from its headers only (no download).
    and inspectPage (d: Describe) =
        let source = TextBox(PlaceholderText = (tr "inspect.aFileAFolder" "a file, a folder, a URL, hf:owner/repo or ms:owner/repo"), MinWidth = 420.0)
        let variant = TextBox(PlaceholderText = (tr "inspect.optionalAVariantName" "(optional) a variant name, e.g. IQ3_XXS"), MinWidth = 260.0)
        let output = LogView()
        let go = accentButton (tr "inspect.inspect" "Inspect") ignore
        go.Click.Add(fun _ ->
            match trimmed source with
            | "" -> ()
            | src ->
                go.IsEnabled <- false
                output.Add("> " + src)
                task {
                    let! messages, _ =
                        setup.Collect ([ "--inspect"; src ] @ (match trimmed variant with "" -> [] | v -> [ v ]))
                    ui (fun () ->
                        for msg in messages do
                            match msg with
                            | Text s | Log s | Say s -> output.Add s
                            | Ok s -> output.Add("[ok] " + s)
                            | Warn s -> output.Add("[!] " + s)
                            | Fail(s, h) -> output.Add(s + " " + h)
                            | _ -> ()
                        go.IsEnabled <- true)
                }
                |> ignore)
        show (
            stack 14.0 [
                heading (tr "inspect.inspectAModelFile" "Inspect a model file")
                muted (tr "inspect.whatAGgufIs" "What a GGUF is and whether XeStrata runs it, read from its headers only (nothing is downloaded or installed).")
                field (tr "inspect.theModelFile" "The model file") (row 8.0 [ source; button (tr "common.browse" "Browse ...") (fun () -> pickFolder (tr "inspect.aFolderWithThe" "A folder with the model files") source) ])
                field (tr "inspect.variant" "Variant") variant
                output.Control
                actions [ button (tr "common.back" "Back") (fun () -> homePage d); go ]
            ]
        )

    /// setup --remove-data: the folders listed with their sizes, deleted after a yes pressed at setup's question.
    and removeData () =
        runPage (tr "remove.title" "Remove XeStrata's data") [ "--remove-data" ] (fun _ -> None) ignore (fun _ -> home ()) |> ignore

    and startModel (index: int) (m: Installed) (extra: string list) (openBrowser: bool) =
        let stopButton = button (tr "running.stop" "Stop") ignore
        attached <- None
        let page (url: string) =
            if openBrowser then openUrl (url + "/")
            show (
                stack 14.0 [
                    yield heading m.ModelName :> Control
                    yield card (
                        stack 8.0 [
                            yield coloured okBrush (tr "running.running" "Running") :> Control
                            match attached with
                            | Some(_, true) -> yield text (tr "running.service" "It runs as the systemd service (to stop it: systemctl --user stop xestrata).") :> Control
                            | Some(_, false) -> yield text (tr "running.elsewhere" "It was already running, started elsewhere: stop it where it was started.") :> Control
                            | None -> ()
                            yield text (trf "running.chat" "Chat in the browser: {url}" [ "url", box (url + "/") ]) :> Control
                            yield text (trf "running.api" "API (OpenAI): {url}/v1   (Anthropic: {url}/v1/messages)" [ "url", box url ]) :> Control
                        ]
                    ) :> Control
                    yield
                        actions [
                            // a server this window did not start is not its to stop
                            if attached.IsNone then yield stopButton :> Control
                            else yield button (tr "common.back" "Back") (fun () -> home ()) :> Control
                            yield accentButton (tr "running.openChat" "Open chat") (fun () -> openUrl (url + "/")) :> Control
                        ] :> Control
                ]
            )
        // several installed models: setup asks which one, in describe's order (the newest first)
        let which (q: string) = if q.StartsWith "Which one" then Some(string (index + 1)) else None
        let r =
            runPage (trf "running.starting" "Starting {model}" [ "model", box m.ModelName ]) ([ "--no-browser" ] @ extra) which (fun url -> page url) (fun _ ->
                running <- None
                if attached.IsNone then home ())
        running <- Some r
        stopButton.Click.Add(fun _ -> r.Stop())

    do
        let header =
            let g = Grid(ColumnDefinitions = ColumnDefinitions "*,Auto")
            let title = heading "XeStrata"
            version.VerticalAlignment <- VerticalAlignment.Center
            Grid.SetColumn(version, 1)
            g.Children.Add title
            g.Children.Add version
            g
        let top = stack 10.0 [ header; updateBar ]
        top.Margin <- Thickness(24.0, 16.0, 24.0, 0.0)
        let dock = DockPanel()
        DockPanel.SetDock(top, Dock.Top)
        dock.Children.Add top
        dock.Children.Add host
        this.Content <- dock
        this.Closing.Add(fun _ -> running |> Option.iter (fun r -> r.Stop()))
        home ()
        checkUpdate ()

    new() = new MainWindow(Setup.real)
