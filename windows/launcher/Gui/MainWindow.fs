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

/// A choice among labelled values -> the control and its value.
let private choice (items: (string * string) list) (selected: int) =
    let c = ComboBox(ItemsSource = [ for label, _ in items -> label ], SelectedIndex = selected, MinWidth = 260.0)
    c, (fun () -> snd items[max 0 c.SelectedIndex])

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

/// The options of how a model runs, which setup takes at an install and at a start alike: the GPUs of a layer split,
/// the port, who may connect, the VRAM left free, the draft layer's tokens, an old driver -> the controls and their
/// options (blank: as set up before, or setup's default).
let private runOptions (gpus: Gpu[]) =
    let splitBoxes = [| for g in gpus -> CheckBox(Content = text (sprintf "%s (%s)" g.Name (gb g.VramGb))) |]
    let layerSplit = TextBox(PlaceholderText = (tr "run.autoMeasuredOnThis" "auto (measured on this PC)"), MinWidth = 260.0)
    let port = TextBox(PlaceholderText = (tr "run.asSetUp8095" "as set up (8095 for a new model)"), MinWidth = 260.0)
    let host, hostValue =
        choice [ (tr "run.thisPcOnly127" "This PC only (127.0.0.1)"), ""; (tr "run.everyIpv4Network0" "Every IPv4 network (0.0.0.0)"), "0.0.0.0"; (tr "run.everyIpv6Network" "Every IPv6 network (::)"), "::" ] 0
    let apiKey = TextBox(PlaceholderText = (tr "run.noneAKeyThe" "(none) a key the clients must send; several: key1,key2"), MinWidth = 420.0)
    let reserve = TextBox(PlaceholderText = (tr "run.theEngineSDefault" "the engine's default (700)"), MinWidth = 260.0)
    let vocab, vocabValue =
        choice
            [ (tr "run.withChineseJapaneseAnd" "With Chinese, Japanese and Korean (default)"), ""
              (tr "run.englishAndCodeOnly" "English and code only (~110 MiB less VRAM)"), "en"
              (tr "run.englishCodeAndThe" "English, code and the Cyrillic script"), "cyrillic"
              (tr "run.englishCodeAndFrench" "English, code and French"), "fr" ]
            0
    let oldDriver = CheckBox(Content = text (tr "run.startAlthoughTheIntel" "Start although the Intel GPU's driver is older than the one setup asks for"))
    let control =
        stack 10.0 [
            if gpus.Length > 1 then
                yield field (tr "run.splitTheModelS" "Split the model's layers over these GPUs (two or more; the first runs the early layers)") (stack 4.0 [ for b in splitBoxes -> b :> Control ]) :> Control
                yield field (tr "run.withASplitThe" "With a split: the first layer of each later GPU's share (e.g. 36, or 16,32 for three GPUs)") layerSplit :> Control
            yield field (tr "run.port" "Port") port :> Control
            yield field (tr "run.whoMayConnectWith" "Who may connect (with networks: set an API key too)") host :> Control
            yield field (tr "run.apiKey" "API key") apiKey :> Control
            yield field (tr "run.vramLeftFreeFor" "VRAM left free for other programs, in MiB") reserve :> Control
            yield field (tr "run.theDraftLayerS" "The draft layer's tokens") vocab :> Control
            yield oldDriver :> Control
        ]
    let args () =
        let split = [ for i, b in Array.indexed splitBoxes do if b.IsChecked = Nullable true then string gpus[i].Index ]
        collectArgs [
            Result.Ok(if split.Length >= 2 then [ "--gpus"; String.Join(",", split) ] else [])
            (if split.Length >= 2 then
                 optionOf "--layer-split" layerSplit (fun t -> t = "auto" || t.Split(',') |> Array.forall (isInt 1)) (tr "opt.layerSplit" "auto or rising layer numbers, e.g. 36 or 16,32")
             else Result.Ok [])
            optionOf "--port" port (fun t -> isInt 1 t && int t <= 65535) (tr "opt.port" "a port number, 1 to 65535")
            Result.Ok(match hostValue () with "" -> [] | h -> [ "--host"; h ])
            Result.Ok(match trimmed apiKey with "" -> [] | k -> [ "--api-key"; k ])
            optionOf "--vram-reserve-mib" reserve (isInt 0) (tr "opt.vramReserveMib" "a number of MiB, 0 or more")
            Result.Ok(match vocabValue () with "" -> [] | v -> [ "--draft-vocab"; v ])
            Result.Ok(flag "--old-driver" oldDriver)
        ]
    control, args

/// How the image models run, which setup (--image-<name>) and tools/image/install.py (--<name>) take alike -> the
/// controls and the (name, value) pairs.
let private imageRunOptions () =
    let ditQ8, ditQ8Value =
        choice [ (tr "img.auto16BitWhen" "Auto: 16-bit when it fits the VRAM with its text encoder, else 8-bit"), "auto"; (tr "img.always8Bit" "Always 8-bit"), "on"; (tr "img.always16Bit" "Always 16-bit"), "off" ] 0
    let ditKeep, ditKeepValue =
        choice [ (tr "img.keepTheOneNot" "Keep the one not in use in RAM (quick switch)"), "on"; (tr "img.letItGoAbout" "Let it go (about 4.4 GB less RAM)"), "off" ] 0
    let ssdOffload = CheckBox(Content = text (tr "img.keepTheImageModels" "Keep the image models' sizes when the RAM is short, reading what does not fit from the SSD (slower)"))
    let policy, policyValue =
        choice
            [ (tr "img.autoStaysWhenBoth" "Auto: stays when both fit, else gives VRAM back or is unloaded"), "auto"
              (tr "img.keepItLoaded" "Keep it loaded"), "keep"
              (tr "img.giveVramBackFor" "Give VRAM back for the image"), "shrink"
              (tr "img.unloadItItsConversation" "Unload it, its conversation restored afterwards"), "unload" ]
            0
    let control =
        stack 10.0 [
            field (tr "img.mageFlowMossImage" "Mage-Flow, MOSS-Image and Anima: the image model on the GPU") ditQ8
            field (tr "img.mageFlowTheImage" "Mage-Flow: the image model not in use (the generation's or the edit's)") ditKeep
            ssdOffload
            field (tr "img.theTextModelWhile" "The text model while an image is made") policy
        ]
    let values () =
        [ "dit-q8", ditQ8Value ()
          "dit-keep", ditKeepValue ()
          "ssd-offload", (if ssdOffload.IsChecked = Nullable true then "on" else "off")
          "gpu-policy", policyValue () ]
    control, values

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
        let r = setup.Start script args onMessage (fun code -> ui (fun () -> if not failed then onDone code))
        run <- Some r
        r

    let runPage = runScriptPage Places.setupPy

    /// The image models to choose from (describe's), `current` checked (--images names: the default first) -> the
    /// control and its --images value (None: none checked).  A choice of the default where several are, of the size
    /// where a model has several (Auto: by the RAM), and of the text encoder with the refusal direction removed
    /// (-uncensored) where the models have one.
    let imagePicker (models: ImageModel[]) (current: string[]) =
        let plain (id: string) =
            let id = id.Split(':')[0]
            if id.EndsWith "-uncensored" then id.Substring(0, id.Length - 11) else id
        let had = current |> Array.map plain
        let boxes =
            [| for m in models ->
                   CheckBox(
                       IsChecked = Nullable(Array.contains m.Id had),
                       Content =
                           stack 2.0 [
                               row 6.0 [
                                   yield subheading m.Title
                                   if not m.Edits then yield badge mutedBrush (tr "img.noEdits" "no edits")
                               ]
                               muted (trf "img.size" "{about}; download {download}, needs ~{ram} of RAM besides the text model" [ "about", box m.About; "download", box (gb m.DownloadGb); "ram", box (gb m.RamGb) ])
                           ]
                   ) |]
        // each model's size where it has several: Auto (the largest that fits the RAM) or one of them
        let sizes =
            [| for m in models ->
                   let c = ComboBox(ItemsSource = Array.append [| tr "img.auto" "Auto" |] m.Quants, SelectedIndex = 0, MinWidth = 110.0)
                   let was = current |> Array.tryFind (fun id -> plain id = m.Id && id.Contains ":")
                   was |> Option.iter (fun id -> c.SelectedIndex <- 1 + max -1 (Array.IndexOf(m.Quants, id.Split(':')[1])))
                   c.IsVisible <- m.Quants.Length > 0
                   c |]
        let uncensored =
            CheckBox(
                Content = text (tr "img.textEncoderWithIts" "Text encoder with its refusal direction removed (-uncensored)"),
                IsChecked = Nullable(current |> Array.exists (fun c -> c.EndsWith "-uncensored"))
            )
        let defaultBox = ComboBox(MinWidth = 260.0)
        let defaultField = field (tr "img.defaultImageModelThe" "Default image model (the others: chosen on the web page or by the API's \"model\")") defaultBox
        let checkedModels () = [ for i, b in Array.indexed boxes do if b.IsChecked = Nullable true then models[i] ]
        let refresh () =
            let ms = checkedModels ()
            let was = match defaultBox.SelectedItem with :? string as t -> t | _ -> ""
            let first = if had.Length > 0 then (models |> Array.tryFind (fun m -> m.Id = had[0])) else None
            defaultBox.ItemsSource <- [ for m in ms -> m.Title ]
            defaultBox.SelectedItem <-
                (match ms |> List.tryFind (fun m -> m.Title = was), first with
                 | Some m, _ -> m.Title
                 | None, Some f when List.contains f ms -> f.Title
                 | _ -> (match ms with m :: _ -> m.Title | [] -> null))
            defaultField.IsVisible <- ms.Length > 1
            uncensored.IsVisible <- ms |> List.exists (fun m -> m.Uncensored)
        for b in boxes do
            b.IsCheckedChanged.Add(fun _ -> refresh ())
        refresh ()
        let value () =
            match checkedModels () with
            | [] -> None
            | ms ->
                let def = match defaultBox.SelectedItem with :? string as t -> t | _ -> ms.Head.Title
                let ordered = (ms |> List.filter (fun m -> m.Title = def)) @ (ms |> List.filter (fun m -> m.Title <> def))
                let name (m: ImageModel) =
                    let id = if uncensored.IsChecked = Nullable true && m.Uncensored then m.Id + "-uncensored" else m.Id
                    let size = sizes[Array.findIndex (fun (x: ImageModel) -> x.Id = m.Id) models]
                    match size.SelectedIndex with
                    | i when i > 0 -> id + ":" + m.Quants[i - 1]
                    | _ -> id
                Some(ordered |> List.map name |> String.concat ",")
        let control =
            stack 8.0 [
                for i, b in Array.indexed boxes do
                    yield row 12.0 [ b :> Control; sizes[i] :> Control ] :> Control
                yield defaultField :> Control
                yield uncensored :> Control
            ]
        control, value

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

    /// The install page; `own`: the model in the user's own files (--gguf PATH, --name NAME) instead of a family's.
    and choosePage (d: Describe) (own: (string * string) option) =
        let families =
            ListBox(
                ItemsSource = (if own.IsSome then d.Families |> Array.filter (fun f -> f.Id = "custom") else d.Families |> Array.filter (fun f -> f.Id <> "custom")),
                MaxHeight = 260.0
            )
        families.ItemTemplate <-
            itemTemplate (fun (f: Family) ->
                stack 2.0 [
                    row 6.0 [
                        subheading f.Title
                        if f.Experimental then badge warnBrush (tr "common.experimental" "experimental")
                    ]
                    muted (f.By + " - " + f.About)
                ])
        let sizes = ListBox(MaxHeight = 320.0)
        sizes.ItemTemplate <-
            itemTemplate (fun (m: Model) ->
                stack 2.0 [
                    row 6.0 [
                        yield subheading m.Id
                        if m.Recommended then yield badge okBrush (tr "install.recommended" "recommended")
                        if not m.Tested then yield badge mutedBrush (tr "install.untested" "untested")
                        if m.Experimental then yield badge warnBrush (tr "common.experimental" "experimental")
                        match m.Fits with
                        | "low-ram" -> yield badge warnBrush (tr "install.lowRamMode" "low-RAM mode")
                        | "budget" -> yield badge warnBrush (tr "install.partFromTheSsd" "part from the SSD")
                        | "slow" | "no" -> yield badge failBrush (tr "install.doesNotFitThe" "does not fit the RAM")
                        | _ -> ()
                    ]
                    muted (trf "install.sizeAbout" "{about}; download {download}, needs ~{ram} of RAM" [ "about", box m.About; "download", box (gb m.DownloadGb); "ram", box (gb m.RamGb) ])
                ])
        let contexts = ComboBox(MinWidth = 160.0)
        let gpus = d.Gpus |> Array.filter (fun g -> isNull g.Problem)
        let gpuBox = ComboBox(ItemsSource = [ for g in gpus -> sprintf "%s (%s)" g.Name (gb g.VramGb) ], MinWidth = 260.0)
        gpuBox.SelectedIndex <- (if d.Gpu.HasValue then defaultArg (Array.tryFindIndex (fun g -> g.Index = d.Gpu.Value) gpus) 0 else 0)
        let dataDir = TextBox(Text = d.DataDir, MinWidth = 420.0)
        let ggufDir = TextBox(PlaceholderText = (tr "install.optionalAFolderWith" "(optional) a folder with the model files already downloaded"), MinWidth = 420.0)
        let license = muted ""
        let install = accentButton (tr "install.install" "Install") ignore
        let problem = coloured failBrush ""
        // setup's other questions (step 2), each shown where setup would ask it
        let pictures = CheckBox(IsChecked = Nullable true, Content = text (tr "install.readPicturesTooScreenshots" "Read pictures too (screenshots, photos, scanned pages; a 0.9 GB download)"))
        let kv, kvValue =
            choice
                [ (tr "install.8BitRecommended" "8-bit (recommended)"), "int8"
                  (tr "install.4BitHalfThe" "4-bit: half the memory, a little less precise on long documents"), "q4_0"
                  (tr "install.8BitKeysAnd" "8-bit keys and 4-bit values: about three quarters of the memory"), "k8v4"
                  (tr "install.16BitTwiceThe" "16-bit: twice the memory of 8-bit"), "fp16" ]
                0
        let kvField = field (tr "install.kvCachePrecisionThe" "KV cache precision (the model's memory of the conversation)") kv
        // a family whose KV cache setup does not ask: 16-bit unless one of the precisions it takes is chosen
        let kvLabels =
            dict [ "fp16", tr "install.kvFp16" "16-bit (default)"; "int8", tr "install.kvInt8" "8-bit"
                   "q4_0", tr "install.kvQ4" "4-bit (Hadamard-rotated)"; "k8v4", tr "install.kvK8v4" "8-bit keys and 4-bit values" ]
        let plainKv = ComboBox(MinWidth = 260.0)
        let plainKvField = field (tr "install.kvCachePrecisionThe" "KV cache precision (the model's memory of the conversation)") plainKv
        let plainKvValue (f: Family) = if plainKv.SelectedIndex >= 0 then f.KvTakes[plainKv.SelectedIndex] else "fp16"
        let rope, ropeValue =
            choice [ (tr "install.yarnKeepsTheTrained" "yarn (keeps the trained angles on the high-frequency pairs)"), "yarn"; (tr "install.linearShrinksEveryAngle" "linear (shrinks every angle)"), "linear" ] 0
        let ropeField = field (tr "install.ropeExtensionAContext" "RoPE extension: a context past the model's trained 262,144 positions rescales its rotary angles") rope
        let projection =
            CheckBox(
                Content =
                    stack 2.0 [
                        text (tr "install.experimentalSpeedProjectionOff" "EXPERIMENTAL - speed projection (off unless you choose it)")
                        muted (tr "install.aSmallControlVector" "A small control vector applied while the model runs (layers 4-44). It changes how the model answers: its package describes it as a refusal-direction projection (the model declines far fewer requests). When on, the web app can switch it off per chat. For fewer refusals, choose SC117's or huihui-ai's abliterated version as the model instead (the refusals removed in its weights).")
                    ]
            )
        let vector = TextBox(PlaceholderText = (tr "install.optionalAnotherControlVector" "(optional) another control vector, a GGUF file"), MinWidth = 420.0)
        let vectorField = field (tr "install.theSpeedProjectionS" "The speed projection's vector") vector
        let images, imagesValue = imagePicker d.Images d.ImageDefault
        let webSearch, webSearchValue =
            choice
                [ (tr "install.off" "Off"), "no"
                  (tr "install.deepseekSSearchApi" "DeepSeek's search API: your API key, billed by DeepSeek per search (USD 0.75 per 1,000); the search words go to DeepSeek"), "deepseek"
                  (tr "install.tinyfishSSearchApi" "TinyFish's search API: your API key, free up to a daily allowance (12,000 searches, 1,000 pages read); the search words and the pages found go to TinyFish"), "tinyfish" ]
                (match d.WebSearch with "deepseek" -> 1 | "tinyfish" -> 2 | _ -> 0)
        // the search APIs' keys: (setup's --web-search value, its name, the box)
        let apiKeys =
            [ for id, who in [ "deepseek", "DeepSeek"; "tinyfish", "TinyFish" ] ->
                  id, who,
                  TextBox(PasswordChar = '*', MinWidth = 420.0,
                          PlaceholderText = (if d.WebSearch = id then tr "install.keySet" "set (type to replace)" else trf "install.keyYour" "your {who} API key" [ "who", box who ])) ]
        let apiKeyFields =
            [ for id, who, box in apiKeys -> id, field (trf "install.keyField" "{who} API key (kept in your settings, never shown)" [ "who", who :> obj ]) box ]
        let showApiKey () =
            for id, f in apiKeyFields do f.IsVisible <- webSearchValue () = id
        showApiKey ()
        webSearch.SelectionChanged.Add(fun _ -> showApiKey ())
        let webSearchField = field (tr "install.webSearchForEvery" "Web search, for every model: the model searches when a request asks for it (the web page's Search switch, the API's web search tools)") webSearch
        let imagesField = field (tr "install.imageGenerationModelsTo" "Image generation: models to make images with, besides the text model") images
        // the options setup takes without asking, under Advanced
        let reader, readerValue =
            choice [ (tr "install.forThisGpuOn" "For this GPU (on the GPU from 24 GB of VRAM, else on the CPU)"), "yes"; (tr "install.onTheGpuThe" "On the GPU (the CPU when it cannot start there)"), "gpu"; (tr "install.onTheCpu" "On the CPU"), "cpu" ] 0
        let readerField = field (tr "install.thePictureReader" "The picture reader") reader
        let visionTokens = TextBox(PlaceholderText = (tr "install.1024OnTheGpu" "1024 on the GPU, 300 on the CPU"), MinWidth = 260.0)
        let visionTokensField = field (tr "install.theMostImageTokens" "The most image tokens a picture becomes (more reads small text better, and takes longer)") visionTokens
        let ropeScale = TextBox(PlaceholderText = (tr "install.theContextOver262" "the context over 262,144"), MinWidth = 260.0)
        let ropeScaleField = field (tr "install.ropeExtensionFactor1" "RoPE extension factor (1 or more)") ropeScale
        let lowRam, lowRamValue =
            choice
                [ (tr "install.autoWhenTheExperts" "Auto: when the experts do not fit the RAM"), "auto"
                  (tr "install.on" "On"), "on"
                  (tr "install.off" "Off"), "off"
                  (tr "install.onTheExpertsThe" "On, the experts the GPU does not hold copied into RAM"), "resident"
                  (tr "install.onTheExpertsThe2" "On, the experts the GPU does not hold read through the OS file cache"), "mmap" ]
                0
        let budget = TextBox(PlaceholderText = (tr "install.setupSChoiceFor" "setup's choice for this RAM"), MinWidth = 260.0)
        let budgetField = field (tr "install.gibOfTheExperts" "GiB of the experts kept in RAM (the rest read from the SSD)") budget
        let ramReserve = TextBox(PlaceholderText = (tr "install.noneGibOfRam" "(none) GiB of RAM to leave free for other programs"), MinWidth = 260.0)
        let parallelBox = TextBox(PlaceholderText = (tr "install.1OneRequestAt" "1: one request at a time"), MinWidth = 260.0)
        let replicas = TextBox(PlaceholderText = "1", MinWidth = 260.0)
        let source, sourceValue =
            choice [ (tr "install.huggingFaceDefault" "Hugging Face (default)"), ""; (tr "install.modelscopeMainlandChinaThe" "ModelScope (mainland China: the same files)"), "modelscope" ] 0
        let modelsDir = TextBox(PlaceholderText = (tr "install.dataFolderModels" "<data folder>\\models"), MinWidth = 420.0)
        let moveData = CheckBox(Content = text (tr "install.moveTheModelFiles" "Move the model files already on this PC into the data folder"))
        let rememberData = CheckBox(Content = text (tr "install.rememberTheDataFolder" "Remember the data folder for later downloads without moving anything"))
        let noWarmup = CheckBox(Content = text (tr "install.skipTheFirstStart" "Skip the first start that compiles the GPU code (the first request then takes ~30 s more)"))
        let imageOptions, imageValues = imageRunOptions ()
        let serverControl, serverArgs = runOptions gpus
        let selectedModel () =
            match sizes.SelectedItem with
            | :? Model as m -> Some m
            | _ -> None
        let selectedContext (f: Family) = f.Contexts[max 0 contexts.SelectedIndex]
        let refresh () =
            match families.SelectedItem, selectedModel () with
            | (:? Family as f), Some m ->
                let ctx = selectedContext f
                pictures.IsVisible <- m.Vision
                readerField.IsVisible <- m.Vision
                visionTokensField.IsVisible <- m.Vision
                kvField.IsVisible <- f.Kv && ctx > 8192
                plainKvField.IsVisible <- not f.Kv && f.KvTakes.Length > 1
                ropeField.IsVisible <- ctx > 262144
                ropeScaleField.IsVisible <- ctx > 262144
                projection.IsVisible <- f.SpeedProjection
                vectorField.IsVisible <- f.SpeedProjection && projection.IsChecked = Nullable true
                budgetField.IsVisible <- m.Budget
                imagesField.IsVisible <- f.Images && d.Images.Length > 0
                imageOptions.IsVisible <- imagesField.IsVisible
            | _ -> ()
        let familyChanged () =
            match families.SelectedItem with
            | :? Family as f ->
                sizes.ItemsSource <- f.Models
                plainKv.ItemsSource <- [ for k in f.KvTakes -> (match kvLabels.TryGetValue k with | true, l -> l | _ -> k) ]
                plainKv.SelectedIndex <- max 0 (Array.IndexOf(f.KvTakes, "fp16"))
                sizes.SelectedItem <- (f.Models |> Array.tryFind (fun m -> m.Recommended) |> Option.defaultValue f.Models[0])
                license.Text <- (if String.IsNullOrEmpty f.License then "" else trf "install.license" "Its license: {license}" [ "license", box f.License ])
            | _ -> ()
        let sizeChanged () =
            match families.SelectedItem, selectedModel () with
            | (:? Family as f), Some m ->
                contexts.ItemsSource <- [ for c in f.Contexts -> contextLabel c ]
                contexts.SelectedIndex <- max 0 (Array.IndexOf(f.Contexts, m.Context))
                install.IsEnabled <- true
                refresh ()
            | _ -> install.IsEnabled <- false
        families.SelectionChanged.Add(fun _ -> familyChanged ())
        sizes.SelectionChanged.Add(fun _ -> sizeChanged ())
        contexts.SelectionChanged.Add(fun _ -> refresh ())
        projection.IsCheckedChanged.Add(fun _ -> refresh ())
        install.Click.Add(fun _ ->
            match families.SelectedItem, selectedModel () with
            | (:? Family as f), Some m ->
                let ctx = selectedContext f
                let questions =
                    [ yield "--context"; yield string ctx
                      if m.Vision then
                          yield "--vision"
                          yield (if pictures.IsChecked = Nullable true then readerValue () else "no")
                      if f.Kv && ctx > 8192 then yield! [ "--kv"; kvValue () ]
                      if ctx > 262144 then yield! [ "--rope-scaling"; ropeValue () ]
                      if f.SpeedProjection then
                          yield "--experimental-speed-projection"
                          yield (match projection.IsChecked = Nullable true, trimmed vector with
                                 | false, _ -> "off"
                                 | true, "" -> "on"
                                 | true, file -> file)
                      match imagesValue () with
                      | Some v when f.Images -> yield! [ "--images"; v ]
                      | _ -> ()
                      yield! [ "--web-search"; webSearchValue () ] ]
                let withImages = f.Images && (imagesValue ()).IsSome
                let advanced =
                    collectArgs [
                        (if m.Vision && pictures.IsChecked = Nullable true then
                             optionOf "--vision-tokens" visionTokens (isInt 1) (tr "opt.visionTokens" "a number of image tokens, 1 or more")
                         else Result.Ok [])
                        (if ctx > 262144 then optionOf "--rope-scale" ropeScale (isNumber 1.0 1e6) (tr "opt.ropeScale" "a number, 1 or more") else Result.Ok [])
                        Result.Ok(match lowRamValue () with "auto" -> [] | v -> [ "--low-ram"; v ])
                        Result.Ok(if f.Kv then [] else match plainKvValue f with "fp16" -> [] | k -> [ "--kv"; k ])
                        (if m.Budget then optionOf "--resident-budget-gib" budget (isNumber 1.0 1e6) (tr "opt.residentBudgetGib" "a number of GiB") else Result.Ok [])
                        optionOf "--ram-reserve-gib" ramReserve (isNumber 0.0 1e6) (tr "opt.ramReserveGib" "a number of GiB, 0 or more")
                        optionOf "--parallel" parallelBox (isInt 1) (tr "opt.parallel" "a number of requests, 1 or more")
                        optionOf "--replicas" replicas (isInt 1) (tr "opt.replicas" "a number of engines, 1 or more")
                        Result.Ok(match sourceValue () with "" -> [] | v -> [ "--source"; v ])
                        Result.Ok(match trimmed modelsDir with "" -> [] | p -> [ "--models-dir"; p ])
                        Result.Ok(flag "--move-data" moveData @ flag "--set-default-data-dir" rememberData @ flag "--no-warmup" noWarmup)
                        Result.Ok(if withImages then [ for name, v in imageValues () do yield! [ "--image-" + name; v ] ] else [])
                        serverArgs ()
                    ]
                let api = apiKeys |> List.tryFind (fun (id, _, _) -> id = webSearchValue ())
                let key = match api with Some (_, _, box) -> trimmed box | None -> ""
                match advanced with
                | Error why -> problem.Text <- why
                | Result.Ok _ when api.IsSome && key = "" && d.WebSearch <> webSearchValue () ->
                    let _, who, _ = api.Value
                    problem.Text <- trf "install.keyNeeded" "Web search with {who} needs your {who} API key." [ "who", box who ]
                | Result.Ok more ->
                    // the key reaches setup in its environment, never on its command line (process lists show that)
                    for id, _, box in apiKeys do
                        let k = if webSearchValue () = id then trimmed box else ""
                        Environment.SetEnvironmentVariable($"XESTRATA_{id.ToUpperInvariant()}_API_KEY", (if k = "" then null else k))
                    let which =
                        match own with
                        | Some(path, name) -> [ "--gguf"; path ] @ (if name = "" then [] else [ "--name"; name ])
                        | None -> [ "--family"; f.Id; "--model"; m.Id ]
                    let args =
                        [ "--setup"; "--yes"; "--no-start" ] @ which @ [ "--data-dir"; dataDir.Text ]
                        @ questions
                        @ (if gpus.Length > 0 then [ "--gpu"; string gpus[max 0 gpuBox.SelectedIndex].Index ] else [])
                        @ (if own.IsSome || String.IsNullOrWhiteSpace ggufDir.Text then [] else [ "--gguf-dir"; ggufDir.Text ])
                        @ more
                    installPage (f.Title + " " + m.Id) args
            | _ -> ())
        families.SelectedIndex <- 0
        let advanced =
            Expander(
                Header = (tr "common.advanced" "Advanced"),
                HorizontalAlignment = HorizontalAlignment.Stretch,
                Content =
                    stack 10.0 [
                        readerField
                        visionTokensField
                        ropeScaleField
                        vectorField
                        plainKvField
                        field (tr "install.lowRamModeThe" "Low-RAM mode: the experts read from the model files instead of all copied into RAM") lowRam
                        budgetField
                        field (tr "install.ramLeftFreeFor" "RAM left free for other programs, in GiB (the model's mode, its experts in RAM and the image models' sizes are chosen with that much less; slower when the experts are then read from the SSD)") ramReserve
                        field (tr "install.requestsDecodedTogetherEach" "Requests decoded together (each takes VRAM from the expert cache)") parallelBox
                        field (tr "install.withALayerSplit" "With a layer split: independent engines, each on an equal group of the GPUs") replicas
                        if own.IsNone then
                            field (tr "install.whereTheModelFiles" "Where the model files come from") source
                            field (tr "install.whereTheModelFiles2" "Where the model files go") (row 8.0 [ modelsDir; button (tr "common.browse" "Browse ...") (fun () -> pickFolder (tr "install.modelFilesFolder" "Model files folder") modelsDir) ])
                        moveData
                        rememberData
                        noWarmup
                        imageOptions
                        serverControl
                    ]
            )
        show (
            stack 14.0 [
                heading (tr "install.installAModel" "Install a model")
                field (tr "install.model" "Model") families
                field (tr "install.size" "Size") sizes
                license
                field (tr "install.contextHowMuchText" "Context: how much text the model sees at once (longer takes more VRAM)") contexts
                if gpus.Length > 1 then field (tr "install.gpu" "GPU") gpuBox
                field (tr "install.dataFolderWhereThe" "Data folder: where the model files go (tens of GB)") (row 8.0 [ dataDir; button (tr "common.browse" "Browse ...") (fun () -> pickFolder (tr "install.dataFolder" "Data folder") dataDir) ])
                if own.IsNone then
                    field (tr "install.modelFilesYouAlready" "Model files you already have") (row 8.0 [ ggufDir; button (tr "common.browse" "Browse ...") (fun () -> pickFolder (tr "install.modelFiles" "Model files") ggufDir) ])
                pictures
                kvField
                ropeField
                projection
                imagesField
                webSearchField
                yield! [ for _, f in apiKeyFields -> f :> Control ]
                advanced
                problem
                actions [ button (tr "common.back" "Back") (fun () -> homePage d); install ]
            ]
        )

    and installPage (what: string) (args: string list) =
        runPage (trf "install.installing" "Installing {what}" [ "what", box what ]) args (fun _ -> None) ignore (fun _ -> home ()) |> ignore

    /// An installed model's image generation: its image models chosen again (none: taken out), the files downloaded
    /// and its config rewritten by tools/image/install.py; the server takes it at its next start.
    and imagesPage (d: Describe) (m: Installed) =
        let picker, value = imagePicker d.Images m.Images
        let options, optionValues = imageRunOptions ()
        let apply () =
            let v = value () |> Option.defaultValue "no"
            let more = if v = "no" then [] else [ for name, x in optionValues () do yield! [ "--" + name; x ] ]
            runScriptPage Places.imageInstallPy (trf "images.title" "Image generation for {model}" [ "model", box m.ModelName ])
                ([ "--config"; m.Config; "--model"; v; "--yes" ] @ more) (fun _ -> None) ignore (fun _ -> home ())
            |> ignore
        show (
            stack 14.0 [
                heading (trf "images.title" "Image generation for {model}" [ "model", box m.ModelName ])
                muted (tr "images.checkTheImageModels" "Check the image models to make images with (none: image generation off). Their files are downloaded, and the model takes them at its next start.")
                picker
                Expander(Header = (tr "common.advanced" "Advanced"), Content = options, HorizontalAlignment = HorizontalAlignment.Stretch)
                actions [ button (tr "common.back" "Back") (fun () -> homePage d); accentButton (tr "images.apply" "Apply") apply ]
            ]
        )

    /// A start of an installed model with setup's options for a start (the GPUs, the port, who may connect, ...).
    and startOptionsPage (d: Describe) (index: int) (m: Installed) =
        let gpus = d.Gpus |> Array.filter (fun g -> isNull g.Problem)
        let control, args = runOptions gpus
        let one, oneValue =
            choice ((tr "start.asSetUp" "As set up", "") :: [ for g in gpus -> trf "start.only" "Only {gpu}, this time" [ "gpu", box g.Name ], string g.Index ]) 0
        let browser = CheckBox(IsChecked = Nullable true, Content = text (tr "start.openTheChatIn" "Open the chat in the browser when the model is ready"))
        let problem = coloured failBrush ""
        let start () =
            match args () with
            | Error why -> problem.Text <- why
            | Result.Ok more ->
                let gpu = match oneValue () with "" -> [] | i -> [ "--gpu"; i ]
                startModel index m (gpu @ more) (browser.IsChecked = Nullable true)
        show (
            stack 14.0 [
                yield heading (trf "start.title" "Start {model}" [ "model", box m.ModelName ]) :> Control
                if gpus.Length > 1 then yield field (tr "start.gpu" "GPU") one :> Control
                yield control :> Control
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
        let which = [ "--model"; m.Model; "--family"; m.Family ]
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
        let page (url: string) =
            if openBrowser then openUrl (url + "/")
            show (
                stack 14.0 [
                    heading m.ModelName
                    card (
                        stack 8.0 [
                            coloured okBrush (tr "running.running" "Running")
                            text (trf "running.chat" "Chat in the browser: {url}" [ "url", box (url + "/") ])
                            text (trf "running.api" "API (OpenAI): {url}/v1   (Anthropic: {url}/v1/messages)" [ "url", box url ])
                        ]
                    )
                    actions [ stopButton; accentButton (tr "running.openChat" "Open chat") (fun () -> openUrl (url + "/")) ]
                ]
            )
        // several installed models: setup asks which one, in describe's order (the newest first)
        let which (q: string) = if q.StartsWith "Which one" then Some(string (index + 1)) else None
        let r =
            runPage (trf "running.starting" "Starting {model}" [ "model", box m.ModelName ]) ([ "--no-browser" ] @ extra) which (fun url -> page url) (fun _ ->
                running <- None
                home ())
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
