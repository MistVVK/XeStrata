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
open Avalonia.Controls.Templates
open Avalonia.Layout
open Avalonia.Platform.Storage
open Avalonia.Threading
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
    | _ -> Error(sprintf "%s takes %s." name takes)

/// The options' words in order, or the first reason one is refused.
let private collectArgs (parts: Result<string list, string> list) =
    parts |> List.fold (fun acc p -> match acc, p with Result.Ok a, Result.Ok b -> Result.Ok(a @ b) | Error e, _ | _, Error e -> Error e) (Result.Ok [])

let private flag (name: string) (box: CheckBox) = if box.IsChecked = Nullable true then [ name ] else []

/// The options of how a model runs, which setup takes at an install and at a start alike: the GPUs of a layer split,
/// the port, who may connect, the VRAM left free, the draft layer's tokens, an old driver -> the controls and their
/// options (blank: as set up before, or setup's default).
let private runOptions (gpus: Gpu[]) =
    let splitBoxes = [| for g in gpus -> CheckBox(Content = text (sprintf "%s (%s)" g.Name (gb g.VramGb))) |]
    let layerSplit = TextBox(PlaceholderText = "auto (measured on this PC)", MinWidth = 260.0)
    let port = TextBox(PlaceholderText = "as set up (8095 for a new model)", MinWidth = 260.0)
    let host, hostValue =
        choice [ "This PC only (127.0.0.1)", ""; "Every IPv4 network (0.0.0.0)", "0.0.0.0"; "Every IPv6 network (::)", "::" ] 0
    let apiKey = TextBox(PlaceholderText = "(none) a key the clients must send; several: key1,key2", MinWidth = 420.0)
    let reserve = TextBox(PlaceholderText = "the engine's default (700)", MinWidth = 260.0)
    let vocab, vocabValue =
        choice
            [ "With Chinese, Japanese and Korean (default)", ""
              "English and code only (~110 MiB less VRAM)", "en"
              "English, code and the Cyrillic script", "cyrillic"
              "English, code and French", "fr" ]
            0
    let oldDriver = CheckBox(Content = text "Start although the Intel GPU's driver is older than the one setup asks for")
    let control =
        stack 10.0 [
            if gpus.Length > 1 then
                yield field "Split the model's layers over these GPUs (two or more; the first runs the early layers)" (stack 4.0 [ for b in splitBoxes -> b :> Control ]) :> Control
                yield field "With a split: the first layer of each later GPU's share (e.g. 36, or 16,32 for three GPUs)" layerSplit :> Control
            yield field "Port" port :> Control
            yield field "Who may connect (with networks: set an API key too)" host :> Control
            yield field "API key" apiKey :> Control
            yield field "VRAM left free for other programs, in MiB" reserve :> Control
            yield field "The draft layer's tokens" vocab :> Control
            yield oldDriver :> Control
        ]
    let args () =
        let split = [ for i, b in Array.indexed splitBoxes do if b.IsChecked = Nullable true then string gpus[i].Index ]
        collectArgs [
            Result.Ok(if split.Length >= 2 then [ "--gpus"; String.Join(",", split) ] else [])
            (if split.Length >= 2 then
                 optionOf "--layer-split" layerSplit (fun t -> t = "auto" || t.Split(',') |> Array.forall (isInt 1)) "auto or rising layer numbers, e.g. 36 or 16,32"
             else Result.Ok [])
            optionOf "--port" port (fun t -> isInt 1 t && int t <= 65535) "a port number, 1 to 65535"
            Result.Ok(match hostValue () with "" -> [] | h -> [ "--host"; h ])
            Result.Ok(match trimmed apiKey with "" -> [] | k -> [ "--api-key"; k ])
            optionOf "--vram-reserve-mib" reserve (isInt 0) "a number of MiB, 0 or more"
            Result.Ok(match vocabValue () with "" -> [] | v -> [ "--draft-vocab"; v ])
            Result.Ok(flag "--old-driver" oldDriver)
        ]
    control, args

/// How the image models run, which setup (--image-<name>) and tools/image/install.py (--<name>) take alike -> the
/// controls and the (name, value) pairs.
let private imageRunOptions () =
    let ditQ8, ditQ8Value =
        choice [ "Auto: 16-bit when it fits the VRAM with its text encoder, else 8-bit", "auto"; "Always 8-bit", "on"; "Always 16-bit", "off" ] 0
    let ditKeep, ditKeepValue =
        choice [ "Keep the one not in use in RAM (quick switch)", "on"; "Let it go (about 4.4 GB less RAM)", "off" ] 0
    let ssdOffload = CheckBox(Content = text "Keep the image models' sizes when the RAM is short, reading what does not fit from the SSD (slower)")
    let policy, policyValue =
        choice
            [ "Auto: stays when both fit, else gives VRAM back or is unloaded", "auto"
              "Keep it loaded", "keep"
              "Give VRAM back for the image", "shrink"
              "Unload it, its conversation restored afterwards", "unload" ]
            0
    let control =
        stack 10.0 [
            field "Mage-Flow, MOSS-Image and Anima: the image model on the GPU" ditQ8
            field "Mage-Flow: the image model not in use (the generation's or the edit's)" ditKeep
            ssdOffload
            field "The text model while an image is made" policy
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

type MainWindow() as this =
    inherit Window(Title = "XeStrata", Width = 980.0, Height = 720.0, MinWidth = 720.0, MinHeight = 520.0)

    let host = ContentControl(Margin = Thickness(24.0, 16.0))
    let version = muted ""
    let updateBar = Border(IsVisible = false, Background = accentTint, CornerRadius = CornerRadius 6.0, Padding = Thickness 10.0)
    let mutable running: Setup.Run option = None
    // setup's warnings while it checked this PC: why a GPU is missing ("the engine could not list the GPUs: ...", #41)
    let mutable checkWarnings: string list = []

    let show (content: Control) =
        host.Content <- ScrollViewer(Content = content, HorizontalScrollBarVisibility = Primitives.ScrollBarVisibility.Disabled)

    let failure (title: string) (detail: string) (back: unit -> unit) =
        show (
            stack 14.0 [
                heading title
                card (coloured failBrush detail)
                actions [ accentButton "Back" back ]
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

    // ---------------------------------------------------------------------------------------------- the update
    let checkUpdate () =
        task {
            let! messages, _ = Setup.collect [ "--check-update" ]
            for m in messages do
                match m with
                | Updated u when u.Newer ->
                    ui (fun () ->
                        updateBar.Child <-
                            stack 6.0 [
                                subheading (sprintf "XeStrata %s is out (this is %s)" u.Latest u.Current)
                                text ("To update: " + u.How + ".")
                                actions [ button "Release page" (fun () -> openUrl u.Url) ]
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
        let stepText = subheading "Starting ..."
        let status = text ""
        let bar = ProgressBar(IsIndeterminate = true, Minimum = 0.0, Maximum = 1.0, Height = 6.0)
        let warnings = stack 4.0 []
        let question = stack 8.0 []
        question.IsVisible <- false
        let log = LogView()
        let recent = Collections.Generic.List<string>()   // the says since the last question: a question's menu
        let details = Expander(Header = "Details", Content = log.Control, HorizontalAlignment = HorizontalAlignment.Stretch)
        let mutable run: Setup.Run option = None
        let mutable failed = false
        let cancel = button "Cancel" (fun () -> run |> Option.iter (fun r -> r.Stop()))
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
                    stepText.Text <- sprintf "Step %d: %s" n t
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
                    failure "Setup stopped" (if String.IsNullOrEmpty hint then s else s + "\n\n" + hint) (fun () -> onDone 1)
                | Progress(what, d, total) ->
                    if total > 0L then
                        bar.IsIndeterminate <- false
                        bar.Value <- float d / float total
                        status.Text <- sprintf "%s: %s of %s" what (gb (float d / 1e9)) (gb (float total / 1e9))
                    else
                        status.Text <- sprintf "%s: %s" what (gb (float d / 1e9))
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
        let r = Setup.Run(script, args, onMessage, (fun code -> ui (fun () -> if not failed then onDone code)))
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
                                   if not m.Edits then yield badge mutedBrush "no edits"
                               ]
                               muted (sprintf "%s; download %s, needs ~%s of RAM besides the text model" m.About (gb m.DownloadGb) (gb m.RamGb))
                           ]
                   ) |]
        // each model's size where it has several: Auto (the largest that fits the RAM) or one of them
        let sizes =
            [| for m in models ->
                   let c = ComboBox(ItemsSource = Array.append [| "Auto" |] m.Quants, SelectedIndex = 0, MinWidth = 110.0)
                   let was = current |> Array.tryFind (fun id -> plain id = m.Id && id.Contains ":")
                   was |> Option.iter (fun id -> c.SelectedItem <- id.Split(':')[1])
                   c.IsVisible <- m.Quants.Length > 0
                   c |]
        let uncensored =
            CheckBox(
                Content = text "Text encoder with its refusal direction removed (-uncensored)",
                IsChecked = Nullable(current |> Array.exists (fun c -> c.EndsWith "-uncensored"))
            )
        let defaultBox = ComboBox(MinWidth = 260.0)
        let defaultField = field "Default image model (the others: chosen on the web page or by the API's \"model\")" defaultBox
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
                    match size.SelectedItem with
                    | :? string as q when q <> "Auto" -> id + ":" + q
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
        show (stack 10.0 [ heading "Checking this PC ..."; ProgressBar(IsIndeterminate = true) ])
        task {
            let! messages, code = Setup.collect [ "--describe" ]
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
                    failure "XeStrata could not check this PC" (sprintf "setup.py exited with %d.\n%s" code why) home)
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
                yield subheading "This PC"
                match gpu with
                | Some g -> yield text (sprintf "GPU: %s, %s VRAM" g.Name (gb g.VramGb))
                | None -> yield coloured failBrush "No GPU here can run XeStrata."
                for g in d.Gpus do
                    if not (isNull g.Problem) then
                        yield coloured warnBrush (sprintf "%s: %s" g.Name g.Problem)
                for w in checkWarnings do
                    yield coloured warnBrush ("⚠ " + w)
                yield text (sprintf "RAM: %s" (gb d.RamGb))
                yield
                    text (
                        sprintf "CPU: %s%s" d.Cpu.Name (if d.Cpu.Avx512 then " (AVX-512)" elif d.Cpu.Avx2 then " (AVX2)" else " (no AVX2: not supported)")
                    )
            ]
        let installed =
            [ for i, m in Array.indexed d.Installed ->
                  card (
                      let info =
                          stack 2.0 [
                              subheading m.ModelName
                              muted (
                                  (if m.Context.HasValue then contextLabel m.Context.Value + " of context, " else "")
                                  + sprintf "http://127.0.0.1:%d/" m.Port
                                  + (if m.Images.Length > 0 then ", images: " + String.Join(", ", m.Images) else "")
                              )
                          ]
                      let imagesOk = d.Families |> Array.exists (fun f -> f.Id = m.Family && f.Images)
                      let buttons =
                          actions [
                              if imagesOk && d.Images.Length > 0 then
                                  yield button "Images ..." (fun () -> imagesPage d m)
                              yield button "Measure ..." (fun () -> benchPage d m)
                              yield button "Tune" (fun () -> calibrateModel i m)
                              yield button "Start with options ..." (fun () -> startOptionsPage d i m)
                              yield accentButton "Start" (fun () -> startModel i m [] false)
                          ]
                      stack 8.0 [ info; buttons ]
                  )
                  :> Control ]
        show (
            stack 14.0 [
                yield card pc :> Control
                if not installed.IsEmpty then
                    yield subheading "Installed models"
                    yield! installed
                yield
                    actions [
                        button "Inspect a model file ..." (fun () -> inspectPage d)
                        button "Remove XeStrata's data ..." removeData
                        (if installed.IsEmpty then accentButton else button) "Install a model ..." (fun () -> choosePage d)
                    ]
            ]
        )

    and choosePage (d: Describe) =
        let families = ListBox(ItemsSource = d.Families, MaxHeight = 260.0)
        families.ItemTemplate <-
            FuncDataTemplate<Family>(fun f _ ->
                stack 2.0 [
                    row 6.0 [
                        subheading f.Title
                        if f.Experimental then badge warnBrush "experimental"
                    ]
                    muted (f.By + " - " + f.About)
                ])
        let sizes = ListBox(MaxHeight = 320.0)
        sizes.ItemTemplate <-
            FuncDataTemplate<Model>(fun m _ ->
                stack 2.0 [
                    row 6.0 [
                        yield subheading m.Id
                        if m.Recommended then yield badge okBrush "recommended"
                        if not m.Tested then yield badge mutedBrush "untested"
                        if m.Experimental then yield badge warnBrush "experimental"
                        match m.Fits with
                        | "low-ram" -> yield badge warnBrush "low-RAM mode"
                        | "budget" -> yield badge warnBrush "part from the SSD"
                        | "slow" | "no" -> yield badge failBrush "does not fit the RAM"
                        | _ -> ()
                    ]
                    muted (sprintf "%s; download %s, needs ~%s of RAM" m.About (gb m.DownloadGb) (gb m.RamGb))
                ])
        let contexts = ComboBox(MinWidth = 160.0)
        let gpus = d.Gpus |> Array.filter (fun g -> isNull g.Problem)
        let gpuBox = ComboBox(ItemsSource = [ for g in gpus -> sprintf "%s (%s)" g.Name (gb g.VramGb) ], MinWidth = 260.0)
        gpuBox.SelectedIndex <- (if d.Gpu.HasValue then max 0 (Array.findIndex (fun g -> g.Index = d.Gpu.Value) gpus) else 0)
        let dataDir = TextBox(Text = d.DataDir, MinWidth = 420.0)
        let ggufDir = TextBox(PlaceholderText = "(optional) a folder with the model files already downloaded", MinWidth = 420.0)
        let license = muted ""
        let install = accentButton "Install" ignore
        let problem = coloured failBrush ""
        // setup's other questions (step 2), each shown where setup would ask it
        let pictures = CheckBox(IsChecked = Nullable true, Content = text "Read pictures too (screenshots, photos, scanned pages; a 0.9 GB download)")
        let kv, kvValue =
            choice
                [ "8-bit (recommended)", "int8"
                  "4-bit: half the memory, a little less precise on long documents", "q4_0"
                  "8-bit keys and 4-bit values: about three quarters of the memory", "k8v4"
                  "16-bit: twice the memory of 8-bit", "fp16" ]
                0
        let kvField = field "KV cache precision (the model's memory of the conversation)" kv
        // a family whose KV cache setup does not ask: 16-bit unless one of the precisions it takes is chosen
        let kvLabels =
            dict [ "fp16", "16-bit (default)"; "int8", "8-bit"; "q4_0", "4-bit (Hadamard-rotated)"; "k8v4", "8-bit keys and 4-bit values" ]
        let plainKv = ComboBox(MinWidth = 260.0)
        let plainKvField = field "KV cache precision (the model's memory of the conversation)" plainKv
        let plainKvValue (f: Family) = if plainKv.SelectedIndex >= 0 then f.KvTakes[plainKv.SelectedIndex] else "fp16"
        let rope, ropeValue =
            choice [ "yarn (keeps the trained angles on the high-frequency pairs)", "yarn"; "linear (shrinks every angle)", "linear" ] 0
        let ropeField = field "RoPE extension: a context past the model's trained 262,144 positions rescales its rotary angles" rope
        let projection =
            CheckBox(
                Content =
                    stack 2.0 [
                        text "EXPERIMENTAL - speed projection (off unless you choose it)"
                        muted "A small control vector applied while the model runs (layers 4-44). It changes how the model answers: its package describes it as a refusal-direction projection (the model declines far fewer requests). When on, the web app can switch it off per chat."
                    ]
            )
        let vector = TextBox(PlaceholderText = "(optional) another control vector, a GGUF file", MinWidth = 420.0)
        let vectorField = field "The speed projection's vector" vector
        let images, imagesValue = imagePicker d.Images d.ImageDefault
        let webSearch, webSearchValue =
            choice
                [ "Off", "no"
                  "DeepSeek's search API: your API key, billed by DeepSeek per search (USD 0.75 per 1,000); the search words go to DeepSeek", "deepseek" ]
                (match d.WebSearch with "deepseek" -> 1 | _ -> 0)
        let deepseekKey =
            TextBox(PasswordChar = '*', MinWidth = 420.0,
                    PlaceholderText = (if d.WebSearch = "deepseek" then "set (type to replace)" else "your DeepSeek API key"))
        let deepseekField = field "DeepSeek API key (kept in your settings, never shown)" deepseekKey
        deepseekField.IsVisible <- webSearchValue () = "deepseek"
        webSearch.SelectionChanged.Add(fun _ -> deepseekField.IsVisible <- webSearchValue () = "deepseek")
        let webSearchField = field "Web search, for every model: the model searches when a request asks for it (the web page's Search switch, the API's web search tools)" webSearch
        let imagesField = field "Image generation: models to make images with, besides the text model" images
        // the options setup takes without asking, under Advanced
        let reader, readerValue =
            choice [ "For this GPU (on the GPU from 24 GB of VRAM, else on the CPU)", "yes"; "On the GPU (the CPU when it cannot start there)", "gpu"; "On the CPU", "cpu" ] 0
        let readerField = field "The picture reader" reader
        let visionTokens = TextBox(PlaceholderText = "1024 on the GPU, 300 on the CPU", MinWidth = 260.0)
        let visionTokensField = field "The most image tokens a picture becomes (more reads small text better, and takes longer)" visionTokens
        let ropeScale = TextBox(PlaceholderText = "the context over 262,144", MinWidth = 260.0)
        let ropeScaleField = field "RoPE extension factor (1 or more)" ropeScale
        let lowRam, lowRamValue =
            choice
                [ "Auto: when the experts do not fit the RAM", "auto"
                  "On", "on"
                  "Off", "off"
                  "On, the experts the GPU does not hold copied into RAM", "resident"
                  "On, the experts the GPU does not hold read through the OS file cache", "mmap" ]
                0
        let budget = TextBox(PlaceholderText = "setup's choice for this RAM", MinWidth = 260.0)
        let budgetField = field "GiB of the experts kept in RAM (the rest read from the SSD)" budget
        let ramReserve = TextBox(PlaceholderText = "(none) GiB of RAM to leave free for other programs", MinWidth = 260.0)
        let parallelBox = TextBox(PlaceholderText = "1: one request at a time", MinWidth = 260.0)
        let replicas = TextBox(PlaceholderText = "1", MinWidth = 260.0)
        let source, sourceValue =
            choice [ "Hugging Face (default)", ""; "ModelScope (mainland China: the same files)", "modelscope" ] 0
        let modelsDir = TextBox(PlaceholderText = "<data folder>\\models", MinWidth = 420.0)
        let moveData = CheckBox(Content = text "Move the model files already on this PC into the data folder")
        let rememberData = CheckBox(Content = text "Remember the data folder for later downloads without moving anything")
        let noWarmup = CheckBox(Content = text "Skip the first start that compiles the GPU code (the first request then takes ~30 s more)")
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
                license.Text <- (if String.IsNullOrEmpty f.License then "" else "Its license: " + f.License)
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
                             optionOf "--vision-tokens" visionTokens (isInt 1) "a number of image tokens, 1 or more"
                         else Result.Ok [])
                        (if ctx > 262144 then optionOf "--rope-scale" ropeScale (isNumber 1.0 1e6) "a number, 1 or more" else Result.Ok [])
                        Result.Ok(match lowRamValue () with "auto" -> [] | v -> [ "--low-ram"; v ])
                        Result.Ok(if f.Kv then [] else match plainKvValue f with "fp16" -> [] | k -> [ "--kv"; k ])
                        (if m.Budget then optionOf "--resident-budget-gib" budget (isNumber 1.0 1e6) "a number of GiB" else Result.Ok [])
                        optionOf "--ram-reserve-gib" ramReserve (isNumber 0.0 1e6) "a number of GiB, 0 or more"
                        optionOf "--parallel" parallelBox (isInt 1) "a number of requests, 1 or more"
                        optionOf "--replicas" replicas (isInt 1) "a number of engines, 1 or more"
                        Result.Ok(match sourceValue () with "" -> [] | v -> [ "--source"; v ])
                        Result.Ok(match trimmed modelsDir with "" -> [] | p -> [ "--models-dir"; p ])
                        Result.Ok(flag "--move-data" moveData @ flag "--set-default-data-dir" rememberData @ flag "--no-warmup" noWarmup)
                        Result.Ok(if withImages then [ for name, v in imageValues () do yield! [ "--image-" + name; v ] ] else [])
                        serverArgs ()
                    ]
                let key = trimmed deepseekKey
                match advanced with
                | Error why -> problem.Text <- why
                | Result.Ok _ when webSearchValue () = "deepseek" && key = "" && d.WebSearch <> "deepseek" ->
                    problem.Text <- "Web search with DeepSeek needs your DeepSeek API key."
                | Result.Ok more ->
                    // the key reaches setup in its environment, never on its command line (process lists show that)
                    Environment.SetEnvironmentVariable("XESTRATA_DEEPSEEK_API_KEY", (if key = "" then null else key))
                    let args =
                        [ "--setup"; "--yes"; "--no-start"; "--family"; f.Id; "--model"; m.Id; "--data-dir"; dataDir.Text ]
                        @ questions
                        @ (if gpus.Length > 0 then [ "--gpu"; string gpus[max 0 gpuBox.SelectedIndex].Index ] else [])
                        @ (if String.IsNullOrWhiteSpace ggufDir.Text then [] else [ "--gguf-dir"; ggufDir.Text ])
                        @ more
                    installPage (f.Title + " " + m.Id) args
            | _ -> ())
        families.SelectedIndex <- 0
        let advanced =
            Expander(
                Header = "Advanced",
                HorizontalAlignment = HorizontalAlignment.Stretch,
                Content =
                    stack 10.0 [
                        readerField
                        visionTokensField
                        ropeScaleField
                        vectorField
                        plainKvField
                        field "Low-RAM mode: the experts read from the model files instead of all copied into RAM" lowRam
                        budgetField
                        field "RAM left free for other programs, in GiB (the model's mode, its experts in RAM and the image models' sizes are chosen with that much less; slower when the experts are then read from the SSD)" ramReserve
                        field "Requests decoded together (each takes VRAM from the expert cache)" parallelBox
                        field "With a layer split: independent engines, each on an equal group of the GPUs" replicas
                        field "Where the model files come from" source
                        field "Where the model files go" (row 8.0 [ modelsDir; button "Browse ..." (fun () -> pickFolder "Model files folder" modelsDir) ])
                        moveData
                        rememberData
                        noWarmup
                        imageOptions
                        serverControl
                    ]
            )
        show (
            stack 14.0 [
                heading "Install a model"
                field "Model" families
                field "Size" sizes
                license
                field "Context: how much text the model sees at once (longer takes more VRAM)" contexts
                if gpus.Length > 1 then field "GPU" gpuBox
                field "Data folder: where the model files go (tens of GB)" (row 8.0 [ dataDir; button "Browse ..." (fun () -> pickFolder "Data folder" dataDir) ])
                field "Model files you already have" (row 8.0 [ ggufDir; button "Browse ..." (fun () -> pickFolder "Model files" ggufDir) ])
                pictures
                kvField
                ropeField
                projection
                imagesField
                webSearchField
                deepseekField
                advanced
                problem
                actions [ button "Back" (fun () -> homePage d); install ]
            ]
        )

    and installPage (what: string) (args: string list) =
        runPage ("Installing " + what) args (fun _ -> None) ignore (fun _ -> home ()) |> ignore

    /// An installed model's image generation: its image models chosen again (none: taken out), the files downloaded
    /// and its config rewritten by tools/image/install.py; the server takes it at its next start.
    and imagesPage (d: Describe) (m: Installed) =
        let picker, value = imagePicker d.Images m.Images
        let options, optionValues = imageRunOptions ()
        let apply () =
            let v = value () |> Option.defaultValue "no"
            let more = if v = "no" then [] else [ for name, x in optionValues () do yield! [ "--" + name; x ] ]
            runScriptPage Places.imageInstallPy ("Image generation for " + m.ModelName)
                ([ "--config"; m.Config; "--model"; v; "--yes" ] @ more) (fun _ -> None) ignore (fun _ -> home ())
            |> ignore
        show (
            stack 14.0 [
                heading ("Image generation for " + m.ModelName)
                muted "Check the image models to make images with (none: image generation off). Their files are downloaded, and the model takes them at its next start."
                picker
                Expander(Header = "Advanced", Content = options, HorizontalAlignment = HorizontalAlignment.Stretch)
                actions [ button "Back" (fun () -> homePage d); accentButton "Apply" apply ]
            ]
        )

    /// A start of an installed model with setup's options for a start (the GPUs, the port, who may connect, ...).
    and startOptionsPage (d: Describe) (index: int) (m: Installed) =
        let gpus = d.Gpus |> Array.filter (fun g -> isNull g.Problem)
        let control, args = runOptions gpus
        let one, oneValue =
            choice (("As set up", "") :: [ for g in gpus -> sprintf "Only %s, this time" g.Name, string g.Index ]) 0
        let browser = CheckBox(IsChecked = Nullable true, Content = text "Open the chat in the browser when the model is ready")
        let problem = coloured failBrush ""
        let start () =
            match args () with
            | Error why -> problem.Text <- why
            | Result.Ok more ->
                let gpu = match oneValue () with "" -> [] | i -> [ "--gpu"; i ]
                startModel index m (gpu @ more) (browser.IsChecked = Nullable true)
        show (
            stack 14.0 [
                yield heading ("Start " + m.ModelName) :> Control
                if gpus.Length > 1 then yield field "GPU" one :> Control
                yield control :> Control
                yield browser :> Control
                yield problem :> Control
                yield actions [ button "Back" (fun () -> homePage d); accentButton "Start" start ] :> Control
            ]
        )

    /// setup --calibrate for an installed model: setup asks whether to measure (it is rarely worth the time).
    and calibrateModel (index: int) (m: Installed) =
        let which = [ "--model"; m.Model; "--family"; m.Family ] |> fun a -> if String.IsNullOrEmpty m.Model then [] else a
        let answer (q: string) = if q.StartsWith "Tune which one" then Some(string (index + 1)) else None
        runPage ("Tuning " + m.ModelName) ([ "--calibrate"; "--no-start" ] @ which) answer ignore (fun _ -> home ()) |> ignore

    /// setup --bench (or upstream's benchmark mode) for an installed model: the record goes to the data folder's
    /// records/.
    and benchPage (d: Describe) (m: Installed) =
        let verbose = CheckBox(Content = text "Where the time goes in more detail (per-stage timings, the kernels, the GPU's load)")
        let rounds = TextBox(Text = "2", MinWidth = 120.0)
        let temperature = TextBox(Text = "0.7", MinWidth = 120.0)
        let compat = CheckBox(Content = text "Measure the way upstream Strata's benchmark mode does, through the server's API (3-10 minutes)")
        let long = CheckBox(Content = text "With upstream's long prompts (--long)")
        let clients = TextBox(PlaceholderText = "1", MinWidth = 120.0)
        let own = stack 10.0 [ verbose; field "Rounds (each case measured this many times)" rounds; field "Temperature of the sampled runs (0 to 2, 0 = greedy only)" temperature ]
        let upstream = stack 10.0 [ long; field "Clients at once" clients ]
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
                        optionOf "--clients" clients (isInt 1) "a number of clients, 1 or more"
                    ]
                else
                    collectArgs [
                        Result.Ok((if verbose.IsChecked = Nullable true then [ "--bench-verbose" ] else [ "--bench" ]) @ which)
                        optionOf "--bench-rounds" rounds (isInt 1) "a number of rounds, 1 or more"
                        optionOf "--bench-temperature" temperature (fun t -> isNumber 0.0 2.0 t && abs (float t * 10.0 - round (float t * 10.0)) < 1e-6) "0 to 2 in steps of 0.1"
                    ]
            match args with
            | Error why -> problem.Text <- why
            | Result.Ok a -> runPage ("Measuring " + m.ModelName) a (fun _ -> None) ignore (fun _ -> home ()) |> ignore
        show (
            stack 14.0 [
                heading ("Measure " + m.ModelName)
                muted "Measures this model's speed on this PC (about 40 minutes; a running XeStrata is stopped first) and writes a record to the data folder's records/. Nothing is sent anywhere."
                compat
                own
                upstream
                problem
                actions [ button "Back" (fun () -> homePage d); accentButton "Measure" start ]
            ]
        )

    /// setup --inspect: what a GGUF is and whether XeStrata runs it, from its headers only (no download).
    and inspectPage (d: Describe) =
        let source = TextBox(PlaceholderText = "a file, a folder, a URL, hf:owner/repo or ms:owner/repo", MinWidth = 420.0)
        let variant = TextBox(PlaceholderText = "(optional) a variant name, e.g. IQ3_XXS", MinWidth = 260.0)
        let output = LogView()
        let go = accentButton "Inspect" ignore
        go.Click.Add(fun _ ->
            match trimmed source with
            | "" -> ()
            | src ->
                go.IsEnabled <- false
                output.Add("> " + src)
                task {
                    let! messages, _ =
                        Setup.collect ([ "--inspect"; src ] @ (match trimmed variant with "" -> [] | v -> [ v ]))
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
                heading "Inspect a model file"
                muted "What a GGUF is and whether XeStrata runs it, read from its headers only (nothing is downloaded or installed)."
                field "The model file" (row 8.0 [ source; button "Browse ..." (fun () -> pickFolder "A folder with the model files" source) ])
                field "Variant" variant
                output.Control
                actions [ button "Back" (fun () -> homePage d); go ]
            ]
        )

    /// setup --remove-data: the folders listed with their sizes, deleted after a yes pressed at setup's question.
    and removeData () =
        runPage "Remove XeStrata's data" [ "--remove-data" ] (fun _ -> None) ignore (fun _ -> home ()) |> ignore

    and startModel (index: int) (m: Installed) (extra: string list) (openBrowser: bool) =
        let stopButton = button "Stop" ignore
        let page (url: string) =
            if openBrowser then openUrl (url + "/")
            show (
                stack 14.0 [
                    heading m.ModelName
                    card (
                        stack 8.0 [
                            coloured okBrush "Running"
                            text ("Chat in the browser: " + url + "/")
                            text ("API (OpenAI): " + url + "/v1   (Anthropic: " + url + "/v1/messages)")
                        ]
                    )
                    actions [ stopButton; accentButton "Open chat" (fun () -> openUrl (url + "/")) ]
                ]
            )
        // several installed models: setup asks which one, in describe's order (the newest first)
        let which (q: string) = if q.StartsWith "Which one" then Some(string (index + 1)) else None
        let r =
            runPage ("Starting " + m.ModelName) ([ "--no-browser" ] @ extra) which (fun url -> page url) (fun _ ->
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
