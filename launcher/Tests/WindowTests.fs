// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// The launcher's window without a screen: each page built from describe.json, filled in, scrolled and clicked as a
/// user would, with setup's runs played by FakeRun (what setup would say, and what the window answered).
module Xestrata.Launcher.Tests.WindowTests

open System
open System.IO
open System.Text.RegularExpressions
open System.Threading.Tasks
open Avalonia
open Avalonia.Controls
open Avalonia.Headless
open Avalonia.Headless.XUnit
open Avalonia.Input
open Avalonia.Interactivity
open Avalonia.LogicalTree
open Avalonia.Threading
open Avalonia.Themes.Fluent
open Xunit
open Xestrata.Launcher
open Xestrata.Launcher.Protocol

type TestApp() =
    inherit Application()

    override this.Initialize() = this.Styles.Add(FluentTheme())

    static member BuildAvaloniaApp() =
        AppBuilder.Configure<TestApp>().UseHeadless(AvaloniaHeadlessPlatformOptions())

[<assembly: AvaloniaTestApplication(typeof<TestApp>)>]
[<assembly: AvaloniaTestFramework>]
do ()

let private described =
    lazy
        (match Protocol.parse ((File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "describe.json"))).ReplaceLineEndings " ") with
         | Described d -> d
         | m -> failwithf "describe.json is not setup's description: %A" m)

/// A run of setup the window follows: what it was started with, the answers it was given, and setup's side.
type FakeRun(script: string, args: string list, onMessage: Message -> unit, onExit: int -> unit) =
    member val Answers = ResizeArray<string>()
    member val Stopped = false with get, set
    member _.Script = script
    member _.Args = args
    /// setup says `m`.
    member _.Say(m: Message) = onMessage m
    /// setup ends with `code`.
    member _.Exit(code: int) = onExit code

    interface Setup.IRun with
        member this.Answer answer = this.Answers.Add answer
        member this.Stop() = this.Stopped <- true

/// setup as the tests play it: --describe answered with `describe`, the other short runs by `others`, and every run
/// the window starts kept.
type FakeSetup(describe: Describe, others: string list -> Message list) =
    member val Runs = ResizeArray<FakeRun>()
    member val Collected = ResizeArray<string list>()
    member this.Last = this.Runs[this.Runs.Count - 1]

    member this.Access: Setup.Access =
        { Collect =
            fun args ->
                this.Collected.Add args
                let messages = match args with [ "--describe" ] -> [ Described describe ] | _ -> others args
                Task.FromResult((messages, 0))
          Start =
            fun script args onMessage onExit ->
                let r = FakeRun(script, args, onMessage, onExit)
                this.Runs.Add r
                r :> Setup.IRun }

let private pump () =
    Dispatcher.UIThread.RunJobs()
    AvaloniaHeadlessPlatform.ForceRenderTimerTick()

/// The window's thread run until `find` finds something (a page comes after a task of setup's), for at most 5 s.
let private waitFor (what: string) (find: unit -> 'T option) =
    let until = DateTime.UtcNow.AddSeconds 5.0
    let rec go () =
        pump ()
        match find () with
        | Some x -> x
        | None when DateTime.UtcNow < until ->
            Threading.Thread.Sleep 10
            go ()
        | None -> failwithf "no %s on the page" what
    go ()

let private all<'T> (w: Window) : 'T list =
    w.GetLogicalDescendants() |> Seq.cast<obj> |> Seq.choose (function :? 'T as x -> Some x | _ -> None) |> List.ofSeq

let private visible (c: Control) = c.GetSelfAndLogicalAncestors() |> Seq.forall (function :? Control as x -> x.IsVisible | _ -> true)

let private texts (w: Window) = all<TextBlock> w |> List.filter visible |> List.map (fun t -> t.Text) |> List.filter (isNull >> not)

let private shows (w: Window) (text: string) = texts w |> List.exists (fun t -> t.Contains text)

let private findButton (w: Window) (label: string) =
    all<Button> w |> List.tryFind (fun b -> visible b && (match b.Content with :? string as s -> s = label | _ -> false))

let private click (w: Window) (label: string) =
    let b = waitFor ("the button " + label) (fun () -> findButton w label)
    b.RaiseEvent(RoutedEventArgs(Button.ClickEvent))
    pump ()

/// The control under a field's label (Views.field: the label above it).
let private fieldOf<'T when 'T :> Control> (w: Window) (label: string) : 'T =
    waitFor ("the field " + label) (fun () ->
        all<StackPanel> w
        |> List.tryPick (fun p ->
            match List.ofSeq p.Children with
            | [ :? TextBlock as t; c ] when t.Text = label ->
                match c with
                | :? 'T as x -> Some x
                | :? StackPanel as r -> r.Children |> Seq.tryPick (function :? 'T as x -> Some x | _ -> None)
                | _ -> None
            | _ -> None))

let private press (w: Window) (key: Key) (physical: PhysicalKey) =
    w.KeyPress(key, RawInputModifiers.None, physical, null)
    w.KeyRelease(key, RawInputModifiers.None, physical, null)
    pump ()

let private tr key english = I18n.tr key english

/// One of setup's options as describe.json has it.
let private spec (name: string) = described.Value.Options |> Array.find (fun o -> o.Name = name)

/// The label of an option's field.
let private labelOf (name: string) = Fields.label (spec name)

/// An option's field: the control under its label.
let private optionField<'T when 'T :> Control> (w: Window) (name: string) : 'T = fieldOf<'T> w (labelOf name)

/// A choice of an option's field chosen by its value.
let private pick (w: Window) (name: string) (value: string) =
    let o = spec name
    let c = optionField<ComboBox> w name
    let english = o.Choices |> Array.find (fun x -> x[0] = value)
    c.SelectedItem <- Fields.choiceLabel o value english[1]
    Assert.Equal(box (Fields.choiceLabel o value english[1]), c.SelectedItem)
    pump ()

/// A window on `setup`, at its home page.
let private open' (setup: FakeSetup) =
    let w = new MainWindow.MainWindow(setup.Access)
    w.Show()
    waitFor "the home page" (fun () -> findButton w (tr "home.install" "Install a model ...")) |> ignore
    w

let private english () = I18n.choose "en"

/// The install page's choices: the model, its version and its quantization, and the list of sizes.
type private Levels = { Model: ComboBox; Version: ComboBox; Quantization: ComboBox; Sizes: ListBox }

let private installPage (w: Window) =
    click w (tr "home.install" "Install a model ...")
    let sizes = waitFor "the text model sizes" (fun () ->
        all<ListBox> w |> List.tryFind (fun s -> s.SelectedItem :? Model))
    { Model = fieldOf<ComboBox> w (tr "install.model" "Model")
      Version = fieldOf<ComboBox> w (tr "install.version" "Version: the original, or whose changed copy")
      Quantization = fieldOf<ComboBox> w (tr "install.quantization" "Quantization: whose model files")
      Sizes = sizes }

/// Every model, version and quantization chosen in turn: `each` gets the family's title the page shows.
let private everyChoice (w: Window) (lv: Levels) (each: string -> unit) =
    for i in 0 .. lv.Model.ItemCount - 1 do
        lv.Model.SelectedIndex <- i
        pump ()
        for j in 0 .. lv.Version.ItemCount - 1 do
            lv.Version.SelectedIndex <- j
            pump ()
            for k in 0 .. lv.Quantization.ItemCount - 1 do
                lv.Quantization.SelectedIndex <- k
                pump ()
                // the family's title as the page shows it whole (the model's own name is a shorter title)
                let on = texts w
                let shown = described.Value.Families |> Array.filter (fun f -> List.contains f.Title on)
                each (match shown with [||] -> failwith "no family on the page" | fs -> (fs |> Array.maxBy (fun f -> f.Title.Length)).Title)

/// The page's choice of a family by its id (its model, version and quantization).
let private choose (w: Window) (lv: Levels) (id: string) =
    let d = described.Value
    let f = d.Families |> Array.find (fun f -> f.Id = id)
    lv.Model.SelectedItem <- (d.Bases |> Array.find (fun t -> t.Id = f.Base)).Title
    pump ()
    if lv.Version.ItemCount > 1 then lv.Version.SelectedItem <- d.Variants[f.Variant]
    pump ()
    if lv.Quantization.ItemCount > 1 then lv.Quantization.SelectedItem <- d.Quantizers[f.Quantizer]
    pump ()
    f

let private none (_: string list) : Message list = []

let private contains (part: string list) (args: string list) =
    let n = part.Length
    args |> List.windowed n |> List.exists ((=) part)

// ------------------------------------------------------------------------------------------------- the home page

[<AvaloniaFact>]
let ``the home page shows this PC and the installed model`` () =
    english ()
    let d = described.Value
    let w = open' (FakeSetup(d, none))
    let g = d.Gpus[0]
    Assert.True(shows w g.Name, "the GPU")
    Assert.True(shows w "RAM: ", "the RAM")
    Assert.True(shows w d.Cpu.Name, "the CPU")
    Assert.True(shows w d.Installed[0].ModelName, "the installed model")
    for label in [ "Images ..."; "Measure ..."; "Tune"; "Start with options ..."; "Start" ] do
        Assert.True((findButton w label).IsSome, label)
    Assert.Equal("xe" + d.Version, (all<TextBlock> w |> List.find (fun t -> t.Text = "xe" + d.Version)).Text)

[<AvaloniaFact>]
let ``a PC without a GPU is told so`` () =
    english ()
    let d = { described.Value with Gpu = Nullable(); Gpus = [||] }
    let w = open' (FakeSetup(d, none))
    Assert.True(shows w "No GPU here can run XeStrata.")

[<AvaloniaFact>]
let ``a newer release shows the update bar`` () =
    english ()
    let update = { Current = "0.2.8-1"; Latest = "0.2.9-1"; Newer = true; Url = "https://example.org/r"; How = "unpack the new 7z" }
    let w = open' (FakeSetup(described.Value, function [ "--check-update" ] -> [ Updated update ] | _ -> []))
    waitFor "the update bar" (fun () -> if shows w "XeStrata 0.2.9-1 is out (this is 0.2.8-1)" then Some() else None)
    Assert.True(shows w "To update: unpack the new 7z.")

[<AvaloniaFact>]
let ``setup that cannot describe the PC leaves a page that says why`` () =
    english ()
    let fake = FakeSetup(described.Value, none)
    let access = { fake.Access with Collect = fun _ -> Task.FromResult(([ Fail("no Python", "unpack the 7z again") ], 1)) }
    let w = new MainWindow.MainWindow(access)
    w.Show()
    waitFor "the failure" (fun () -> if shows w "XeStrata could not check this PC" then Some() else None)
    Assert.True(shows w "no Python unpack the 7z again")

// ------------------------------------------------------------------------------------------------ the install page

/// #50: a list scrolled by the wheel or the down key recycles its items' containers, and Avalonia then builds an
/// item's template for null; the window closed on the second turn of the wheel.
[<AvaloniaFact>]
let ``the install page's lists scroll to their ends`` () =
    english ()
    let w = open' (FakeSetup(described.Value, none))
    let lv = installPage w
    choose w lv "unsloth" |> ignore                          // the family of the most sizes (10)
    let sizes = lv.Sizes
    let centre = sizes.TranslatePoint(Point(sizes.Bounds.Width / 2.0, sizes.Bounds.Height / 2.0), w)
    Assert.True(centre.HasValue, "the list is not on the page")
    for _ in 1..30 do
        w.MouseWheel(centre.Value, Vector(0.0, -1.0))
        pump ()
    for _ in 1..30 do
        w.MouseWheel(centre.Value, Vector(0.0, 1.0))
        pump ()
    sizes.SelectedIndex <- 0
    (sizes.ContainerFromIndex 0).Focus() |> ignore
    for _ in 1 .. sizes.ItemCount do
        press w Key.Down PhysicalKey.ArrowDown
    Assert.Equal(sizes.ItemCount - 1, sizes.SelectedIndex)
    for _ in 1 .. sizes.ItemCount do
        press w Key.Up PhysicalKey.ArrowUp
    Assert.Equal(0, sizes.SelectedIndex)

/// The model setup recommends is chosen first, with its recommended size's quantization and that size; a level of
/// one choice is not shown.
[<AvaloniaFact>]
let ``the install page starts at setup's recommendation`` () =
    english ()
    let d = described.Value
    let w = open' (FakeSetup(d, none))
    let lv = installPage w
    let f = d.Families |> Array.find (fun f -> f.Id = d.RecommendedFamily)
    Assert.Equal(box (d.Bases |> Array.find (fun t -> t.Id = f.Base)).Title, lv.Model.SelectedItem)
    Assert.Equal(f.RecommendedSize, (lv.Sizes.SelectedItem :?> Model).Id)
    Assert.True(shows w f.Title)
    choose w lv "llmjp" |> ignore
    Assert.False(visible lv.Version, "llm-jp has one version")
    Assert.False(visible lv.Quantization, "llm-jp has one quantization")
    choose w lv "qwen" |> ignore
    Assert.True(visible lv.Quantization, "Qwen3.8-Flash-Next has four quantizations")
    Assert.Equal(4, lv.Quantization.ItemCount)

/// A size shows its model files to a tenth of a GB at any size (setup's files_gb: 83.5 and 83.6 GB are two files).
[<AvaloniaFact>]
let ``a size shows its files to a tenth of a GB`` () =
    english ()
    let d = described.Value
    let w = open' (FakeSetup(d, none))
    let lv = installPage w
    let m = lv.Sizes.SelectedItem :?> Model
    Assert.True(m.DownloadGb >= 10.0, sprintf "%s: a size of 10 GB or more is needed here" m.Id)
    Assert.True(shows w (sprintf "download %s GB," (m.DownloadGb.ToString("0.0"))), sprintf "%s: %g GB" m.Id m.DownloadGb)

/// Every family, each of its sizes and each context the install page offers, the fields setup would ask shown.
[<AvaloniaFact>]
let ``every family, size and context can be chosen`` () =
    english ()
    let d = described.Value
    let w = open' (FakeSetup(d, none))
    let lv = installPage w
    let sizes = lv.Sizes
    let contexts = fieldOf<ComboBox> w "Context: how much text the model sees at once (longer takes more VRAM)"
    let imagesField = fieldOf<StackPanel> w "Image generation: models to make images with, besides the text model"
    let seen = Collections.Generic.HashSet<string>()
    everyChoice w lv (fun title ->
        let f = d.Families |> Array.find (fun f -> f.Title = title)
        seen.Add f.Id |> ignore
        let m = sizes.SelectedItem :?> Model
        // the sizes of the quantization chosen, its recommended one chosen
        Assert.True(m.Recommended, sprintf "%s: %s is not the recommended size" f.Id m.Id)
        for j in 0 .. sizes.ItemCount - 1 do
            sizes.SelectedIndex <- j
            pump ()
            for k in 0 .. contexts.ItemCount - 1 do
                contexts.SelectedIndex <- k
                pump ()
        // the image models are offered for every family when the PC has a GPU (setup_plain's included)
        Assert.True(visible imagesField, f.Id + ": no image models offered"))
    Assert.Equal(d.Families.Length, seen.Count)

/// The install button passes the page's choices to setup as its options, and the run's page follows setup.
[<AvaloniaFact>]
let ``an install passes the choices to setup and follows its run`` () =
    english ()
    let d = described.Value
    let setup = FakeSetup(d, none)
    let w = open' setup
    let lv = installPage w
    choose w lv "qwen" |> ignore
    let m = lv.Sizes.SelectedItem :?> Model
    (optionField<TextBox> w "port").Text <- "8100"
    (optionField<TextBox> w "vram-reserve-mib").Text <- "512"
    click w "Install"
    let r = setup.Last
    Assert.Equal(Places.setupPy, r.Script)
    for part in
        [ [ "--setup"; "--yes"; "--no-start" ]
          [ "--family"; "qwen"; "--model"; m.Id ]
          [ "--data-dir"; d.DataDir ]
          [ "--port"; "8100" ]
          [ "--vram-reserve-mib"; "512" ]
          [ "--web-search"; "no" ] ] do
        Assert.True(contains part r.Args, sprintf "%A not in %A" part r.Args)
    Assert.True(shows w ("Installing Qwen3.8-Flash-Next " + m.Id))
    // setup's steps, progress, warnings and a question, as its --json says them
    r.Say(Step(3, "The engine"))
    r.Say(Progress("Downloading", 2_000_000_000L, 8_000_000_000L))
    pump ()
    Assert.True(shows w "Step 3: The engine")
    Assert.True(shows w "Downloading: 2.0 GB of 8.0 GB")
    r.Say(Warn "a warning")
    r.Say(Say "the list of choices")
    r.Say(Ask("Go on?", [ "yes"; "no" ], "yes"))
    pump ()
    Assert.True(shows w "⚠ a warning")
    Assert.True(shows w "Go on?")
    Assert.True(shows w "the list of choices")
    click w "no"
    Assert.Equal<string list>([ "no" ], List.ofSeq r.Answers)
    click w "Cancel"
    Assert.True(r.Stopped)
    r.Exit 0
    waitFor "the home page again" (fun () -> findButton w (tr "home.install" "Install a model ...")) |> ignore

/// Every option setup describes is a field: on the install page (its Advanced included), and those of a start and of
/// the images on their own pages.
[<AvaloniaFact>]
let ``every option setup describes is a field`` () =
    english ()
    let d = described.Value
    let present (w: Window) name = all<TextBlock> w |> List.exists (fun t -> t.Text = labelOf name)
    let w = open' (FakeSetup(d, none))
    installPage w |> ignore
    for o in d.Options do
        Assert.True(present w o.Name, o.Name + " is not on the install page")
    click w "Back"
    for button, place in [ "Start with options ...", "start"; "Images ...", "images" ] do
        click w button
        for o in d.Options do
            if o.Place = place then Assert.True(present w o.Name, sprintf "%s is not on the page of %s" o.Name button)
        click w "Back"

/// A field is shown where setup would ask its question, with the family's own choices: the KV cache's precision above
/// 8K of context, and a plain family's precisions, 16-bit first.
[<AvaloniaFact>]
let ``an option is shown where setup asks it, with the family's choices`` () =
    english ()
    let d = described.Value
    let w = open' (FakeSetup(d, none))
    let lv = installPage w
    choose w lv "qwen" |> ignore
    let contexts = fieldOf<ComboBox> w "Context: how much text the model sees at once (longer takes more VRAM)"
    let kv = optionField<ComboBox> w "kv"
    contexts.SelectedItem <- Views.contextLabel 8192
    pump ()
    Assert.False(visible kv, "the KV cache asked at 8K")
    contexts.SelectedItem <- Views.contextLabel 32768
    pump ()
    Assert.True(visible kv, "the KV cache not asked at 32K")
    Assert.Equal(4, kv.ItemCount)
    let f = choose w lv "llmjp"
    pump ()
    Assert.True(visible kv, "llm-jp's precisions not offered")
    Assert.Equal(f.KvTakes.Length, kv.ItemCount)
    Assert.Equal(box (Fields.choiceLabel (spec "kv") "fp16" "16-bit: twice the memory of 8-bit"), kv.SelectedItem)

[<AvaloniaFact>]
let ``an option the page cannot take is refused before setup runs`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    installPage w |> ignore
    for text, why in [ "abc", "--port takes a port number, 1 to 65535."; "70000", "--port takes a port number, 1 to 65535." ] do
        (optionField<TextBox> w "port").Text <- text
        click w "Install"
        Assert.True(shows w why, text)
    (optionField<TextBox> w "port").Text <- ""
    (optionField<TextBox> w "parallel").Text <- "0"
    click w "Install"
    Assert.True(shows w "--parallel takes a number of requests, 1 or more.")
    Assert.Empty setup.Runs

[<AvaloniaFact>]
let ``a web search API needs its key`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    installPage w |> ignore
    pick w "web-search" "deepseek"
    click w "Install"
    Assert.True(shows w "Web search with DeepSeek needs your DeepSeek API key.")
    Assert.Empty setup.Runs

[<AvaloniaFact>]
let ``setup's failure is shown with its hint`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    installPage w |> ignore
    click w "Install"
    setup.Last.Say(Fail("the download failed", "try again later"))
    pump ()
    Assert.True(shows w "Setup stopped")
    Assert.True(shows w "the download failed\n\ntry again later")
    click w "Back"
    waitFor "the home page again" (fun () -> findButton w (tr "home.install" "Install a model ...")) |> ignore

[<AvaloniaFact>]
let ``a nonzero exit without a failure message is shown`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    installPage w |> ignore
    click w "Install"
    let before = setup.Runs.Count
    setup.Last.Exit 3
    pump ()
    Assert.True(shows w "Setup stopped")
    Assert.True(shows w "setup.py exited with 3.")
    Assert.Equal(before, setup.Runs.Count)   // not back at the home page until the person has read it
    click w "Back"
    waitFor "the home page again" (fun () -> findButton w (tr "home.install" "Install a model ...")) |> ignore

// --------------------------------------------------------------------------------------------- an installed model

[<AvaloniaFact>]
let ``a start follows setup until the server is ready, and stops it`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    click w "Start"
    let r = setup.Last
    Assert.Equal<string list>([ "--no-browser" ], r.Args)
    r.Say(Log "ready: http://127.0.0.1:8095/v1 (32K)")
    pump ()
    Assert.True(shows w "Running")
    Assert.True(shows w "Chat in the browser: http://127.0.0.1:8095/")
    Assert.True(shows w "API (OpenAI): http://127.0.0.1:8095/v1   (Anthropic: http://127.0.0.1:8095/v1/messages)")
    click w "Stop"
    Assert.True(r.Stopped)
    r.Exit 0
    waitFor "the home page again" (fun () -> findButton w (tr "home.install" "Install a model ...")) |> ignore

[<AvaloniaFact>]
let ``a start with options passes them`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    click w "Start with options ..."
    (optionField<TextBox> w "port").Text <- "x"
    click w "Start"
    Assert.True(shows w "--port takes a port number, 1 to 65535.")
    Assert.Empty setup.Runs
    (optionField<TextBox> w "port").Text <- "8200"
    (optionField<TextBox> w "api-key").Text <- "k1"
    pick w "host" "0.0.0.0"
    pick w "script-guard" "ja"
    pick w "ple-io" "ram"
    click w "Start"
    let args = setup.Last.Args
    for part in
        [ [ "--port"; "8200" ]; [ "--host"; "0.0.0.0" ]; [ "--api-key"; "k1" ]; [ "--script-guard"; "ja" ]
          [ "--ple-io"; "ram" ] ] do
        Assert.True(contains part args, sprintf "%A not in %A" part args)

/// The options left as they are pass nothing: setup keeps what was set before.
[<AvaloniaFact>]
let ``a start with the options untouched passes none of them`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    click w "Start with options ..."
    click w "Start"
    let args = setup.Last.Args
    for name in [ "--script-guard"; "--ple-io"; "--port"; "--host" ] do
        Assert.DoesNotContain(name, args)

[<AvaloniaFact>]
let ``an installed model's image models are set by install.py`` () =
    english ()
    let d = described.Value
    let setup = FakeSetup(d, none)
    let w = open' setup
    click w "Images ..."
    click w "Add to selection"
    // the image jobs on the GPU of the list (by its PCI address), and SAM 3.1's license agreed to
    pick w "image-gpu" "same"
    let sam3 = all<CheckBox> w |> List.find (fun b -> match b.Content with :? TextBlock as t -> t.Text.StartsWith "Agree to Meta's SAM License" | _ -> false)
    sam3.IsChecked <- Nullable true
    let training = all<CheckBox> w |> List.find (fun b -> match b.Content with :? TextBlock as t -> t.Text = labelOf "image-lora-training" | _ -> false)
    training.IsChecked <- Nullable true
    pump ()
    click w "Apply"
    let r = setup.Last
    Assert.Equal(Places.imageInstallPy, r.Script)
    for part in
        [ [ "--config"; d.Installed[0].Config ]; [ "--model"; d.Images[0].Id + "-uncensored" ]; [ "--yes" ]; [ "--gpu"; "same" ]
          [ "--sam3-license"; "agree" ]; [ "--lora-training" ] ] do
        Assert.True(contains part r.Args, sprintf "%A not in %A" part r.Args)

/// The install page passes the image options as setup's --image-<name>, SAM 3.1's license not agreed unless checked;
/// once agreed (setup's record), it is not asked again.
[<AvaloniaFact>]
let ``an install passes the image options, SAM 3.1's license only where it is not agreed yet`` () =
    english ()
    let d = described.Value
    let setup = FakeSetup(d, none)
    let w = open' setup
    installPage w |> ignore
    let training = all<CheckBox> w |> List.find (fun b -> match b.Content with :? TextBlock as t -> t.Text = labelOf "image-lora-training" | _ -> false)
    training.IsChecked <- Nullable true
    pump ()
    click w "Install"
    let args = setup.Last.Args
    for part in [ [ "--image-gpu"; "auto" ]; [ "--image-sam3-license"; "no" ]; [ "--image-lora-training" ] ] do
        Assert.True(contains part args, sprintf "%A not in %A" part args)
    let agreed = FakeSetup({ d with Sam3 = { d.Sam3 with Agreed = true } }, none)
    let w2 = open' agreed
    installPage w2 |> ignore
    Assert.False(shows w2 "The license: ", "SAM 3.1's license asked again")
    click w2 "Install"
    Assert.DoesNotContain("--image-sam3-license", agreed.Last.Args)

[<AvaloniaFact>]
let ``taking the image models out passes no`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    click w "Images ..."
    click w "Apply"
    Assert.True(contains [ "--model"; "no" ] setup.Last.Args)

[<AvaloniaFact>]
let ``tuning answers setup's question of which model`` () =
    english ()
    let d = described.Value
    let setup = FakeSetup(d, none)
    let w = open' setup
    click w "Tune"
    let r = setup.Last
    Assert.True(contains [ "--calibrate"; "--no-start" ] r.Args)
    r.Say(Ask("Tune which one? ", [ "1"; "2" ], "1"))
    pump ()
    Assert.Equal<string list>([ "1" ], List.ofSeq r.Answers)

[<AvaloniaFact>]
let ``a measurement checks its options`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    click w "Measure ..."
    let temperature = fieldOf<TextBox> w "Temperature of the sampled runs (0 to 2, 0 = greedy only)"
    temperature.Text <- "0.15"
    click w "Measure"
    Assert.True(shows w "--bench-temperature takes 0 to 2 in steps of 0.1.")
    Assert.Empty setup.Runs
    temperature.Text <- "0.5"
    (fieldOf<TextBox> w "Rounds (each case measured this many times)").Text <- "3"
    click w "Measure"
    let args = setup.Last.Args
    Assert.True(contains [ "--bench-rounds"; "3" ] args, sprintf "%A" args)
    Assert.True(contains [ "--bench-temperature"; "0.5" ] args, sprintf "%A" args)

[<AvaloniaFact>]
let ``an installed model setup cannot name does not break the pages`` () =
    // setup names an Edge0 config after its model; a config it cannot name must not reach ArgumentList as null
    english ()
    let installed = described.Value.Installed[0]
    let d = { described.Value with Installed = [| { installed with Model = null } |] }
    let setup = FakeSetup(d, none)
    let w = open' setup
    click w "Measure ..."
    click w "Measure"
    Assert.DoesNotContain("--model", setup.Last.Args)
    Assert.DoesNotContain("--family", setup.Last.Args)

[<AvaloniaFact>]
let ``upstream's benchmark mode takes its own options`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    click w "Measure ..."
    let compat = all<CheckBox> w |> List.find (fun b -> match b.Content with :? TextBlock as t -> t.Text.StartsWith "Measure the way upstream" | _ -> false)
    compat.IsChecked <- Nullable true
    pump ()
    (fieldOf<TextBox> w "Clients at once").Text <- "4"
    click w "Measure"
    let args = setup.Last.Args
    Assert.True(contains [ "--bench-original-compat" ] args && contains [ "--clients"; "4" ] args, sprintf "%A" args)

/// A start whose model the systemd user service already serves: the window shows it running there, says how to stop
/// it, and has no Stop of its own (setup's attach_running).
[<AvaloniaFact>]
let ``a model the service already serves is shown, not stopped`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    click w "Start"
    let r = setup.Last
    r.Say(Attached("http://127.0.0.1:8095", true))
    r.Say(Say "ready: http://127.0.0.1:8095/v1  (already running)")
    r.Exit 0
    pump ()
    Assert.True(shows w "Running")
    Assert.True(shows w "It runs as the systemd service (to stop it: systemctl --user stop xestrata).")
    Assert.True((findButton w "Stop").IsNone, "a Stop for the service's server")
    Assert.True((findButton w "Open chat").IsSome)

// ---------------------------------------------------------------------------------------------- Windows and Linux

let private envOf (pairs: (string * string) list) = fun (name: string) -> pairs |> List.tryFind (fst >> (=) name) |> Option.map snd |> Option.defaultValue null

[<Fact>]
let ``Linux takes the places the entry passes, and the log by XDG`` () =
    let l = Platform.layout false (envOf [ "XESTRATA_APP_DIR", "/src/x"; "XESTRATA_PYTHON", "/src/x/.venv/bin/python" ]) "/usr/lib/xestrata/launcher" "/home/u"
    Assert.Equal("/src/x", l.App)
    Assert.Equal("/src/x/.venv/bin/python", l.Python)
    Assert.Equal("/home/u/.local/state/xestrata/launcher.log", l.Log)
    Assert.Equal<string list>([ "/src/x/setup.py"; "/src/x/.venv/bin/python" ], l.Required)
    let p = Platform.layout false (envOf [ "XDG_STATE_HOME", "/s" ]) "/usr/lib/xestrata/launcher" "/home/u"
    Assert.Equal("/usr/share/xestrata", p.App)
    Assert.Equal("python3", p.Python)
    Assert.Equal("/s/xestrata/launcher.log", p.Log)
    let win = Platform.layout true (envOf [ "LOCALAPPDATA", "L" ]) "R" "H"
    Assert.Equal(IO.Path.Combine("R", "app"), win.App)
    Assert.Equal(IO.Path.Combine("L", "xestrata", "state", "launcher.log"), win.Log)

[<Fact>]
let ``Linux speaks the locale's language, and needs a screen`` () =
    Assert.Equal("ja", I18n.pick (Platform.culture false (envOf [ "LANG", "ja_JP.UTF-8" ])))
    Assert.Equal("de", I18n.pick (Platform.culture false (envOf [ "LC_ALL", "de_DE@euro"; "LANG", "ja_JP.UTF-8" ])))
    Assert.Equal("en", I18n.pick (Platform.culture false (envOf [ "LANG", "C.UTF-8" ])))
    Assert.True(Platform.noScreen false (envOf []))
    Assert.False(Platform.noScreen false (envOf [ "WAYLAND_DISPLAY", "wayland-0" ]))
    Assert.False(Platform.noScreen false (envOf [ "DISPLAY", ":0" ]))
    Assert.False(Platform.noScreen true (envOf []))

// ------------------------------------------------------------------------------------------------ the other pages

[<AvaloniaFact>]
let ``inspecting a file shows what setup read`` () =
    english ()
    let setup = FakeSetup(described.Value, function "--inspect" :: _ -> [ Ok "a GGUF of 3 tensors"; Warn "an old type" ] | _ -> [])
    let w = open' setup
    click w "Inspect a model file ..."
    (all<TextBox> w |> List.find (fun b -> b.PlaceholderText = "a file, a folder, a URL, hf:owner/repo or ms:owner/repo")).Text <- "hf:a/b"
    click w "Inspect"
    let log = waitFor "the result" (fun () -> all<TextBox> w |> List.tryFind (fun b -> b.IsReadOnly && not (isNull b.Text) && b.Text.Contains "[ok]"))
    Assert.Contains("[ok] a GGUF of 3 tensors", log.Text)
    Assert.Contains("[!] an old type", log.Text)
    Assert.Contains([ "--inspect"; "hf:a/b" ], setup.Collected)

[<AvaloniaFact>]
let ``removing the data runs setup's remove`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    click w "Remove XeStrata's data ..."
    Assert.Equal<string list>([ "--remove-data" ], setup.Last.Args)

[<AvaloniaFact>]
let ``own files need a file, and setup's refusal is shown`` () =
    english ()
    let setup = FakeSetup(described.Value, function "--describe" :: "--gguf" :: _ -> [ Fail("not a GGUF", "") ] | _ -> [])
    let w = open' setup
    click w "Install from files on this PC ..."
    click w "Next"
    Assert.True(shows w "Choose the model's .gguf file.")
    (all<TextBox> w |> List.find (fun b -> b.PlaceholderText = "the model's .gguf file (the first one of a split model)")).Text <- "C:\\m.gguf"
    click w "Next"
    waitFor "setup's refusal" (fun () -> if shows w "not a GGUF" then Some() else None)

// ------------------------------------------------------------------------------------------------- the languages

/// Every language: each page made, every family chosen, and no {placeholder} left unfilled on a page.
[<AvaloniaFact>]
let ``every page is made in every language`` () =
    let slot = Regex(@"\{\w+\}")
    let unfilled (w: Window) = texts w |> List.filter slot.IsMatch
    try
        for code, _ in I18n.languages do
            I18n.choose code
            if code <> "en" then Assert.NotEqual<string>("Install a model ...", tr "home.install" "Install a model ...")
            let setup = FakeSetup(described.Value, none)
            let w = open' setup
            let home () = click w (tr "common.back" "Back")
            let check page = Assert.True((unfilled w).IsEmpty, sprintf "%s, %s: %A" code page (unfilled w))
            check "home"
            let lv = installPage w
            everyChoice w lv (fun title -> check ("install " + title))
            home ()
            for label, page in
                [ "home.ownFiles", "Install from files on this PC ..."
                  "home.inspectAModelFile", "Inspect a model file ..."
                  "home.images", "Images ..."
                  "home.measure", "Measure ..."
                  "home.startWithOptions", "Start with options ..." ] do
                let key = if label = "home.ownFiles" then "home.installFromFilesOn" else label
                click w (tr key page)
                check page
                home ()
            w.Close()
    finally
        english ()
