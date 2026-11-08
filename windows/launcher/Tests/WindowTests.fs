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

/// A window on `setup`, at its home page.
let private open' (setup: FakeSetup) =
    let w = new MainWindow.MainWindow(setup.Access)
    w.Show()
    waitFor "the home page" (fun () -> findButton w (tr "home.install" "Install a model ...")) |> ignore
    w

let private english () = I18n.choose "en"

/// The install page's two lists (the model families and the sizes).
let private installPage (w: Window) =
    click w (tr "home.install" "Install a model ...")
    waitFor "the lists" (fun () -> match all<ListBox> w with [ a; b ] -> Some(a, b) | _ -> None)

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
    let families, sizes = installPage w
    for list in [ families; sizes ] do
        let centre = list.TranslatePoint(Point(list.Bounds.Width / 2.0, list.Bounds.Height / 2.0), w)
        Assert.True(centre.HasValue, "the list is not on the page")
        for _ in 1..30 do
            w.MouseWheel(centre.Value, Vector(0.0, -1.0))
            pump ()
        for _ in 1..30 do
            w.MouseWheel(centre.Value, Vector(0.0, 1.0))
            pump ()
    families.SelectedIndex <- 0
    (families.ContainerFromIndex 0).Focus() |> ignore
    for _ in 1 .. families.ItemCount do
        press w Key.Down PhysicalKey.ArrowDown
    Assert.Equal(families.ItemCount - 1, families.SelectedIndex)
    for _ in 1 .. families.ItemCount do
        press w Key.Up PhysicalKey.ArrowUp
    Assert.Equal(0, families.SelectedIndex)

/// Every family, each of its sizes and each context the install page offers, the fields setup would ask shown.
[<AvaloniaFact>]
let ``every family, size and context can be chosen`` () =
    english ()
    let d = described.Value
    let w = open' (FakeSetup(d, none))
    let families, sizes = installPage w
    let contexts = fieldOf<ComboBox> w "Context: how much text the model sees at once (longer takes more VRAM)"
    let imagesField = fieldOf<StackPanel> w "Image generation: models to make images with, besides the text model"
    for i in 0 .. families.ItemCount - 1 do
        families.SelectedIndex <- i
        pump ()
        let f = families.SelectedItem :?> Family
        Assert.True(sizes.SelectedItem :? Model, f.Id + ": no size chosen")
        let recommended = f.Models |> Array.tryFind (fun m -> m.Recommended)
        recommended |> Option.iter (fun m -> Assert.Equal(m.Id, (sizes.SelectedItem :?> Model).Id))
        for j in 0 .. sizes.ItemCount - 1 do
            sizes.SelectedIndex <- j
            pump ()
            for k in 0 .. contexts.ItemCount - 1 do
                contexts.SelectedIndex <- k
                pump ()
        // the image models are offered for every family when the PC has a GPU (setup_plain's included)
        Assert.True(visible imagesField, f.Id + ": no image models offered")

/// The install button passes the page's choices to setup as its options, and the run's page follows setup.
[<AvaloniaFact>]
let ``an install passes the choices to setup and follows its run`` () =
    english ()
    let d = described.Value
    let setup = FakeSetup(d, none)
    let w = open' setup
    let families, sizes = installPage w
    families.SelectedIndex <- Array.findIndex (fun (f: Family) -> f.Id = "qwen") (families.ItemsSource :?> Family[])
    pump ()
    let m = sizes.SelectedItem :?> Model
    (fieldOf<TextBox> w "Port").Text <- "8100"
    (fieldOf<TextBox> w "VRAM left free for other programs, in MiB").Text <- "512"
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

[<AvaloniaFact>]
let ``an option the page cannot take is refused before setup runs`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    installPage w |> ignore
    for text, why in [ "abc", "--port takes a port number, 1 to 65535."; "70000", "--port takes a port number, 1 to 65535." ] do
        (fieldOf<TextBox> w "Port").Text <- text
        click w "Install"
        Assert.True(shows w why, text)
    (fieldOf<TextBox> w "Port").Text <- ""
    (fieldOf<TextBox> w "Requests decoded together (each takes VRAM from the expert cache)").Text <- "0"
    click w "Install"
    Assert.True(shows w "--parallel takes a number of requests, 1 or more.")
    Assert.Empty setup.Runs

[<AvaloniaFact>]
let ``a web search API needs its key`` () =
    english ()
    let setup = FakeSetup(described.Value, none)
    let w = open' setup
    installPage w |> ignore
    let search = fieldOf<ComboBox> w "Web search, for every model: the model searches when a request asks for it (the web page's Search switch, the API's web search tools)"
    search.SelectedIndex <- 1
    pump ()
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
    (fieldOf<TextBox> w "Port").Text <- "x"
    click w "Start"
    Assert.True(shows w "--port takes a port number, 1 to 65535.")
    Assert.Empty setup.Runs
    (fieldOf<TextBox> w "Port").Text <- "8200"
    (fieldOf<TextBox> w "API key").Text <- "k1"
    (fieldOf<ComboBox> w "Who may connect (with networks: set an API key too)").SelectedIndex <- 1
    click w "Start"
    let args = setup.Last.Args
    for part in [ [ "--port"; "8200" ]; [ "--host"; "0.0.0.0" ]; [ "--api-key"; "k1" ] ] do
        Assert.True(contains part args, sprintf "%A not in %A" part args)

[<AvaloniaFact>]
let ``an installed model's image models are set by install.py`` () =
    english ()
    let d = described.Value
    let setup = FakeSetup(d, none)
    let w = open' setup
    click w "Images ..."
    let first = all<CheckBox> w |> List.find (fun b -> match b.Content with :? StackPanel -> true | _ -> false)
    first.IsChecked <- Nullable true
    pump ()
    click w "Apply"
    let r = setup.Last
    Assert.Equal(Places.imageInstallPy, r.Script)
    for part in [ [ "--config"; d.Installed[0].Config ]; [ "--model"; d.Images[0].Id ]; [ "--yes" ] ] do
        Assert.True(contains part r.Args, sprintf "%A not in %A" part r.Args)

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
            let families, _ = installPage w
            for i in 0 .. families.ItemCount - 1 do
                families.SelectedIndex <- i
                pump ()
                check ("install " + string i)
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
