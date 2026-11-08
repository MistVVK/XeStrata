// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// The window (xestrata.exe on Windows, xestrata-gui on Linux), after the check that the program's files are there and,
/// on Windows, the VC++ runtime is installed: said in a message box of Windows' own (the window's library may need what
/// is missing), on Linux on the terminal, where there is also no window without a screen.
module Xestrata.Launcher.Program

open System
open System.Runtime.InteropServices
open Avalonia
open Avalonia.Controls.ApplicationLifetimes
open Avalonia.Themes.Fluent

[<DllImport("user32.dll", CharSet = CharSet.Unicode)>]
extern int MessageBoxW(nativeint hWnd, string text, string caption, uint32 kind)

/// A message that stops the launcher: Windows' message box, or the terminal on Linux.
let private tell (text: string) =
    if Platform.isWindows then MessageBoxW(0n, text, "XeStrata", 0x10u) |> ignore   // MB_ICONERROR
    else eprintfn "%s" text

/// Linux: an error of the window's own thread in a small window of Avalonia's, the launcher ended when it is closed
/// (Windows has its message box for that).
let private errorWindow (text: string) =
    let w = Avalonia.Controls.Window(Title = "XeStrata", Width = 560.0, SizeToContent = Avalonia.Controls.SizeToContent.Height)
    let ok = Views.accentButton "OK" (fun () -> w.Close())
    w.Content <- Views.stack 14.0 [ Views.coloured Views.failBrush text; Views.actions [ ok ] ]
    (w.Content :?> Avalonia.Controls.Control).Margin <- Thickness 16.0
    w.Closed.Add(fun _ -> exit 1)
    w.Show()

/// An error nothing caught, written to the launcher's log with its stack (#50: the window closed and left no trace of
/// why); `stops`: it ends the launcher, which the user is told with the log's place - once, though the window's thread
/// and then the process report the same error.
let mutable private told: obj = null

let private logError (where: string) (e: obj) (stops: bool) : string =
    if not (stops && obj.ReferenceEquals(e, told)) then
        let path = Xestrata.Launcher.Places.launcherLog
        let written =
            try
                IO.Directory.CreateDirectory(IO.Path.GetDirectoryName path) |> ignore
                IO.File.AppendAllText(path, sprintf "%s %s:\n%O\n\n" (DateTime.Now.ToString "yyyy-MM-dd HH:mm:ss") where e)
                true
            with _ -> false
        if stops then
            told <- e
            let what = match e with :? exn as x -> sprintf "%s: %s" (x.GetType().Name) x.Message | o -> string o
            let text =
                I18n.trf "error.stopped" "XeStrata's window stopped on an error:\n{what}" [ "what", box what ] + "\n\n"
                + (if written then I18n.trf "error.log" "The details are in {path}" [ "path", box path ]
                   else I18n.trf "error.noLog" "The details could not be written to {path}" [ "path", box path ])
            tell text
            text
        else null
    else null

type App() =
    inherit Application()

    override this.Initialize() = this.Styles.Add(FluentTheme())

    override this.OnFrameworkInitializationCompleted() =
        Avalonia.Threading.Dispatcher.UIThread.UnhandledException.Add(fun e ->
            if Platform.isWindows then
                logError "the window's thread" e.Exception true |> ignore
            else
                // the terminal was told (tell); the window says it too, and ends the launcher when it is closed
                match logError "the window's thread" e.Exception true with
                | null -> ()
                | text ->
                    e.Handled <- true
                    errorWindow text)
        match this.ApplicationLifetime with
        | :? IClassicDesktopStyleApplicationLifetime as desktop -> desktop.MainWindow <- new MainWindow.MainWindow()
        | _ -> ()
        base.OnFrameworkInitializationCompleted()

/// --lang CODE: the window in that language (en, de, fr, es, ja, ko, pt, ru, zh) instead of Windows' display
/// language or the locale -> the code and the other arguments.
let private language (argv: string[]) =
    match Array.tryFindIndex ((=) "--lang") argv with
    | Some i when i + 1 < argv.Length -> Some argv[i + 1], Array.append argv[.. i - 1] argv[i + 2 ..]
    | _ -> None, argv

[<EntryPoint; STAThread>]
let main argv =
    let lang, argv = language argv
    lang |> Option.iter I18n.choose
    match Runtime.problem () with
    | Some why ->
        tell why
        1
    | None when Platform.noScreen Platform.isWindows Environment.GetEnvironmentVariable ->
        // checked before Avalonia starts, which would fail on it
        tell (
            I18n.tr "screen.none" "There is no screen to open XeStrata's window on (no X or Wayland display: an ssh session, a server).\nOpen it at a desk with a screen, or use the terminal's xestrata --setup and the browser's http://<this PC>:<port>/ - they do everything the window does."
        )
        1
    | None ->
        AppDomain.CurrentDomain.UnhandledException.Add(fun e -> logError "unhandled" e.ExceptionObject e.IsTerminating |> ignore)
        Threading.Tasks.TaskScheduler.UnobservedTaskException.Add(fun e -> logError "unobserved task" e.Exception false |> ignore)
        // --software-render: the window drawn by the CPU, not through the GPU's driver (#50: a crash on scrolling on an
        // Arc 140V that no .NET handler caught, which this tells apart from one in the launcher's own code)
        let software = Array.contains "--software-render" argv
        let builder = AppBuilder.Configure<App>().UsePlatformDetect()
        let builder =
            if Platform.isWindows then
                if software then builder.With<Win32PlatformOptions>(Win32PlatformOptions(RenderingMode = [| Win32RenderingMode.Software |])) else builder
            else
                // X11 (XWayland under Wayland), with the input methods (IBus, Fcitx) for Japanese and the other
                // languages typed through one, and fontconfig's CJK fonts behind the theme's for the text it lacks
                let x11 = X11PlatformOptions(EnableIme = true)
                if software then x11.RenderingMode <- [| X11RenderingMode.Software |]
                let fonts =
                    Media.FontManagerOptions(
                        FontFallbacks = [| for f in [ "Noto Sans CJK JP"; "Noto Sans CJK SC"; "Noto Sans CJK KR"; "Noto Sans" ] -> Media.FontFallback(FontFamily = Media.FontFamily f) |])
                builder.With<X11PlatformOptions>(x11).With<Media.FontManagerOptions>(fonts)
        builder.StartWithClassicDesktopLifetime(argv |> Array.filter ((<>) "--software-render"))
