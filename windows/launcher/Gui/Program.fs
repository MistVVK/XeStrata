// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// xestrata.exe: the window, after the check that the 7z is complete and the VC++ runtime is installed (said in a
/// message box of Windows' own, as the window's library may need what is missing).
module Xestrata.Launcher.Program

open System
open System.Runtime.InteropServices
open Avalonia
open Avalonia.Controls.ApplicationLifetimes
open Avalonia.Themes.Fluent

[<DllImport("user32.dll", CharSet = CharSet.Unicode)>]
extern int MessageBoxW(nativeint hWnd, string text, string caption, uint32 kind)

/// An error nothing caught, written to the launcher's log with its stack (#50: the window closed and left no trace of
/// why); `stops`: it ends the launcher, which the user is told with the log's place - once, though the window's thread
/// and then the process report the same error.
let mutable private told: obj = null

let private logError (where: string) (e: obj) (stops: bool) =
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
            let text = sprintf "XeStrata's window stopped on an error:\n%s\n\n%s" what
                           (if written then "The details are in " + path else "The details could not be written to " + path)
            MessageBoxW(0n, text, "XeStrata", 0x10u) |> ignore   // MB_ICONERROR

type App() =
    inherit Application()

    override this.Initialize() = this.Styles.Add(FluentTheme())

    override this.OnFrameworkInitializationCompleted() =
        Avalonia.Threading.Dispatcher.UIThread.UnhandledException.Add(fun e -> logError "the window's thread" e.Exception true)
        match this.ApplicationLifetime with
        | :? IClassicDesktopStyleApplicationLifetime as desktop -> desktop.MainWindow <- new MainWindow.MainWindow()
        | _ -> ()
        base.OnFrameworkInitializationCompleted()

[<EntryPoint; STAThread>]
let main argv =
    match Runtime.problem () with
    | Some why ->
        MessageBoxW(0n, why, "XeStrata", 0x10u) |> ignore   // MB_ICONERROR
        1
    | None ->
        AppDomain.CurrentDomain.UnhandledException.Add(fun e -> logError "unhandled" e.ExceptionObject e.IsTerminating)
        Threading.Tasks.TaskScheduler.UnobservedTaskException.Add(fun e -> logError "unobserved task" e.Exception false)
        AppBuilder.Configure<App>().UsePlatformDetect().StartWithClassicDesktopLifetime argv
