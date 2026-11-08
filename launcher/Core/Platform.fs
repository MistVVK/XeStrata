// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// What differs between Windows and Linux, in one place, chosen at run time (one assembly, whose tests take both
/// branches): where the program's parts and the launcher's log are, the language the window speaks, and whether
/// there is a screen to open it on.  Each is a function of what it reads (the environment, the folder the launcher
/// runs from), so the tests give it a made-up one.
module Xestrata.Launcher.Platform

open System
open System.IO

let isWindows = OperatingSystem.IsWindows()

/// Where the program's parts are, and the files that must be there.
type Layout =
    { /// setup.py's folder (with serve/, tools/, data/ and chat.py)
      App: string
      /// the Python that runs setup.py
      Python: string
      /// the launcher's own log of the errors that stop it (#50), in the folder serve/places.py keeps the logs in
      Log: string
      Required: string list }

/// Windows: the unpacked 7z, the launcher beside app\ (XeStrata's setup.py, serve\, tools\, data\ and engine\, with
/// its PACKAGED record) and python\ (Python's embeddable distribution); the log in %LOCALAPPDATA%\xestrata\state.
/// Linux: what the entry that starts the launcher passes, XESTRATA_APP_DIR and XESTRATA_PYTHON (the packages'
/// /usr/share/xestrata and /usr/bin/python3; a checkout and its .venv/bin/python for ./setup.sh gui); the log in
/// $XDG_STATE_HOME/xestrata (~/.local/state/xestrata).
let layout (windows: bool) (env: string -> string) (baseDir: string) (home: string) =
    let get name = match env name with null -> "" | v -> v
    if windows then
        let app = Path.Combine(baseDir, "app")
        let python = Path.Combine(baseDir, "python", "python.exe")
        let local = match get "LOCALAPPDATA" with "" -> Path.Combine(home, "AppData", "Local") | d -> d
        { App = app
          Python = python
          Log = Path.Combine(local, "xestrata", "state", "launcher.log")
          Required = [ Path.Combine(app, "setup.py"); python; Path.Combine(app, "PACKAGED") ] }
    else
        let app = match get "XESTRATA_APP_DIR" with "" -> "/usr/share/xestrata" | d -> d
        let python = match get "XESTRATA_PYTHON" with "" -> "python3" | p -> p
        let state = match get "XDG_STATE_HOME" with "" -> Path.Combine(home, ".local", "state") | d -> d
        { App = app
          Python = python
          Log = Path.Combine(state, "xestrata", "launcher.log")
          Required = [ yield Path.Combine(app, "setup.py"); if Path.IsPathRooted python then yield python ] }

let current =
    layout isWindows Environment.GetEnvironmentVariable AppContext.BaseDirectory
        (Environment.GetFolderPath Environment.SpecialFolder.UserProfile)

/// The language the user asked for, as a culture name: Windows' display language; on Linux the first of LC_ALL,
/// LC_MESSAGES and LANG that is set (ja_JP.UTF-8), as gettext reads them - the launcher runs without ICU there
/// (InvariantGlobalization), whose culture would say nothing.
let culture (windows: bool) (env: string -> string) =
    if windows then
        Globalization.CultureInfo.CurrentUICulture.Name
    else
        [ "LC_ALL"; "LC_MESSAGES"; "LANG" ] |> List.tryPick (fun v -> match env v with null | "" -> None | x -> Some x)
        |> Option.defaultValue ""

/// Linux: no screen to open the window on (neither an X display nor a Wayland one: ssh alone, a server).
let noScreen (windows: bool) (env: string -> string) =
    not windows && [ "DISPLAY"; "WAYLAND_DISPLAY" ] |> List.forall (fun v -> String.IsNullOrEmpty(env v))
